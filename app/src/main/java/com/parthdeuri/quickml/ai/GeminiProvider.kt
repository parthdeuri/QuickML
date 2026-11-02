package com.parthdeuri.quickml.ai

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.atomic.AtomicReference

private val sseJson = Json { ignoreUnknownKeys = true }

private fun parseObject(text: String): JsonObject? = try {
    sseJson.parseToJsonElement(text) as? JsonObject
} catch (e: Exception) {
    null
}

/**
 * Text delta of one SSE line of `streamGenerateContent?alt=sse`, or null when the line carries
 * none (blank, comment, finishReason-only chunk, empty candidates, error object, bad JSON).
 * Concatenates all `candidates[0].content.parts[*].text`.
 */
fun parseGeminiSseLine(line: String): String? {
    val trimmed = line.trim()
    if (!trimmed.startsWith("data:")) return null
    val obj = parseObject(trimmed.removePrefix("data:").trim()) ?: return null
    val candidates = obj["candidates"] as? JsonArray ?: return null
    val first = candidates.firstOrNull() as? JsonObject ?: return null
    val content = first["content"] as? JsonObject ?: return null
    val parts = content["parts"] as? JsonArray ?: return null
    val sb = StringBuilder()
    for (part in parts) {
        val text = ((part as? JsonObject)?.get("text") as? JsonPrimitive)?.contentOrNull
        if (text != null) sb.append(text)
    }
    return if (sb.isEmpty()) null else sb.toString()
}

/**
 * `error.message` of an SSE line or of a whole error body (JSON object with an `error` object),
 * or null when [text] carries no error.
 */
fun parseGeminiErrorMessage(text: String): String? {
    val trimmed = text.trim()
    val payload = if (trimmed.startsWith("data:")) trimmed.removePrefix("data:").trim() else trimmed
    val error = parseObject(payload)?.get("error") as? JsonObject ?: return null
    val message = (error["message"] as? JsonPrimitive)?.contentOrNull
    return if (message.isNullOrBlank()) "unknown error" else message
}

/**
 * Gemini REST client (no SDK): `streamGenerateContent?alt=sse`, key in the `x-goog-api-key`
 * header. Key and model are read from [settings] on every request; nothing happens until the
 * returned flow is collected.
 */
class GeminiProvider(private val settings: AiSettings) : AiProvider {

    override fun ask(system: String, history: List<ChatTurn>): Flow<String> = channelFlow {
        val key = settings.getKey() ?: throw AiError.NoKey
        val model = settings.model.value.trim().removePrefix("models/")
        if (!MODEL_REGEX.matches(model)) throw AiError.ModelNotFound
        val body = buildBody(system, history).toString().toByteArray(Charsets.UTF_8)
        val connRef = AtomicReference<HttpURLConnection?>(null)

        launch(Dispatchers.IO) {
            try {
                val url = URL("$BASE_URL$model:streamGenerateContent?alt=sse")
                val conn = url.openConnection() as HttpURLConnection
                connRef.set(conn)
                conn.requestMethod = "POST"
                conn.connectTimeout = CONNECT_TIMEOUT_MS
                conn.readTimeout = READ_TIMEOUT_MS
                conn.doOutput = true
                conn.setRequestProperty("Content-Type", "application/json")
                conn.setRequestProperty("x-goog-api-key", key)
                conn.outputStream.use { it.write(body) }

                val code = conn.responseCode
                if (code !in 200..299) {
                    val text = try {
                        val stream = conn.errorStream ?: conn.inputStream
                        stream.bufferedReader(Charsets.UTF_8).use { it.readText() }
                    } catch (e: IOException) {
                        ""
                    }
                    throw mapHttpError(code, parseGeminiErrorMessage(text))
                }

                var emitted = false
                conn.inputStream.bufferedReader(Charsets.UTF_8).use { reader ->
                    var line = reader.readLine()
                    while (line != null) {
                        if (line.contains("\"error\"")) {
                            val err = parseGeminiErrorMessage(line)
                            if (err != null) throw AiError.Server(0, err)
                        }
                        val delta = parseGeminiSseLine(line)
                        if (delta != null) {
                            emitted = true
                            send(delta)
                        }
                        line = reader.readLine()
                    }
                }
                if (!emitted) throw AiError.Blocked
                close()
            } catch (e: CancellationException) {
                throw e
            } catch (e: AiError) {
                close(e)
            } catch (e: IOException) {
                close(AiError.Network)
            } catch (e: Exception) {
                close(AiError.Unknown)
            }
        }

        // Runs when the collector cancels or after close(); disconnect() also unblocks a read.
        awaitClose { connRef.get()?.disconnect() }
    }

    private fun buildBody(system: String, history: List<ChatTurn>): JsonObject = buildJsonObject {
        putJsonObject("systemInstruction") {
            putJsonArray("parts") {
                add(buildJsonObject { put("text", system) })
            }
        }
        put("contents", buildJsonArray {
            for (turn in history.takeLast(MAX_TURNS)) {
                add(buildJsonObject {
                    put("role", if (turn.role == ChatTurn.ROLE_MODEL) "model" else "user")
                    putJsonArray("parts") {
                        add(buildJsonObject { put("text", turn.text) })
                    }
                })
            }
        })
    }

    private fun mapHttpError(code: Int, message: String?): AiError = when {
        code == 401 || code == 403 -> AiError.KeyRejected
        code == 400 && message?.contains("API key", ignoreCase = true) == true ->
            AiError.KeyRejected
        code == 404 -> AiError.ModelNotFound
        code == 429 -> AiError.RateLimited
        else -> AiError.Server(code, message)
    }

    private companion object {
        const val BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
        const val CONNECT_TIMEOUT_MS = 15_000
        const val READ_TIMEOUT_MS = 60_000
        const val MAX_TURNS = 12
        val MODEL_REGEX = Regex("[A-Za-z0-9._-]+")
    }
}
