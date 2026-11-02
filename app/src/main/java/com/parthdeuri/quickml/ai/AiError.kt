package com.parthdeuri.quickml.ai

/** Failure of an AI request; [message] is always a short text that can be shown to the user. */
sealed class AiError(message: String) : Exception(message) {
    /** No API key saved yet. */
    object NoKey : AiError("Add your Gemini key in Settings first")

    /** HTTP 400 (API key), 401 or 403. */
    object KeyRejected : AiError("Key rejected, check it in Settings")

    /** HTTP 429. */
    object RateLimited : AiError("Daily or per-minute limit reached, try later")

    /** The model name does not exist (HTTP 404) or is malformed. */
    object ModelNotFound : AiError("Model not found, check the name in Settings")

    /** I/O failure: offline, timeout, DNS. */
    object Network : AiError("No connection")

    /** The stream ended without any text (safety block, empty candidate). */
    object Blocked : AiError("No answer (blocked by the model)")

    /** Any other HTTP status (or an in-stream error, [code] 0) with the server text. */
    class Server(val code: Int, val detail: String?) : AiError(serverMessage(code, detail))

    /** Unexpected failure while reading the response. */
    object Unknown : AiError("Something went wrong, try again")
}

private fun serverMessage(code: Int, detail: String?): String {
    val extra = if (detail.isNullOrBlank()) "" else ": $detail"
    return when {
        code >= 500 -> "Gemini is unavailable right now, try again"
        code > 0 -> "Request failed ($code)$extra"
        else -> "Request failed$extra"
    }
}
