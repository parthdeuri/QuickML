package com.parthdeuri.quickml.data.content

import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonPrimitive

@Serializable
data class ContentFile(
    val contentVersion: Int,
    val problems: List<ProblemJson> = emptyList(),
    val sheets: List<SheetJson> = emptyList(),
    /**
     * The pattern library (List<PatternJson>), kept raw so that a missing or malformed list
     * never breaks the problem import; decoded item by item in ContentImporter.
     */
    val patterns: JsonElement? = null
)

/** One pattern of the library (top-level "patterns" in problems.json). */
@Serializable
data class PatternJson(
    val id: String,
    val name: String = "",
    val summary: String = "",
    val whenToUse: List<String> = emptyList(),
    val keyIdea: String = "",
    /** A string with newlines (a list of lines is accepted too). */
    val template: JsonElement? = null,
    val complexity: String = "",
    val pitfalls: List<String> = emptyList(),
    val problems: List<String> = emptyList(),
    val aliases: List<String> = emptyList()
) {
    val templateText: String
        get() = when (val t = template) {
            null, is JsonNull -> ""
            is JsonPrimitive -> t.content
            is JsonArray -> t.joinToString("\n") { line ->
                if (line is JsonPrimitive && line !is JsonNull) line.content else ""
            }
            else -> ""
        }
}

@Serializable
data class ProblemJson(
    val slug: String,
    val title: String,
    val platform: String = "leetcode",
    val url: String,
    val difficulty: String,
    val topics: List<String> = emptyList(),
    val patterns: List<String> = emptyList(),
    val summary: String? = null,
    val intuition: String? = null,
    val approach: List<String>? = null,
    val time: String? = null,
    val space: String? = null,
    val solution: String? = null,
    val edgeCases: List<String> = emptyList(),
    val related: List<String> = emptyList(),
    val description: String? = null,
    val examples: List<ExampleJson> = emptyList(),
    val constraints: List<String> = emptyList(),
    val bruteForce: String? = null,
    val whyItWorks: String? = null,
    val dryRun: List<String> = emptyList(),
    /** 1-based line numbers in [solution] that hold the core idea (empty for metadata-only). */
    val keyLines: List<Int> = emptyList()
)

@Serializable
data class ExampleJson(
    val input: String,
    val output: String,
    val explanation: String = ""
)

/** Encodes/decodes the JSON strings stored in ProblemEntity. Decoding never throws. */
object ContentJson {
    private val json = Json {
        ignoreUnknownKeys = true
        coerceInputValues = true
    }
    private val examplesSerializer = ListSerializer(ExampleJson.serializer())
    private val stringsSerializer = ListSerializer(String.serializer())
    private val intsSerializer = ListSerializer(Int.serializer())

    fun encodeExamples(items: List<ExampleJson>): String? =
        if (items.isEmpty()) null else json.encodeToString(examplesSerializer, items)

    fun encodeConstraints(items: List<String>): String? = encodeStrings(items)

    fun encodeStrings(items: List<String>): String? =
        if (items.isEmpty()) null else json.encodeToString(stringsSerializer, items)

    fun decodeStrings(text: String?): List<String> = decodeConstraints(text)

    /** Like [encodeStrings] but never null ("[]" for an empty list), for NOT NULL columns. */
    fun encodeStringList(items: List<String>): String =
        json.encodeToString(stringsSerializer, items)

    fun encodeInts(items: List<Int>): String? =
        if (items.isEmpty()) null else json.encodeToString(intsSerializer, items)

    fun decodeInts(text: String?): List<Int> {
        if (text.isNullOrBlank()) return emptyList()
        return try {
            json.decodeFromString(intsSerializer, text)
        } catch (e: Exception) {
            emptyList()
        }
    }

    fun decodeExamples(text: String?): List<ExampleJson> {
        if (text.isNullOrBlank()) return emptyList()
        return try {
            json.decodeFromString(examplesSerializer, text)
        } catch (e: Exception) {
            emptyList()
        }
    }

    fun decodeConstraints(text: String?): List<String> {
        if (text.isNullOrBlank()) return emptyList()
        return try {
            json.decodeFromString(stringsSerializer, text)
        } catch (e: Exception) {
            emptyList()
        }
    }
}

@Serializable
data class SheetJson(
    val id: String,
    val name: String,
    val description: String = "",
    val entries: List<SheetEntryJson> = emptyList()
)

@Serializable
data class SheetEntryJson(
    val slug: String,
    val section: String,
    val order: Int
)
