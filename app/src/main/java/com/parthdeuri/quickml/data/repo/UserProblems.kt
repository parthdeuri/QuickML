package com.parthdeuri.quickml.data.repo

import com.parthdeuri.quickml.data.db.ProblemEntity

/** The add / edit form of an own problem; every field is plain text as typed. */
data class UserProblemDraft(
    val title: String = "",
    val link: String = "",
    val difficulty: String = "Medium",
    /** Comma-separated. */
    val topics: String = "",
    val description: String = "",
    val intuition: String = "",
    /** One step per line. */
    val approach: String = "",
    val solution: String = "",
    val time: String = "",
    val space: String = "",
    val notes: String = ""
) {
    val topicList: List<String>
        get() = topics.split(',').map { it.trim() }.filter { it.isNotEmpty() }.distinct()

    val isValid: Boolean
        get() = title.isNotBlank()
}

val USER_DIFFICULTIES = listOf("Easy", "Medium", "Hard")

/** leetcode / gfg / codingninjas from the link's domain, otherwise "other". */
fun detectPlatform(url: String): String {
    val host = url.trim().lowercase()
        .substringAfter("://")
        .substringBefore('/')
    return when {
        host.isEmpty() -> "other"
        "leetcode" in host -> "leetcode"
        "geeksforgeeks" in host -> "gfg"
        "naukri" in host || "codingninjas" in host -> "codingninjas"
        else -> "other"
    }
}

/** Adds https:// to a bare domain; blank stays blank. */
fun normalizeLink(link: String): String {
    val trimmed = link.trim()
    if (trimmed.isEmpty()) return ""
    return if ("://" in trimmed) trimmed else "https://$trimmed"
}

private fun String.orNullIfBlank(): String? = trim().takeIf { it.isNotEmpty() }

fun UserProblemDraft.toEntity(slug: String): ProblemEntity {
    val url = normalizeLink(link)
    return ProblemEntity(
        slug = slug,
        title = title.trim(),
        platform = detectPlatform(url),
        url = url,
        difficulty = difficulty.takeIf { it in USER_DIFFICULTIES } ?: "Medium",
        topics = topicList,
        patterns = emptyList(),
        summary = null,
        intuition = intuition.orNullIfBlank(),
        approach = approach.lines().map { it.trim() }.filter { it.isNotEmpty() },
        time = time.orNullIfBlank(),
        space = space.orNullIfBlank(),
        solution = solution.trimEnd().takeIf { it.isNotBlank() },
        edgeCases = emptyList(),
        related = emptyList(),
        description = description.orNullIfBlank(),
        userAdded = true
    )
}

fun ProblemEntity.toDraft(note: String?): UserProblemDraft = UserProblemDraft(
    title = title,
    link = url,
    difficulty = difficulty,
    topics = topics.joinToString(", "),
    description = description.orEmpty(),
    intuition = intuition.orEmpty(),
    approach = approach.joinToString("\n"),
    solution = solution.orEmpty(),
    time = time.orEmpty(),
    space = space.orEmpty(),
    notes = note.orEmpty()
)

/** Creation time encoded in an own problem's slug ("user-<epochMillis>"), for newest-first. */
fun userProblemTime(slug: String): Long =
    slug.removePrefix(ProblemRepository.USER_SLUG_PREFIX).toLongOrNull() ?: 0L
