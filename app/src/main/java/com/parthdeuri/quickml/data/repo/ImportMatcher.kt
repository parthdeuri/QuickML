package com.parthdeuri.quickml.data.repo

import com.parthdeuri.quickml.data.db.ConceptEntity

/** Result of matching pasted lines against the problems ("Import solved problems"). */
data class ImportPreview(
    /** Matched problems that are not solved yet, in input order (these get marked solved). */
    val toMark: List<String>,
    /** Matched problems that are already solved. */
    val alreadySolved: Int,
    /** Input lines / items that matched nothing, as typed. */
    val notFound: List<String>
)

/**
 * Matches LeetCode / GFG URLs, slugs or exact titles (case-insensitive) against the problems.
 * Lines are tried whole first (titles like "Pow(x, n)" contain commas), then split on commas.
 */
object ImportMatcher {
    private val nonAlnum = Regex("[^a-z0-9]+")
    private val numberPrefix = Regex("^\\d+\\s*[.)]\\s*")

    fun match(text: String, problems: List<ConceptEntity>, solved: Set<String>): ImportPreview {
        val index = buildIndex(problems)
        val matched = LinkedHashSet<String>()
        val notFound = ArrayList<String>()
        for (rawLine in text.lines()) {
            val line = rawLine.trim()
            if (line.isEmpty()) continue
            val whole = resolve(line, index)
            if (whole != null) {
                matched.add(whole)
                continue
            }
            val parts = line.split(',').map { it.trim() }.filter { it.isNotEmpty() }
            if (parts.size <= 1) {
                notFound.add(line)
                continue
            }
            for (part in parts) {
                val slug = resolve(part, index)
                if (slug != null) matched.add(slug) else notFound.add(part)
            }
        }
        return ImportPreview(
            toMark = matched.filter { it !in solved },
            alreadySolved = matched.count { it in solved },
            notFound = notFound.distinct()
        )
    }

    private fun buildIndex(problems: List<ConceptEntity>): Map<String, String> {
        val index = HashMap<String, String>()
        fun put(key: String, slug: String) {
            if (key.isNotEmpty() && key !in index) index[key] = slug
        }
        // Slugs first so they win over a title that happens to slugify the same way.
        for (p in problems) put(p.slug.lowercase(), p.slug)
        for (p in problems) {
            put(p.title.trim().lowercase(), p.slug)
            put(slugify(p.title), p.slug)
            val url = p.url.lowercase()
            urlKey(url)?.let { put(it, p.slug) }
            problemSegment(url)?.let { put(SEGMENT_PREFIX + it, p.slug) }
        }
        return index
    }

    private fun resolve(token: String, index: Map<String, String>): String? {
        val t = token.trim().trim('"', '\'', '`', '<', '>').trim()
        if (t.isEmpty()) return null
        val lower = t.lowercase()
        if (looksLikeUrl(lower)) {
            urlKey(lower)?.let { key -> index[key]?.let { return it } }
            problemSegment(lower)?.let { segment ->
                index[SEGMENT_PREFIX + segment]?.let { return it }
                index[segment]?.let { return it }
                index[slugify(segment)]?.let { return it }
            }
            return null
        }
        index[lower]?.let { return it }
        index[slugify(lower)]?.let { return it }
        // "1. Two Sum" (as copied from a problem list).
        val withoutNumber = lower.replace(numberPrefix, "")
        if (withoutNumber != lower) {
            index[withoutNumber]?.let { return it }
            index[slugify(withoutNumber)]?.let { return it }
        }
        return null
    }

    private fun looksLikeUrl(lower: String): Boolean =
        lower.startsWith("http://") || lower.startsWith("https://") || lower.startsWith("www.") ||
            lower.contains(".com/") || lower.contains(".org/") || lower.contains(".cn/")

    /** host + path without scheme, "www.", query, fragment or trailing slashes. */
    private fun urlKey(lower: String): String? {
        val noScheme = lower.substringAfter("://")
        val noQuery = noScheme.substringBefore('?').substringBefore('#')
        val key = noQuery.removePrefix("www.").trimEnd('/')
        return key.ifEmpty { null }
    }

    /** The path segment after "/problems/" (LeetCode slug, GFG problem name). */
    private fun problemSegment(lower: String): String? {
        val path = lower.substringBefore('?').substringBefore('#')
        val marker = "/problems/"
        val start = path.indexOf(marker)
        if (start < 0) return null
        val segment = path.substring(start + marker.length).substringBefore('/')
        return segment.ifEmpty { null }
    }

    fun slugify(text: String): String =
        text.lowercase().replace(nonAlnum, "-").trim('-')

    private const val SEGMENT_PREFIX = "seg:"
}
