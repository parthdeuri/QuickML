package com.parthdeuri.quickml.speech

import com.parthdeuri.quickml.ui.common.stripMath

/** What the Intuition tab reads: "When not to use. … Key insight. … Why it works. …". */
fun intuitionSpeech(bruteForce: String?, insight: String, whyItWorks: String?): String {
    val parts = ArrayList<String>()
    if (!bruteForce.isNullOrBlank()) parts.add("When not to use. " + sentence(bruteForce))
    if (insight.isNotBlank()) {
        val hasLabels = !bruteForce.isNullOrBlank() || !whyItWorks.isNullOrBlank()
        parts.add(if (hasLabels) "Key insight. " + sentence(insight) else sentence(insight))
    }
    if (!whyItWorks.isNullOrBlank()) parts.add("Why it works. " + sentence(whyItWorks))
    return cleanForSpeech(parts.joinToString(" "))
}

/** What the Approach tab reads: the numbered steps, then time and space. */
fun approachSpeech(steps: List<String>, time: String?, space: String?): String {
    val parts = ArrayList<String>()
    steps.forEachIndexed { index, step ->
        parts.add("Step " + (index + 1) + ". " + sentence(step))
    }
    if (!time.isNullOrBlank()) parts.add("Time: " + sentence(time))
    if (!space.isNullOrBlank()) parts.add("Space: " + sentence(space))
    return cleanForSpeech(parts.joinToString(" "))
}

/** Ends [text] with a full stop so the engine pauses between parts. */
private fun sentence(text: String): String {
    val t = text.trim()
    if (t.isEmpty()) return t
    return if (t.last() in ".!?:") t else "$t."
}

private val Replacements = listOf(
    "→" to " to ",
    "ŷ" to " y hat ",
    "∇" to " gradient of ",
    "‖" to " norm of ",
    "ᵀ" to " transpose ",
    "∑" to " sum of ",
    "σ" to " sigma ",
    "->" to " to ",
    "←" to " from ",
    "<=" to " at most ",
    "≤" to " at most ",
    ">=" to " at least ",
    "≥" to " at least ",
    "!=" to " not equal to ",
    "≠" to " not equal to ",
    "==" to " equals ",
    "+=" to " plus equals ",
    "-=" to " minus equals ",
    "//" to " integer divided by ",
    " < " to " less than ",
    " > " to " greater than ",
    " = " to " equals ",
    " % " to " mod ",
    " / " to " over ",
    "^2" to " squared",
    "^3" to " cubed",
    "^" to " to the power ",
    "*" to " times ",
    "×" to " times ",
    "·" to " times ",
    "√" to " square root of ",
    "∞" to " infinity ",
    "—" to ", ",
    "–" to ", ",
    "…" to ". ",
    "`" to "",
    "_" to " ",
    "#" to " hash ",
    "[" to " ",
    "]" to " ",
    "{" to " ",
    "}" to " ",
    "|" to " "
)

private val Spaces = Regex("[ \\t]+")
private val SpaceBeforePunctuation = Regex(" +([,.;:?!])")

/** Turns symbols the TTS engine reads badly into words ("O(n)" → "O of n", "≤" → "at most"). */
fun cleanForSpeech(text: String): String {
    var s = speakBigO(stripMath(text))
    for ((target, replacement) in Replacements) {
        s = s.replace(target, replacement)
    }
    s = s.replace('\n', ' ')
    s = Spaces.replace(s, " ")
    s = SpaceBeforePunctuation.replace(s, "\$1")
    return s.trim()
}

/** "O(n log n)" → "O of n log n"; nested parentheses inside the O are kept balanced. */
private fun speakBigO(text: String): String {
    val out = StringBuilder()
    var i = 0
    while (i < text.length) {
        val isBigO = text[i] == 'O' &&
            i + 1 < text.length && text[i + 1] == '(' &&
            (i == 0 || !text[i - 1].isLetterOrDigit())
        if (!isBigO) {
            out.append(text[i])
            i++
            continue
        }
        var depth = 0
        var end = -1
        var j = i + 1
        while (j < text.length) {
            if (text[j] == '(') depth++
            if (text[j] == ')') {
                depth--
                if (depth == 0) {
                    end = j
                    break
                }
            }
            j++
        }
        if (end < 0) {
            out.append(text[i])
            i++
            continue
        }
        val inner = text.substring(i + 2, end)
            .replace("!", " factorial")
            .replace("(", " of ")
            .replace(")", " ")
        out.append("O of ").append(inner)
        i = end + 1
    }
    return out.toString()
}

/** Splits [text] at sentence boundaries into pieces of at most [maxLength] characters. */
fun splitForSpeech(text: String, maxLength: Int): List<String> {
    val clean = text.trim()
    if (clean.isEmpty()) return emptyList()
    if (clean.length <= maxLength) return listOf(clean)
    val chunks = ArrayList<String>()
    var rest = clean
    while (rest.length > maxLength) {
        var cut = rest.lastIndexOf(". ", maxLength)
        if (cut <= 0) cut = rest.lastIndexOf(' ', maxLength)
        if (cut <= 0) cut = maxLength - 1
        chunks.add(rest.substring(0, cut + 1).trim())
        rest = rest.substring(cut + 1).trim()
    }
    if (rest.isNotEmpty()) chunks.add(rest)
    return chunks.filter { it.isNotEmpty() }
}
