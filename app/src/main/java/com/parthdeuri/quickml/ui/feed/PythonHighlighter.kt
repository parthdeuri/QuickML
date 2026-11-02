package com.parthdeuri.quickml.ui.feed

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.withStyle
import com.parthdeuri.quickml.ui.theme.CodePalette

/** Tiny hand-written Python tokenizer; colours come from the active theme's [CodePalette]. */
object PythonHighlighter {
    private val keywords = setOf(
        "False", "None", "True", "and", "as", "assert", "async", "await", "break",
        "class", "continue", "def", "del", "elif", "else", "except", "finally",
        "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
        "not", "or", "pass", "raise", "return", "try", "while", "with", "yield"
    )

    private val builtins = setOf(
        "abs", "all", "any", "bin", "bool", "bytes", "chr", "dict", "divmod",
        "enumerate", "filter", "float", "frozenset", "getattr", "hash", "hex",
        "int", "isinstance", "iter", "len", "list", "map", "max", "min", "next",
        "object", "ord", "pow", "print", "range", "repr", "reversed", "round",
        "set", "sorted", "str", "sum", "super", "tuple", "type", "zip",
        "Counter", "defaultdict", "deque", "heapq", "heappush", "heappop",
        "List", "Optional", "Dict", "Set", "Tuple"
    )

    private const val STRING_PREFIX_CHARS = "rRbBfFuU"

    fun highlight(src: String, palette: CodePalette): AnnotatedString = buildAnnotatedString {
        val n = src.length
        var i = 0
        // 0 = none, 1 = expecting a function name, 2 = expecting a class name
        var expectName = 0

        fun span(text: String, color: Color, italic: Boolean = false) {
            withStyle(
                SpanStyle(
                    color = color,
                    fontStyle = if (italic) FontStyle.Italic else null
                )
            ) { append(text) }
        }

        while (i < n) {
            val c = src[i]
            when {
                c == '#' -> {
                    var end = src.indexOf('\n', i)
                    if (end < 0) end = n
                    span(src.substring(i, end), palette.comment, italic = true)
                    i = end
                }

                c == '"' || c == '\'' -> {
                    val end = stringEnd(src, i)
                    span(src.substring(i, end), palette.string)
                    i = end
                    expectName = 0
                }

                c.isDigit() || (c == '.' && i + 1 < n && src[i + 1].isDigit()) -> {
                    var end = i + 1
                    while (end < n && (src[end].isLetterOrDigit() || src[end] == '.' || src[end] == '_')) {
                        end++
                    }
                    span(src.substring(i, end), palette.number)
                    i = end
                    expectName = 0
                }

                c == '@' && isLineStart(src, i) -> {
                    var end = i + 1
                    while (end < n && (src[end].isLetterOrDigit() || src[end] == '_' || src[end] == '.')) {
                        end++
                    }
                    span(src.substring(i, end), palette.decorator)
                    i = end
                    expectName = 0
                }

                c.isLetter() || c == '_' -> {
                    var end = i + 1
                    while (end < n && (src[end].isLetterOrDigit() || src[end] == '_')) {
                        end++
                    }
                    val word = src.substring(i, end)
                    val isStringPrefix = word.length <= 2 &&
                        word.all { STRING_PREFIX_CHARS.indexOf(it) >= 0 } &&
                        end < n && (src[end] == '"' || src[end] == '\'')
                    if (isStringPrefix) {
                        val strEnd = stringEnd(src, end)
                        span(src.substring(i, strEnd), palette.string)
                        i = strEnd
                        expectName = 0
                    } else {
                        when {
                            expectName == 1 -> span(word, palette.function)
                            expectName == 2 -> span(word, palette.className)
                            word == "self" || word == "cls" -> span(word, palette.self)
                            word in keywords -> span(word, palette.keyword)
                            word in builtins -> span(word, palette.builtin)
                            else -> append(word)
                        }
                        expectName = when {
                            expectName != 0 -> 0
                            word == "def" -> 1
                            word == "class" -> 2
                            else -> 0
                        }
                        i = end
                    }
                }

                else -> {
                    append(c)
                    if (!c.isWhitespace()) expectName = 0
                    i++
                }
            }
        }
    }

    /** True if only whitespace precedes [index] on its line. */
    private fun isLineStart(s: String, index: Int): Boolean {
        var j = index - 1
        while (j >= 0 && s[j] != '\n') {
            if (!s[j].isWhitespace()) return false
            j--
        }
        return true
    }

    /** Returns the exclusive end index of the string literal starting at [start] (a quote char). */
    private fun stringEnd(s: String, start: Int): Int {
        val quote = s[start]
        val triple = start + 2 < s.length && s[start + 1] == quote && s[start + 2] == quote
        val tripleQuote = "$quote$quote$quote"
        var i = start + if (triple) 3 else 1
        while (i < s.length) {
            val ch = s[i]
            if (ch == '\\') {
                i += 2
                continue
            }
            if (triple) {
                if (ch == quote && s.startsWith(tripleQuote, i)) return i + 3
            } else {
                if (ch == quote) return i + 1
                if (ch == '\n') return i
            }
            i++
        }
        return s.length
    }
}
