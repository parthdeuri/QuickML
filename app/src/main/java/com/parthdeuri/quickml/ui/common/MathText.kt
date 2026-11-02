package com.parthdeuri.quickml.ui.common

import android.graphics.Bitmap
import android.graphics.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.text.InlineTextContent
import androidx.compose.foundation.text.appendInlineContent
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.Placeholder
import androidx.compose.ui.text.PlaceholderVerticalAlign
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.sp
import ru.noties.jlatexmath.JLatexMathDrawable

/** A piece of card text: plain words, or one LaTeX formula that was written between dollar signs. */
private class MathPart(val value: String, val isMath: Boolean)

private val MATH_REGEX = Regex("\\$([^$]+)\\$")

private fun splitMath(text: String): List<MathPart> {
    val parts = ArrayList<MathPart>()
    var last = 0
    for (m in MATH_REGEX.findAll(text)) {
        if (m.range.first > last) parts.add(MathPart(text.substring(last, m.range.first), false))
        parts.add(MathPart(m.groupValues[1].trim(), true))
        last = m.range.last + 1
    }
    if (last < text.length) parts.add(MathPart(text.substring(last), false))
    return parts
}

/** Draws one formula to a bitmap sized for [sizePx] text; null if the formula cannot be typeset. */
private fun renderFormula(latex: String, sizePx: Float, argb: Int): ImageBitmap? = try {
    val drawable = JLatexMathDrawable.builder(latex)
        .textSize(sizePx)
        .color(argb)
        .build()
    val w = drawable.intrinsicWidth
    val h = drawable.intrinsicHeight
    if (w <= 0 || h <= 0) {
        null
    } else {
        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        drawable.setBounds(0, 0, w, h)
        drawable.draw(Canvas(bitmap))
        bitmap.asImageBitmap()
    }
} catch (e: Throwable) {
    null
}

/**
 * Text that typesets LaTeX written between dollar signs inline with the words around it.
 * Text without a dollar sign is a plain [Text]; a formula that fails to render is shown as its source.
 */
@Composable
fun MathText(
    text: String,
    modifier: Modifier = Modifier,
    style: TextStyle = LocalTextStyle.current,
    color: Color = Color.Unspecified,
    lineHeight: TextUnit = TextUnit.Unspecified,
    fontWeight: FontWeight? = null
) {
    if (!text.contains('$')) {
        Text(
            text = text,
            modifier = modifier,
            style = style,
            color = color,
            lineHeight = lineHeight,
            fontWeight = fontWeight
        )
        return
    }
    val density = LocalDensity.current
    val sizeSp = if (style.fontSize == TextUnit.Unspecified) 16.sp else style.fontSize
    val sizePx = with(density) { sizeSp.toPx() }
    val tint = when {
        color != Color.Unspecified -> color
        style.color != Color.Unspecified -> style.color
        else -> LocalContentColor.current
    }
    val argb = tint.toArgb()
    val parts = remember(text) { splitMath(text) }
    val bitmaps = remember(text, sizePx, argb) {
        parts.map { if (it.isMath) renderFormula(it.value, sizePx * 1.05f, argb) else null }
    }
    val annotated = buildAnnotatedString {
        parts.forEachIndexed { i, part ->
            if (part.isMath && bitmaps[i] != null) {
                appendInlineContent("m$i", part.value)
            } else {
                append(part.value)
            }
        }
    }
    val inline = HashMap<String, InlineTextContent>()
    parts.forEachIndexed { i, part ->
        val bitmap = bitmaps[i]
        if (part.isMath && bitmap != null) {
            inline["m$i"] = InlineTextContent(
                Placeholder(
                    width = with(density) { bitmap.width.toSp() },
                    height = with(density) { bitmap.height.toSp() },
                    placeholderVerticalAlign = PlaceholderVerticalAlign.TextCenter
                )
            ) { alt ->
                Image(
                    bitmap = bitmap,
                    contentDescription = alt,
                    modifier = Modifier
                )
            }
        }
    }
    Text(
        text = annotated,
        modifier = modifier,
        style = style,
        color = color,
        lineHeight = lineHeight,
        fontWeight = fontWeight,
        inlineContent = inline
    )
}

/** Plain-text version of card text: used for speech and the Ask prompt where pictures do not help. */
fun stripMath(text: String): String {
    if (!text.contains('$')) return text
    return MATH_REGEX.replace(text) { m ->
        var s = m.groupValues[1]
        s = s.replace(Regex("\\\\frac\\{([^{}]*)\\}\\{([^{}]*)\\}"), "$1 over $2")
        s = s.replace(Regex("\\\\sqrt\\{([^{}]*)\\}"), "square root of $1")
        s = s.replace("\\sum", "sum").replace("\\prod", "product").replace("\\cdot", " times ")
        s = s.replace("\\leq", " at most ").replace("\\geq", " at least ").replace("\\neq", " not equal to ")
        s = s.replace("\\approx", " approximately ").replace("\\infty", "infinity")
        s = s.replace(Regex("\\\\([a-zA-Z]+)"), "$1").replace("{", "").replace("}", "")
        s.replace("^", " to the power ").replace("_", " sub ").replace("\\", "").trim()
    }
}
