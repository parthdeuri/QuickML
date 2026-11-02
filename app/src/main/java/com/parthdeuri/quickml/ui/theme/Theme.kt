package com.parthdeuri.quickml.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.ReadOnlyComposable
import androidx.compose.runtime.remember
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import com.parthdeuri.quickml.data.settings.AccentChoice
import com.parthdeuri.quickml.domain.srs.Rating

/** Python syntax colours and the background of code boxes. */
@Immutable
data class CodePalette(
    val background: Color,
    /** Plain code, dry-run lines and example values. */
    val text: Color,
    val lineNumber: Color,
    val keyword: Color,
    val builtin: Color,
    val string: Color,
    val comment: Color,
    val number: Color,
    val decorator: Color,
    val self: Color,
    val function: Color,
    val className: Color
)

/**
 * App colours beyond the Material scheme, one set per theme. Read them with
 * `QuickMLTheme.colors`; text stays in the scheme's on-surface colours unless it is a deliberate
 * label (links, selected states, pills, the "Key insight" title).
 */
@Immutable
data class QuickMLColors(
    val isDark: Boolean,
    /** Brand teal: selected tabs, icons, links, progress bars. */
    val accent: Color,
    /** Cards, tiles and settings groups (one step above the background). */
    val card: Color,
    val easy: Color,
    val medium: Color,
    val hard: Color,
    /** Difficulty colour of an unknown label. */
    val neutral: Color,
    val ratingForgot: Color,
    val ratingHard: Color,
    val ratingGood: Color,
    val ratingEasy: Color,
    /** Correct / wrong answers of the pattern quiz, "Solved" pill, destructive menu items. */
    val success: Color,
    val danger: Color,
    /** Double-tap bookmark pop. */
    val heart: Color,
    /** Activity heatmap levels 0..4: no activity (close to the surface) up to the busiest days. */
    val heatmap: List<Color>,
    /** Recessive chart gridlines. */
    val chartGrid: Color,
    val code: CodePalette
) {
    fun difficulty(label: String, fallback: Color = neutral): Color = when (label) {
        "Easy" -> easy
        "Medium" -> medium
        "Hard" -> hard
        else -> fallback
    }

    fun rating(rating: Rating): Color = when (rating) {
        Rating.FORGOT -> ratingForgot
        Rating.HARD -> ratingHard
        Rating.GOOD -> ratingGood
        Rating.EASY -> ratingEasy
    }

    /**
     * [color] used as text on its own translucent tint (pills, chips, quiz answers): unchanged in
     * dark, one step darker in light so small labels keep their contrast.
     */
    fun onTint(color: Color): Color = if (isDark) color else lerp(color, Color.Black, 0.2f)
}

// ---- Dark ----------------------------------------------------------------------------------

private val DarkAccent = Color(0xFF1DE9B6)
private val DarkCard = Color(0xFF161B22)
private val DarkSurfaceVariant = Color(0xFF21262D)

private val DarkCode = CodePalette(
    background = Color(0xFF0B0F14),
    text = Color(0xFFE6EDF3),
    lineNumber = Color(0xFF6E7681),
    keyword = Color(0xFFFF7B72),
    builtin = Color(0xFF79C0FF),
    string = Color(0xFFA5D6A7),
    comment = Color(0xFF8B949E),
    number = Color(0xFFF2CC60),
    decorator = Color(0xFFD2A8FF),
    self = Color(0xFFFFA657),
    function = Color(0xFFD2A8FF),
    className = Color(0xFFFFD27F)
)

private val DarkQuickMLColors = QuickMLColors(
    isDark = true,
    accent = DarkAccent,
    card = DarkCard,
    easy = Color(0xFF4ADE80),
    medium = Color(0xFFFBBF24),
    hard = Color(0xFFF87171),
    neutral = Color(0xFF9DA7B3),
    ratingForgot = Color(0xFFE5737B),
    ratingHard = Color(0xFFF2B84B),
    ratingGood = DarkAccent,
    ratingEasy = Color(0xFF60A5FA),
    success = Color(0xFF4ADE80),
    danger = Color(0xFFF87171),
    heart = Color(0xFFFF4D6D),
    heatmap = listOf(
        DarkSurfaceVariant,
        lerp(DarkCard, DarkAccent, 0.30f),
        lerp(DarkCard, DarkAccent, 0.50f),
        lerp(DarkCard, DarkAccent, 0.75f),
        DarkAccent
    ),
    chartGrid = Color(0xFF9DA7B3).copy(alpha = 0.18f),
    code = DarkCode
)

private val DarkColors = darkColorScheme(
    primary = DarkAccent,
    onPrimary = Color(0xFF00382B),
    primaryContainer = Color(0xFF005140),
    onPrimaryContainer = Color(0xFF6FF7D6),
    secondary = Color(0xFF80CBC4),
    onSecondary = Color(0xFF003732),
    tertiary = Color(0xFFB39DDB),
    background = Color(0xFF0D1117),
    onBackground = Color(0xFFE6EDF3),
    surface = DarkCard,
    onSurface = Color(0xFFE6EDF3),
    surfaceVariant = DarkSurfaceVariant,
    onSurfaceVariant = Color(0xFF9DA7B3),
    surfaceContainerLow = DarkCard,
    surfaceContainer = Color(0xFF1C2128),
    surfaceContainerHigh = Color(0xFF21262D),
    surfaceContainerHighest = Color(0xFF282E36),
    outline = Color(0xFF424A53),
    outlineVariant = Color(0xFF30363D)
)

// ---- Light ---------------------------------------------------------------------------------
// Calm near-white surfaces (white cards on a cool off-white background), a deeper teal accent
// (5.3:1 on white) and status colours that keep >= 4.5:1 on white.

private val LightAccent = Color(0xFF00796B)
private val LightCard = Color(0xFFFFFFFF)
private val LightText = Color(0xFF1F2328)
private val LightMuted = Color(0xFF59636E)

/** GitHub-light-like syntax colours on a near-white code box. */
private val LightCode = CodePalette(
    background = Color(0xFFF4F6F9),
    text = LightText,
    lineNumber = Color(0xFF8C959F),
    keyword = Color(0xFFCF222E),
    builtin = Color(0xFF0550AE),
    string = Color(0xFF0A3069),
    comment = Color(0xFF6E7781),
    number = Color(0xFF0550AE),
    decorator = Color(0xFF8250DF),
    self = Color(0xFF953800),
    function = Color(0xFF8250DF),
    className = Color(0xFF953800)
)

private val LightQuickMLColors = QuickMLColors(
    isDark = false,
    accent = LightAccent,
    card = LightCard,
    easy = Color(0xFF1A7F37),
    medium = Color(0xFF9A6700),
    hard = Color(0xFFCF222E),
    neutral = Color(0xFF6E7781),
    ratingForgot = Color(0xFFCF222E),
    ratingHard = Color(0xFF9A6700),
    ratingGood = LightAccent,
    ratingEasy = Color(0xFF0969DA),
    success = Color(0xFF1A7F37),
    danger = Color(0xFFCF222E),
    heart = Color(0xFFE8385E),
    // Pale tint just above the surface up to deep teal.
    heatmap = listOf(
        Color(0xFFEBEEF2),
        Color(0xFFC4EADF),
        Color(0xFF80CFBB),
        Color(0xFF2AA58B),
        Color(0xFF00695C)
    ),
    chartGrid = Color(0xFFE1E5EA),
    code = LightCode
)

private val LightColors = lightColorScheme(
    primary = LightAccent,
    onPrimary = Color(0xFFFFFFFF),
    primaryContainer = Color(0xFFCDEFE7),
    onPrimaryContainer = Color(0xFF00382E),
    secondary = Color(0xFF41675F),
    onSecondary = Color(0xFFFFFFFF),
    secondaryContainer = Color(0xFFD7ECE6),
    onSecondaryContainer = Color(0xFF0E2E28),
    tertiary = Color(0xFF6E40C9),
    onTertiary = Color(0xFFFFFFFF),
    tertiaryContainer = Color(0xFFEADDFF),
    onTertiaryContainer = Color(0xFF2B0A66),
    background = Color(0xFFF2F4F7),
    onBackground = LightText,
    surface = LightCard,
    onSurface = LightText,
    surfaceVariant = Color(0xFFEDF0F3),
    onSurfaceVariant = LightMuted,
    surfaceContainerLowest = Color(0xFFFFFFFF),
    surfaceContainerLow = Color(0xFFFFFFFF),
    surfaceContainer = Color(0xFFFFFFFF),
    surfaceContainerHigh = Color(0xFFFFFFFF),
    surfaceContainerHighest = Color(0xFFE6EAEE),
    outline = Color(0xFF8A949E),
    outlineVariant = Color(0xFFD8DEE4)
)

private val LocalQuickMLColors = staticCompositionLocalOf { DarkQuickMLColors }

/** Access point for the app colours: `QuickMLTheme.colors.accent`. */
object QuickMLTheme {
    val colors: QuickMLColors
        @Composable
        @ReadOnlyComposable
        get() = LocalQuickMLColors.current
}

/** The dark set recoloured with [accent]; Teal is the base set unchanged. */
private fun darkFor(accent: AccentChoice): Pair<QuickMLColors, ColorScheme> {
    if (accent == AccentChoice.TEAL) return DarkQuickMLColors to DarkColors
    val a = Color(accent.dark)
    // ratingGood and success keep their colours so ratings and quiz answers keep their meaning.
    val colors = DarkQuickMLColors.copy(
        accent = a,
        heatmap = listOf(
            DarkSurfaceVariant,
            lerp(DarkCard, a, 0.30f),
            lerp(DarkCard, a, 0.50f),
            lerp(DarkCard, a, 0.75f),
            a
        )
    )
    val scheme = DarkColors.copy(
        primary = a,
        onPrimary = Color(0xFF0B0F14),
        primaryContainer = lerp(DarkCard, a, 0.35f),
        onPrimaryContainer = lerp(a, Color.White, 0.6f),
        secondary = lerp(a, Color(0xFF9DA7B3), 0.5f),
        onSecondary = Color(0xFF0B0F14)
    )
    return colors to scheme
}

/** The light set recoloured with [accent]; Teal is the base set unchanged. */
private fun lightFor(accent: AccentChoice): Pair<QuickMLColors, ColorScheme> {
    if (accent == AccentChoice.TEAL) return LightQuickMLColors to LightColors
    val a = Color(accent.light)
    val colors = LightQuickMLColors.copy(
        accent = a,
        heatmap = listOf(
            Color(0xFFEBEEF2),
            lerp(Color.White, a, 0.25f),
            lerp(Color.White, a, 0.50f),
            lerp(Color.White, a, 0.80f),
            lerp(a, Color.Black, 0.15f)
        )
    )
    val scheme = LightColors.copy(
        primary = a,
        onPrimary = Color.White,
        primaryContainer = lerp(Color.White, a, 0.18f),
        onPrimaryContainer = lerp(a, Color.Black, 0.6f),
        secondary = lerp(a, Color(0xFF59636E), 0.5f),
        onSecondary = Color.White,
        secondaryContainer = lerp(Color.White, a, 0.12f),
        onSecondaryContainer = lerp(a, Color.Black, 0.7f)
    )
    return colors to scheme
}

/** Light or dark Material scheme plus the matching [QuickMLColors], tinted with [accent]. */
@Composable
fun QuickMLTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    accent: AccentChoice = AccentChoice.TEAL,
    content: @Composable () -> Unit
) {
    val (colors, scheme) = remember(darkTheme, accent) {
        if (darkTheme) darkFor(accent) else lightFor(accent)
    }
    CompositionLocalProvider(LocalQuickMLColors provides colors) {
        MaterialTheme(
            colorScheme = scheme,
            content = content
        )
    }
}
