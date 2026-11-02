package com.parthdeuri.quickml.data.settings

import android.content.SharedPreferences
import com.parthdeuri.quickml.data.repo.ConceptRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Card preferences that change how a problem card is shown. */
data class CardPrefs(
    /** Revise mode asks "Which pattern fits?" before revealing topics and patterns. */
    val quizBeforeReveal: Boolean = true,
    /** Code tab masks the solution's key lines. */
    val practiceMode: Boolean = false,
    /** Think timer on the Concept tab in Revise / Quiz modes; 0 = off. */
    val thinkTimerMinutes: Int = 0
)

/** App colour theme chosen in Settings. */
enum class ThemeMode(val label: String) {
    SYSTEM("System"),
    LIGHT("Light"),
    DARK("Dark")
}

/**
 * Accent colour chosen in Settings. [dark] is used on the dark theme, [light] on the light one
 * (deeper, so it keeps >= 3:1 on the off-white background). ARGB values.
 */
enum class AccentChoice(val label: String, val dark: Long, val light: Long) {
    TEAL("Teal", 0xFF1DE9B6, 0xFF00796B),
    BLUE("Blue", 0xFF64B5F6, 0xFF1565C0),
    PURPLE("Purple", 0xFFC4A1FF, 0xFF6E40C9),
    PINK("Pink", 0xFFFF6FA5, 0xFFC2185B),
    ORANGE("Orange", 0xFFFFA040, 0xFFBF5B00),
    GREEN("Green", 0xFF9CCC65, 0xFF2E7D32)
}

/**
 * Selected sheet, card preferences and theme, persisted in SharedPreferences and observable app-wide,
 * so changing one of them in one screen updates every other screen.
 */
class StudySettings(private val prefs: SharedPreferences) {
    private val selectedSheet = MutableStateFlow(
        prefs.getString(KEY_SHEET, null) ?: ConceptRepository.DEFAULT_SHEET_ID
    )
    val selectedSheetId: StateFlow<String> = selectedSheet.asStateFlow()

    private val card = MutableStateFlow(
        CardPrefs(
            quizBeforeReveal = prefs.getBoolean(KEY_QUIZ_BEFORE_REVEAL, true),
            practiceMode = prefs.getBoolean(KEY_PRACTICE_MODE, false),
            thinkTimerMinutes = sanitizeTimer(prefs.getInt(KEY_THINK_TIMER, 0))
        )
    )
    val cardPrefs: StateFlow<CardPrefs> = card.asStateFlow()

    private val theme = MutableStateFlow(parseThemeMode(prefs.getString(KEY_THEME_MODE, null)))
    val themeMode: StateFlow<ThemeMode> = theme.asStateFlow()

    private val accentFlow = MutableStateFlow(parseAccent(prefs.getString(KEY_ACCENT, null)))
    val accent: StateFlow<AccentChoice> = accentFlow.asStateFlow()

    private val voice = MutableStateFlow(prefs.getString(KEY_TTS_VOICE, null))

    /** Name of the chosen read-aloud voice; null = the system default. */
    val voiceName: StateFlow<String?> = voice.asStateFlow()

    private val rate = MutableStateFlow(sanitizeRate(prefs.getFloat(KEY_TTS_RATE, 1.0f)))

    /** Read-aloud speed, one of [SPEECH_RATE_CHOICES]. */
    val speechRate: StateFlow<Float> = rate.asStateFlow()

    fun selectSheet(id: String) {
        if (id == selectedSheet.value) return
        prefs.edit().putString(KEY_SHEET, id).apply()
        selectedSheet.value = id
    }

    fun setQuizBeforeReveal(on: Boolean) {
        prefs.edit().putBoolean(KEY_QUIZ_BEFORE_REVEAL, on).apply()
        card.value = card.value.copy(quizBeforeReveal = on)
    }

    fun setPracticeMode(on: Boolean) {
        prefs.edit().putBoolean(KEY_PRACTICE_MODE, on).apply()
        card.value = card.value.copy(practiceMode = on)
    }

    /** 0 (off), 1, 2 or 5 minutes. */
    fun setThinkTimerMinutes(minutes: Int) {
        val value = sanitizeTimer(minutes)
        prefs.edit().putInt(KEY_THINK_TIMER, value).apply()
        card.value = card.value.copy(thinkTimerMinutes = value)
    }

    fun setThemeMode(mode: ThemeMode) {
        if (mode == theme.value) return
        prefs.edit().putString(KEY_THEME_MODE, mode.name).apply()
        theme.value = mode
    }

    fun setAccent(choice: AccentChoice) {
        if (choice == accentFlow.value) return
        prefs.edit().putString(KEY_ACCENT, choice.name).apply()
        accentFlow.value = choice
    }

    fun setVoiceName(name: String?) {
        if (name == voice.value) return
        if (name == null) {
            prefs.edit().remove(KEY_TTS_VOICE).apply()
        } else {
            prefs.edit().putString(KEY_TTS_VOICE, name).apply()
        }
        voice.value = name
    }

    fun setSpeechRate(value: Float) {
        val clean = sanitizeRate(value)
        prefs.edit().putFloat(KEY_TTS_RATE, clean).apply()
        rate.value = clean
    }

    companion object {
        /** Read-aloud speeds offered in Settings. */
        val SPEECH_RATE_CHOICES = listOf(0.8f, 1.0f, 1.2f, 1.5f)

        /** Choices offered in Settings (0 = off). */
        val THINK_TIMER_CHOICES = listOf(0, 1, 2, 5)

        private const val KEY_SHEET = "selected_sheet_id"
        private const val KEY_QUIZ_BEFORE_REVEAL = "quiz_before_reveal"
        private const val KEY_PRACTICE_MODE = "code_practice_mode"
        private const val KEY_THINK_TIMER = "think_timer_minutes"
        private const val KEY_THEME_MODE = "theme_mode"
        private const val KEY_ACCENT = "accent_choice"
        private const val KEY_TTS_VOICE = "tts_voice"
        private const val KEY_TTS_RATE = "tts_rate"

        private fun parseThemeMode(value: String?): ThemeMode =
            ThemeMode.values().firstOrNull { it.name == value } ?: ThemeMode.SYSTEM

        private fun parseAccent(value: String?): AccentChoice =
            AccentChoice.values().firstOrNull { it.name == value } ?: AccentChoice.TEAL

        private fun sanitizeRate(value: Float): Float =
            SPEECH_RATE_CHOICES.firstOrNull { it == value } ?: 1.0f

        private fun sanitizeTimer(minutes: Int): Int =
            if (minutes in THINK_TIMER_CHOICES) minutes else 0
    }
}
