package com.parthdeuri.quickml.reminder

import android.content.SharedPreferences

/** Daily reminder preferences (default: off, 20:00). */
class ReminderSettings(private val prefs: SharedPreferences) {
    var enabled: Boolean
        get() = prefs.getBoolean(KEY_ENABLED, false)
        set(value) {
            prefs.edit().putBoolean(KEY_ENABLED, value).apply()
        }

    var hour: Int
        get() = prefs.getInt(KEY_HOUR, DEFAULT_HOUR)
        set(value) {
            prefs.edit().putInt(KEY_HOUR, value).apply()
        }

    var minute: Int
        get() = prefs.getInt(KEY_MINUTE, 0)
        set(value) {
            prefs.edit().putInt(KEY_MINUTE, value).apply()
        }

    private companion object {
        const val KEY_ENABLED = "reminder_enabled"
        const val KEY_HOUR = "reminder_hour"
        const val KEY_MINUTE = "reminder_minute"
        const val DEFAULT_HOUR = 20
    }
}
