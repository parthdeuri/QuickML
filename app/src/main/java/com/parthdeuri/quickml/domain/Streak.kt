package com.parthdeuri.quickml.domain

import java.time.LocalDate

/** Consecutive active days ending today, or yesterday when nothing happened yet today. */
fun currentStreak(activeDays: Set<LocalDate>, today: LocalDate): Int {
    var cursor = if (activeDays.contains(today)) today else today.minusDays(1)
    var streak = 0
    while (activeDays.contains(cursor)) {
        streak++
        cursor = cursor.minusDays(1)
    }
    return streak
}
