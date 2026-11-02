package com.parthdeuri.quickml.domain.srs

import java.time.LocalDate
import kotlin.math.max
import kotlin.math.roundToInt

enum class Rating { FORGOT, HARD, GOOD, EASY }

val Rating.label: String
    get() = when (this) {
        Rating.FORGOT -> "Forgot"
        Rating.HARD -> "Hard"
        Rating.GOOD -> "Good"
        Rating.EASY -> "Easy"
    }

const val DEFAULT_EASE = 2.5

/** Scheduling state of one problem. [dueDay] is a local epoch day (LocalDate.toEpochDay()). */
data class ReviewState(
    val slug: String,
    val ease: Double = DEFAULT_EASE,
    val intervalDays: Int = 0,
    val reps: Int = 0,
    val lapses: Int = 0,
    val dueDay: Long = 0L,
    val lastRating: String? = null,
    val lastReviewedAt: Long? = null
)

/** Behind an interface so a different algorithm (e.g. FSRS) can replace SM-2 later. */
interface ReviewScheduler {
    fun schedule(state: ReviewState, rating: Rating, today: LocalDate): ReviewState

    /** The interval (in days) each rating button would give. */
    fun preview(state: ReviewState, today: LocalDate): Map<Rating, Int>
}

object Sm2Scheduler : ReviewScheduler {
    private const val MIN_EASE = 1.3
    private const val MAX_INTERVAL_DAYS = 365

    override fun schedule(state: ReviewState, rating: Rating, today: LocalDate): ReviewState {
        val next = compute(state, rating)
        return next.copy(
            dueDay = today.toEpochDay() + next.intervalDays,
            lastRating = rating.name,
            lastReviewedAt = System.currentTimeMillis()
        )
    }

    override fun preview(state: ReviewState, today: LocalDate): Map<Rating, Int> =
        Rating.values().associateWith { compute(state, it).intervalDays }

    private fun capped(days: Int): Int = days.coerceIn(1, MAX_INTERVAL_DAYS)

    /** New interval / ease / reps / lapses; dueDay and the last* fields are set by [schedule]. */
    private fun compute(state: ReviewState, rating: Rating): ReviewState {
        val interval = state.intervalDays
        val reps = state.reps
        return when (rating) {
            Rating.FORGOT -> state.copy(
                reps = 0,
                lapses = state.lapses + 1,
                intervalDays = capped(1),
                ease = max(MIN_EASE, state.ease - 0.2)
            )

            Rating.HARD -> state.copy(
                reps = reps + 1,
                intervalDays = capped(
                    if (reps == 0) 1 else max(interval + 1, (interval * 1.2).roundToInt())
                ),
                ease = max(MIN_EASE, state.ease - 0.15)
            )

            Rating.GOOD -> state.copy(
                reps = reps + 1,
                intervalDays = capped(
                    when (reps) {
                        0 -> 1
                        1 -> 3
                        else -> max(interval + 1, (interval * state.ease).roundToInt())
                    }
                )
            )

            Rating.EASY -> state.copy(
                reps = reps + 1,
                intervalDays = capped(
                    when (reps) {
                        0 -> 3
                        1 -> 5
                        else -> max(interval + 1, (interval * state.ease * 1.3).roundToInt())
                    }
                ),
                ease = state.ease + 0.15
            )
        }
    }
}

/** "1d", "3d", "2w", "3mo", "12mo". */
fun formatInterval(days: Int): String = when {
    days < 7 -> days.toString() + "d"
    days < 30 -> (days / 7f).roundToInt().toString() + "w"
    else -> (days / 30f).roundToInt().toString() + "mo"
}
