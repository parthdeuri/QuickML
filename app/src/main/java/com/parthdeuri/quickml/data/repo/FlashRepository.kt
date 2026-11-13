package com.parthdeuri.quickml.data.repo

import androidx.room.withTransaction
import com.parthdeuri.quickml.data.db.ActivityLogEntity
import com.parthdeuri.quickml.data.db.FlashReviewEntity
import com.parthdeuri.quickml.data.db.FlashcardEntity
import com.parthdeuri.quickml.data.db.QuickMLDatabase
import com.parthdeuri.quickml.data.db.SheetEntity
import com.parthdeuri.quickml.domain.srs.Rating
import com.parthdeuri.quickml.domain.srs.ReviewScheduler
import com.parthdeuri.quickml.domain.srs.ReviewState
import kotlinx.coroutines.flow.Flow
import java.time.LocalDate

/** What a study session would contain right now, for the deck picker. */
data class FlashCounts(
    /** Reviews due today. */
    val due: Int,
    /** New cards still allowed today (daily limit minus the ones already started today). */
    val newToday: Int,
    /** Cards in the selection, rated or not. */
    val total: Int
) {
    val session: Int
        get() = due + newToday
}

/**
 * Flash cards: due reviews first, then new cards up to the daily limit. Scheduling reuses the
 * same SM-2 [ReviewScheduler] as the concept cards but lives in its own table (flash_review), so
 * the Revise due count, the widget and the reminder are not affected.
 */
class FlashRepository(
    private val db: QuickMLDatabase,
    private val scheduler: ReviewScheduler
) {
    private val dao
        get() = db.flashDao()

    /** The decks are the sheets. */
    fun observeDecks(): Flow<List<SheetEntity>> = db.contentDao().observeSheets()

    /** Reviews due today across every deck (dashboard). */
    fun observeDueTotal(today: LocalDate = LocalDate.now()): Flow<Int> =
        dao.observeDueTotal(today.toEpochDay())

    /** [deck] is a sheet id, or null for all decks. */
    private fun deckPattern(deck: String?): String = if (deck == null) "%" else "%|" + deck + "|%"

    suspend fun counts(
        deck: String?,
        kinds: Set<String>,
        newLimit: Int,
        today: LocalDate = LocalDate.now()
    ): FlashCounts {
        if (kinds.isEmpty()) return FlashCounts(0, 0, 0)
        val pattern = deckPattern(deck)
        val kindList = kinds.toList()
        val epoch = today.toEpochDay()
        val remaining = (newLimit - dao.countIntroducedOn(epoch)).coerceAtLeast(0)
        return FlashCounts(
            due = dao.countDue(pattern, kindList, epoch),
            newToday = minOf(remaining, dao.countNew(pattern, kindList)),
            total = dao.countAll(pattern, kindList)
        )
    }

    /** Due reviews (most overdue first) followed by new cards in curriculum order. */
    suspend fun buildQueue(
        deck: String?,
        kinds: Set<String>,
        newLimit: Int,
        today: LocalDate = LocalDate.now()
    ): List<FlashcardEntity> {
        if (kinds.isEmpty()) return emptyList()
        val pattern = deckPattern(deck)
        val kindList = kinds.toList()
        val epoch = today.toEpochDay()
        val due = dao.dueCards(pattern, kindList, epoch, MAX_DUE_PER_SESSION)
        val remaining = (newLimit - dao.countIntroducedOn(epoch)).coerceAtLeast(0)
        val fresh = if (remaining > 0) dao.newCards(pattern, kindList, remaining) else emptyList()
        return due + fresh
    }

    /** The interval (days) each rating button would give for this card. */
    suspend fun preview(cardId: String, today: LocalDate = LocalDate.now()): Map<Rating, Int> {
        val state = dao.getReview(cardId)?.toState()
            ?: ReviewState(slug = cardId, dueDay = today.toEpochDay())
        return scheduler.preview(state, today)
    }

    /** Applies [rating], logs a FLASH event and returns the new interval in days. */
    suspend fun rate(card: FlashcardEntity, rating: Rating, today: LocalDate = LocalDate.now()): Int =
        db.withTransaction {
            val epoch = today.toEpochDay()
            val existing = dao.getReview(card.id)
            val state = existing?.toState() ?: ReviewState(slug = card.id, dueDay = epoch)
            val next = scheduler.schedule(state, rating, today)
            dao.upsertReview(next.toFlashEntity(existing?.firstSeenDay ?: epoch))
            db.activityDao().insert(
                ActivityLogEntity(
                    slug = card.slug,
                    type = LOG_FLASH,
                    timestamp = System.currentTimeMillis(),
                    durationMs = 0L,
                    rating = rating.name
                )
            )
            next.intervalDays
        }

    companion object {
        const val LOG_FLASH = "FLASH"
        private const val MAX_DUE_PER_SESSION = 200
    }
}

private fun FlashReviewEntity.toState(): ReviewState = ReviewState(
    slug = cardId,
    ease = ease,
    intervalDays = intervalDays,
    reps = reps,
    lapses = lapses,
    dueDay = dueDay,
    lastRating = lastRating,
    lastReviewedAt = lastReviewedAt
)

private fun ReviewState.toFlashEntity(firstSeenDay: Long): FlashReviewEntity = FlashReviewEntity(
    cardId = slug,
    ease = ease,
    intervalDays = intervalDays,
    reps = reps,
    lapses = lapses,
    dueDay = dueDay,
    lastRating = lastRating,
    lastReviewedAt = lastReviewedAt,
    firstSeenDay = firstSeenDay
)
