package com.parthdeuri.quickml.data.backup

import android.content.Context
import android.net.Uri
import androidx.room.withTransaction
import com.parthdeuri.quickml.data.db.ActivityLogEntity
import com.parthdeuri.quickml.data.db.ConceptEntity
import com.parthdeuri.quickml.data.db.FlashReviewEntity
import com.parthdeuri.quickml.data.db.ReviewStateEntity
import com.parthdeuri.quickml.data.db.QuickMLDatabase
import com.parthdeuri.quickml.data.db.UserProgressEntity
import com.parthdeuri.quickml.data.repo.ConceptRepository
import com.parthdeuri.quickml.data.settings.StudySettings
import com.parthdeuri.quickml.reminder.ReminderScheduler
import com.parthdeuri.quickml.reminder.ReminderSettings
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.IOException
import java.time.Instant

@Serializable
data class BackupFile(
    val app: String = "",
    val backupVersion: Int = 0,
    /** ISO-8601 instant. */
    val exportedAt: String = "",
    val appVersion: String = "",
    val userProgress: List<BackupProgress> = emptyList(),
    val reviewState: List<BackupReview> = emptyList(),
    /** Flash card schedules; absent in backups made before the Flash tab existed. */
    val flashReview: List<BackupFlashReview> = emptyList(),
    val activityLog: List<BackupActivity> = emptyList(),
    /** Full rows of the user's own problems. */
    val userConcepts: List<BackupConcept> = emptyList(),
    val settings: BackupSettings = BackupSettings()
)

@Serializable
data class BackupProgress(
    val slug: String,
    val status: String = ConceptRepository.STATUS_UNSOLVED,
    val bookmarked: Boolean = false,
    val note: String? = null,
    val updatedAt: Long = 0L,
    val solvedAt: Long? = null,
    val mySolution: String? = null
)

@Serializable
data class BackupFlashReview(
    val cardId: String,
    val ease: Double = 2.5,
    val intervalDays: Int = 0,
    val reps: Int = 0,
    val lapses: Int = 0,
    val dueDay: Long = 0L,
    val lastRating: String? = null,
    val lastReviewedAt: Long? = null,
    val firstSeenDay: Long = 0L
)

@Serializable
data class BackupReview(
    val slug: String,
    val ease: Double = 2.5,
    val intervalDays: Int = 0,
    val reps: Int = 0,
    val lapses: Int = 0,
    val dueDay: Long = 0L,
    val lastRating: String? = null,
    val lastReviewedAt: Long? = null
)

@Serializable
data class BackupActivity(
    val slug: String,
    val type: String,
    val timestamp: Long,
    val durationMs: Long = 0L,
    val rating: String? = null
)

@Serializable
data class BackupConcept(
    val slug: String,
    val title: String,
    val platform: String = "other",
    val url: String = "",
    val difficulty: String = "Medium",
    val topics: List<String> = emptyList(),
    val patterns: List<String> = emptyList(),
    val summary: String? = null,
    val intuition: String? = null,
    val approach: List<String> = emptyList(),
    val time: String? = null,
    val space: String? = null,
    val solution: String? = null,
    val edgeCases: List<String> = emptyList(),
    val related: List<String> = emptyList(),
    val description: String? = null,
    val examplesJson: String? = null,
    val constraintsJson: String? = null,
    val bruteForce: String? = null,
    val whyItWorks: String? = null,
    val dryRunJson: String? = null,
    val keyLinesJson: String? = null
)

/** Null fields are left unchanged on restore. */
@Serializable
data class BackupSettings(
    val selectedSheetId: String? = null,
    val quizBeforeReveal: Boolean? = null,
    val practiceMode: Boolean? = null,
    val thinkTimerMinutes: Int? = null,
    val reminderEnabled: Boolean? = null,
    val reminderHour: Int? = null,
    val reminderMinute: Int? = null
)

/** Counts shown in toasts and the restore confirmation. */
data class BackupSummary(
    val solved: Int,
    val reviews: Int,
    val activity: Int,
    val notes: Int,
    val ownConcepts: Int,
    val flashReviews: Int = 0
) {
    fun describe(): String = buildList {
        add(plural(solved, "learned card", "learned cards"))
        add(plural(reviews, "review schedule", "review schedules"))
        if (notes > 0) add(plural(notes, "note", "notes"))
        if (ownConcepts > 0) add(plural(ownConcepts, "own card", "own cards"))
        if (flashReviews > 0) add(plural(flashReviews, "flash card schedule", "flash card schedules"))
        add(plural(activity, "activity entry", "activity entries"))
    }.joinToString(", ")

    private fun plural(n: Int, one: String, many: String): String =
        n.toString() + " " + if (n == 1) one else many
}

/** A friendly, user-facing failure (nothing was changed). */
class BackupException(message: String, cause: Throwable? = null) : Exception(message, cause)

/** Settings -> back up / restore: user tables, own problems and study settings as JSON. */
class BackupManager(
    private val context: Context,
    private val db: QuickMLDatabase,
    private val studySettings: StudySettings,
    private val reminderSettings: ReminderSettings,
    private val onRestored: () -> Unit
) {
    private val json = Json {
        ignoreUnknownKeys = true
        coerceInputValues = true
        encodeDefaults = true
    }

    /** Writes a backup to [uri] (from CreateDocument). Throws [BackupException]. */
    suspend fun export(uri: Uri): BackupSummary = withContext(Dispatchers.IO) {
        val backup = try {
            buildBackup()
        } catch (e: Exception) {
            throw BackupException("Could not read your progress.", e)
        }
        val bytes = json.encodeToString(BackupFile.serializer(), backup).toByteArray(Charsets.UTF_8)
        try {
            val resolver = context.contentResolver
            // "wt" truncates; a few providers only support the default mode.
            val opened = try {
                resolver.openOutputStream(uri, "wt")
            } catch (e: Exception) {
                resolver.openOutputStream(uri)
            }
            val stream = opened ?: throw IOException("No output stream")
            stream.use { it.write(bytes) }
        } catch (e: Exception) {
            throw BackupException("Could not write the backup file.", e)
        }
        summaryOf(backup)
    }

    /** Reads and validates a backup from [uri] (from OpenDocument). Throws [BackupException]. */
    suspend fun read(uri: Uri): BackupFile = withContext(Dispatchers.IO) {
        val text = try {
            context.contentResolver.openInputStream(uri)
                ?.bufferedReader(Charsets.UTF_8)
                ?.use { it.readText() }
                ?: throw IOException("No input stream")
        } catch (e: Exception) {
            throw BackupException("Could not open the file.", e)
        }
        val backup = try {
            json.decodeFromString(BackupFile.serializer(), text)
        } catch (e: Exception) {
            throw BackupException("This file isn't a QuickML backup.", e)
        }
        if (backup.app != APP_ID) throw BackupException("This file isn't a QuickML backup.")
        if (backup.backupVersion < 1) throw BackupException("This backup format isn't supported.")
        backup
    }

    fun summaryOf(backup: BackupFile): BackupSummary = BackupSummary(
        solved = backup.userProgress.count { it.status == ConceptRepository.STATUS_SOLVED },
        reviews = backup.reviewState.size,
        activity = backup.activityLog.size,
        notes = backup.userProgress.count { !it.note.isNullOrBlank() },
        ownConcepts = backup.userConcepts.size,
        flashReviews = backup.flashReview.size
    )

    /**
     * Replaces the user tables with [backup] in one transaction, upserts the own problems,
     * then restores the settings and reschedules the reminder. Throws [BackupException]
     * (the database is unchanged when the transaction fails).
     */
    suspend fun restore(backup: BackupFile) {
        withContext(Dispatchers.IO) {
            val progress = backup.userProgress
                .filter { it.slug.isNotBlank() }
                .distinctBy { it.slug }
                .map { it.toEntity() }
            val reviews = backup.reviewState
                .filter { it.slug.isNotBlank() }
                .distinctBy { it.slug }
                .map { it.toEntity() }
            val flashReviews = backup.flashReview
                .filter { it.cardId.isNotBlank() }
                .distinctBy { it.cardId }
                .map { it.toEntity() }
            val activity = backup.activityLog
                .filter { it.slug.isNotBlank() && it.type.isNotBlank() }
                .map { it.toEntity() }
            // Only own problems: a backup can never overwrite bundled content.
            val problems = backup.userConcepts
                .filter { it.slug.startsWith(ConceptRepository.USER_SLUG_PREFIX) && it.title.isNotBlank() }
                .distinctBy { it.slug }
                .map { it.toEntity() }
            try {
                db.withTransaction {
                    val progressDao = db.progressDao()
                    val reviewDao = db.reviewDao()
                    val activityDao = db.activityDao()
                    progressDao.clearAll()
                    reviewDao.clearAll()
                    activityDao.clearAll()
                    db.contentDao().insertConcepts(problems)
                    progressDao.insertAll(progress)
                    reviewDao.insertAll(reviews)
                    // A backup from before the Flash tab has none: keep the current schedules.
                    if (flashReviews.isNotEmpty()) {
                        db.flashDao().clearReviews()
                        db.flashDao().insertReviews(flashReviews)
                    }
                    activityDao.insertAll(activity)
                }
            } catch (e: Exception) {
                throw BackupException("Could not restore the backup. Nothing was changed.", e)
            }
        }
        applySettings(backup.settings)
        onRestored()
    }

    private fun applySettings(settings: BackupSettings) {
        settings.selectedSheetId?.takeIf { it.isNotBlank() }?.let { studySettings.selectSheet(it) }
        settings.quizBeforeReveal?.let { studySettings.setQuizBeforeReveal(it) }
        settings.practiceMode?.let { studySettings.setPracticeMode(it) }
        settings.thinkTimerMinutes?.let { studySettings.setThinkTimerMinutes(it) }
        val hour = settings.reminderHour
        if (hour != null && hour in 0..23) reminderSettings.hour = hour
        val minute = settings.reminderMinute
        if (minute != null && minute in 0..59) reminderSettings.minute = minute
        settings.reminderEnabled?.let { reminderSettings.enabled = it }
        // Reschedule (or cancel) the reminder for the restored settings.
        if (reminderSettings.enabled) {
            ReminderScheduler.schedule(context, reminderSettings.hour, reminderSettings.minute)
        } else {
            ReminderScheduler.cancel(context)
        }
    }

    private suspend fun buildBackup(): BackupFile {
        val progress = db.progressDao().getAll()
        val reviews = db.reviewDao().getAll()
        val activity = db.activityDao().getAll()
        val problems = db.contentDao().getUserConcepts()
        val card = studySettings.cardPrefs.value
        return BackupFile(
            app = APP_ID,
            backupVersion = BACKUP_VERSION,
            exportedAt = Instant.now().toString(),
            appVersion = appVersionName(),
            userProgress = progress.map { it.toBackup() },
            reviewState = reviews.map { it.toBackup() },
            flashReview = db.flashDao().getAllReviews().map { it.toBackup() },
            activityLog = activity.map { it.toBackup() },
            userConcepts = problems.map { it.toBackup() },
            settings = BackupSettings(
                selectedSheetId = studySettings.selectedSheetId.value,
                quizBeforeReveal = card.quizBeforeReveal,
                practiceMode = card.practiceMode,
                thinkTimerMinutes = card.thinkTimerMinutes,
                reminderEnabled = reminderSettings.enabled,
                reminderHour = reminderSettings.hour,
                reminderMinute = reminderSettings.minute
            )
        )
    }

    @Suppress("DEPRECATION")
    private fun appVersionName(): String = try {
        context.packageManager.getPackageInfo(context.packageName, 0).versionName ?: ""
    } catch (e: Exception) {
        ""
    }

    companion object {
        const val APP_ID = "quickml"
        const val BACKUP_VERSION = 1
    }
}

private fun UserProgressEntity.toBackup() = BackupProgress(
    slug = slug,
    status = status,
    bookmarked = bookmarked,
    note = note,
    updatedAt = updatedAt,
    solvedAt = solvedAt,
    mySolution = mySolution
)

private fun BackupProgress.toEntity() = UserProgressEntity(
    slug = slug,
    status = if (status == ConceptRepository.STATUS_SOLVED) {
        ConceptRepository.STATUS_SOLVED
    } else {
        ConceptRepository.STATUS_UNSOLVED
    },
    bookmarked = bookmarked,
    note = note,
    updatedAt = updatedAt,
    solvedAt = solvedAt,
    mySolution = mySolution
)

private fun FlashReviewEntity.toBackup() = BackupFlashReview(
    cardId = cardId,
    ease = ease,
    intervalDays = intervalDays,
    reps = reps,
    lapses = lapses,
    dueDay = dueDay,
    lastRating = lastRating,
    lastReviewedAt = lastReviewedAt,
    firstSeenDay = firstSeenDay
)

private fun BackupFlashReview.toEntity() = FlashReviewEntity(
    cardId = cardId,
    ease = ease,
    intervalDays = intervalDays,
    reps = reps,
    lapses = lapses,
    dueDay = dueDay,
    lastRating = lastRating,
    lastReviewedAt = lastReviewedAt,
    firstSeenDay = firstSeenDay
)

private fun ReviewStateEntity.toBackup() = BackupReview(
    slug = slug,
    ease = ease,
    intervalDays = intervalDays,
    reps = reps,
    lapses = lapses,
    dueDay = dueDay,
    lastRating = lastRating,
    lastReviewedAt = lastReviewedAt
)

private fun BackupReview.toEntity() = ReviewStateEntity(
    slug = slug,
    ease = ease,
    intervalDays = intervalDays,
    reps = reps,
    lapses = lapses,
    dueDay = dueDay,
    lastRating = lastRating,
    lastReviewedAt = lastReviewedAt
)

private fun ActivityLogEntity.toBackup() = BackupActivity(
    slug = slug,
    type = type,
    timestamp = timestamp,
    durationMs = durationMs,
    rating = rating
)

/** id 0: the restored rows get fresh auto-generated ids. */
private fun BackupActivity.toEntity() = ActivityLogEntity(
    id = 0,
    slug = slug,
    type = type,
    timestamp = timestamp,
    durationMs = durationMs,
    rating = rating
)

private fun ConceptEntity.toBackup() = BackupConcept(
    slug = slug,
    title = title,
    platform = platform,
    url = url,
    difficulty = difficulty,
    topics = topics,
    patterns = patterns,
    summary = summary,
    intuition = intuition,
    approach = approach,
    time = time,
    space = space,
    solution = solution,
    edgeCases = edgeCases,
    related = related,
    description = description,
    examplesJson = examplesJson,
    constraintsJson = constraintsJson,
    bruteForce = bruteForce,
    whyItWorks = whyItWorks,
    dryRunJson = dryRunJson,
    keyLinesJson = keyLinesJson
)

private fun BackupConcept.toEntity() = ConceptEntity(
    slug = slug,
    title = title,
    platform = platform,
    url = url,
    difficulty = difficulty,
    topics = topics,
    patterns = patterns,
    summary = summary,
    intuition = intuition,
    approach = approach,
    time = time,
    space = space,
    solution = solution,
    edgeCases = edgeCases,
    related = related,
    description = description,
    examplesJson = examplesJson,
    constraintsJson = constraintsJson,
    bruteForce = bruteForce,
    whyItWorks = whyItWorks,
    dryRunJson = dryRunJson,
    keyLinesJson = keyLinesJson,
    userAdded = true
)
