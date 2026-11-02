package com.parthdeuri.quickml.data.repo

import androidx.room.withTransaction
import com.parthdeuri.quickml.data.content.ContentJson
import com.parthdeuri.quickml.data.content.ExampleJson
import com.parthdeuri.quickml.data.db.ActivityLogEntity
import com.parthdeuri.quickml.data.db.PatternEntity
import com.parthdeuri.quickml.data.db.ConceptEntity
import com.parthdeuri.quickml.data.db.ReviewStateEntity
import com.parthdeuri.quickml.data.db.QuickMLDatabase
import com.parthdeuri.quickml.data.db.SheetEntity
import com.parthdeuri.quickml.data.db.SheetEntryEntity
import com.parthdeuri.quickml.data.db.UserProgressEntity
import com.parthdeuri.quickml.domain.srs.DEFAULT_EASE
import com.parthdeuri.quickml.domain.srs.Rating
import com.parthdeuri.quickml.domain.srs.ReviewScheduler
import com.parthdeuri.quickml.domain.srs.ReviewState
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.map
import java.time.LocalDate

data class FeedItem(
    val problem: ConceptEntity,
    val section: String,
    val order: Int,
    val status: String,
    val bookmarked: Boolean,
    val review: ReviewStateEntity? = null,
    /** The user's note (Edge cases tab, "My notes"). */
    val note: String? = null,
    /** The user's own solution (Code tab, "Mine"). */
    val mySolution: String? = null
) {
    val solved: Boolean
        get() = status == ConceptRepository.STATUS_SOLVED

    /** One of the user's own problems ("My problems"). */
    val userAdded: Boolean
        get() = problem.userAdded == true

    val hasNote: Boolean
        get() = !note.isNullOrBlank()

    val hasMySolution: Boolean
        get() = !mySolution.isNullOrBlank()

    /** Local epoch day this problem is due for revision, or null without a review state. */
    val dueDay: Long?
        get() = review?.dueDay

    /** Scheduler input for this problem (defaults when a solved problem has no row yet). */
    val reviewState: ReviewState
        get() = review?.toDomain() ?: ReviewState(slug = problem.slug)

    /** Decoded lazily from the raw JSON columns; decode failures give empty lists. */
    val examples: List<ExampleJson> by lazy(LazyThreadSafetyMode.NONE) {
        ContentJson.decodeExamples(problem.examplesJson)
    }
    val constraints: List<String> by lazy(LazyThreadSafetyMode.NONE) {
        ContentJson.decodeConstraints(problem.constraintsJson)
    }
    val dryRun: List<String> by lazy(LazyThreadSafetyMode.NONE) {
        ContentJson.decodeStrings(problem.dryRunJson)
    }
    val keyLines: List<Int> by lazy(LazyThreadSafetyMode.NONE) {
        ContentJson.decodeInts(problem.keyLinesJson)
    }
    val diagram: List<String> by lazy(LazyThreadSafetyMode.NONE) {
        ContentJson.decodeStrings(problem.diagramJson)
    }

    /** A problem is full when intuition, approach and solution are all present. */
    val isFull: Boolean
        get() = problem.intuition != null &&
            problem.approach.isNotEmpty() &&
            problem.solution != null
}

/** One pattern of the library, decoded. */
data class PatternInfo(
    val id: String,
    val name: String,
    val summary: String,
    val whenToUse: List<String>,
    val keyIdea: String,
    val template: String,
    val complexity: String,
    val pitfalls: List<String>,
    /** Linked problem slugs, best first. */
    val problems: List<String>,
    /** Pattern strings used in problems' `patterns` that belong to this pattern. */
    val aliases: List<String>
)

private fun PatternEntity.toInfo(): PatternInfo = PatternInfo(
    id = id,
    name = name,
    summary = summary,
    whenToUse = ContentJson.decodeStrings(whenToUseJson),
    keyIdea = keyIdea,
    template = template,
    complexity = complexity,
    pitfalls = ContentJson.decodeStrings(pitfallsJson),
    problems = ContentJson.decodeStrings(problemsJson),
    aliases = ContentJson.decodeStrings(aliasesJson)
)

/** A sheet with its progress (a problem counts as solved in every sheet that contains it). */
data class SheetInfo(
    val id: String,
    val name: String,
    val solved: Int,
    val total: Int
)

data class FeedData(
    /** The selected sheet (falls back to the default / first sheet when it doesn't exist). */
    val sheetId: String,
    val sheetName: String,
    /** Every sheet, in display order. */
    val sheets: List<SheetInfo>,
    /** The selected sheet's problems, in its order and with its sections. */
    val items: List<FeedItem>,
    /**
     * Every problem across all sheets (stable order: by the first sheet that lists it). Section
     * labels come from the selected sheet when it contains the problem, otherwise from that
     * first sheet.
     */
    val allItems: List<FeedItem>
)

/** Raw rows of all tables the feed is built from. */
internal class FeedSnapshot(
    val sheets: List<SheetEntity>,
    val entries: List<SheetEntryEntity>,
    val problems: List<ConceptEntity>,
    val progress: List<UserProgressEntity>,
    val reviews: List<ReviewStateEntity>
)

fun ReviewStateEntity.toDomain(): ReviewState = ReviewState(
    slug = slug,
    ease = ease,
    intervalDays = intervalDays,
    reps = reps,
    lapses = lapses,
    dueDay = dueDay,
    lastRating = lastRating,
    lastReviewedAt = lastReviewedAt
)

fun ReviewState.toEntity(): ReviewStateEntity = ReviewStateEntity(
    slug = slug,
    ease = ease,
    intervalDays = intervalDays,
    reps = reps,
    lapses = lapses,
    dueDay = dueDay,
    lastRating = lastRating,
    lastReviewedAt = lastReviewedAt
)

class ConceptRepository(
    private val db: QuickMLDatabase,
    private val scheduler: ReviewScheduler,
    /** Called after solve / rate / import / own-problem changes (refreshes the widget). */
    private val onUserDataChanged: () -> Unit = {}
) {
    private val contentDao = db.contentDao()
    private val progressDao = db.progressDao()
    private val activityDao = db.activityDao()
    private val reviewDao = db.reviewDao()

    /**
     * Sheet entries joined with problems, progress and review state. [selectedSheetId] picks the
     * sheet for [FeedData.items]; [FeedData.allItems] always holds every problem.
     */
    fun observeFeed(selectedSheetId: Flow<String>): Flow<FeedData> = combine(
        combine(
            contentDao.observeSheets(),
            contentDao.observeEntries(),
            contentDao.observeConcepts(),
            progressDao.observeAll(),
            reviewDao.observeAll()
        ) { sheets, entries, problems, progress, reviews ->
            FeedSnapshot(sheets, entries, problems, progress, reviews)
        },
        selectedSheetId
    ) { snapshot, selectedId ->
        buildFeed(snapshot, selectedId)
    }

    /** The pattern library in content order (empty when the content has none). */
    fun observePatterns(): Flow<List<PatternInfo>> =
        contentDao.observePatterns().map { list -> list.map { it.toInfo() } }

    private fun buildFeed(snapshot: FeedSnapshot, selectedId: String): FeedData {
        val sheets = snapshot.sheets.sortedWith(
            compareBy<SheetEntity>({ sheetRank(it.id) }, { it.name })
        )
        val mineSelected = selectedId == MINE_SHEET_ID
        val allSelected = selectedId == ALL_SHEET_ID
        val sheet = if (mineSelected || allSelected) {
            null
        } else {
            sheets.firstOrNull { it.id == selectedId }
                ?: sheets.firstOrNull { it.id == DEFAULT_SHEET_ID }
                ?: sheets.firstOrNull()
        }

        val problemBySlug = snapshot.problems.associateBy { it.slug }
        val progressBySlug = snapshot.progress.associateBy { it.slug }
        val reviewBySlug = snapshot.reviews.associateBy { it.slug }
        val entriesBySheet = snapshot.entries
            .filter { it.slug in problemBySlug }
            .groupBy { it.sheetId }
            .mapValues { (_, list) -> list.sortedBy { it.order } }

        fun itemFor(problem: ConceptEntity, section: String, order: Int): FeedItem {
            val p = progressBySlug[problem.slug]
            return FeedItem(
                problem = problem,
                section = section,
                order = order,
                status = p?.status ?: STATUS_UNSOLVED,
                bookmarked = p?.bookmarked ?: false,
                review = reviewBySlug[problem.slug],
                note = p?.note,
                mySolution = p?.mySolution
            )
        }

        fun itemForEntry(problem: ConceptEntity, entry: SheetEntryEntity?): FeedItem = itemFor(
            problem = problem,
            section = entry?.section ?: if (problem.userAdded == true) MINE_SHEET_NAME else "",
            order = entry?.order ?: 0
        )

        // "My problems": the user's own problems, newest first (a virtual sheet).
        val mineConcepts = snapshot.problems
            .filter { it.userAdded == true }
            .sortedByDescending { userConceptTime(it.slug) }

        // Selected sheet, in its order ("All problems" is filled in below from allItems).
        val baseItems = when {
            allSelected -> emptyList()
            mineSelected -> mineConcepts.mapIndexed { index, problem ->
                itemFor(problem, MINE_SHEET_NAME, index + 1)
            }
            sheet == null -> emptyList()
            else -> entriesBySheet[sheet.id].orEmpty().mapNotNull { entry ->
                problemBySlug[entry.slug]?.let { itemForEntry(it, entry) }
            }
        }
        val selectedBySlug = baseItems.associateBy { it.problem.slug }

        // First entry of every problem, by sheet display order.
        val sheetIndex = HashMap<String, Int>()
        val firstEntry = HashMap<String, SheetEntryEntity>()
        sheets.forEachIndexed { index, s ->
            sheetIndex[s.id] = index
            for (entry in entriesBySheet[s.id].orEmpty()) {
                if (entry.slug !in firstEntry) firstEntry[entry.slug] = entry
            }
        }
        val allItems = snapshot.problems
            .sortedWith(
                compareBy<ConceptEntity>(
                    { firstEntry[it.slug]?.let { e -> sheetIndex[e.sheetId] } ?: Int.MAX_VALUE },
                    { firstEntry[it.slug]?.order ?: 0 },
                    { it.title }
                )
            )
            .map { problem ->
                selectedBySlug[problem.slug] ?: itemForEntry(problem, firstEntry[problem.slug])
            }

        // "All problems": every problem once, grouped by its first topic (alphabetical sections,
        // allItems' order inside each); a problem without topics goes to "Other".
        val items = if (allSelected) {
            fun topicOf(item: FeedItem): String =
                item.problem.topics.firstOrNull { it.isNotBlank() } ?: ALL_OTHER_SECTION
            allItems
                .withIndex()
                .sortedWith(
                    compareBy<IndexedValue<FeedItem>>(
                        { topicOf(it.value) == ALL_OTHER_SECTION },
                        { topicOf(it.value) },
                        { it.index }
                    )
                )
                .mapIndexed { order, entry ->
                    entry.value.copy(section = topicOf(entry.value), order = order + 1)
                }
        } else {
            baseItems
        }

        val infos = sheets.map { s ->
            val list = entriesBySheet[s.id].orEmpty()
            SheetInfo(
                id = s.id,
                name = s.name,
                solved = list.count { progressBySlug[it.slug]?.status == STATUS_SOLVED },
                total = list.size
            )
        }
        // The virtual sheet is listed once content exists (no sheets = still importing).
        val mineInfo = SheetInfo(
            id = MINE_SHEET_ID,
            name = MINE_SHEET_NAME,
            solved = mineConcepts.count { progressBySlug[it.slug]?.status == STATUS_SOLVED },
            total = mineConcepts.size
        )
        val allInfo = SheetInfo(
            id = ALL_SHEET_ID,
            name = ALL_SHEET_NAME,
            solved = allItems.count { it.status == STATUS_SOLVED },
            total = allItems.size
        )
        val allInfos = if (infos.isEmpty()) infos else infos + allInfo + mineInfo
        return FeedData(
            sheetId = when {
                allSelected -> ALL_SHEET_ID
                mineSelected -> MINE_SHEET_ID
                else -> sheet?.id ?: ""
            },
            sheetName = when {
                allSelected -> ALL_SHEET_NAME
                mineSelected -> MINE_SHEET_NAME
                else -> sheet?.name ?: "Revise"
            },
            sheets = allInfos,
            items = items,
            allItems = allItems
        )
    }

    /** Known sheets first, in a fixed order; unknown ones after them (then by name). */
    private fun sheetRank(id: String): Int {
        val index = SHEET_ORDER.indexOf(id)
        return if (index >= 0) index else SHEET_ORDER.size
    }

    /**
     * The sheet to show [slug] in: [currentSheetId] when it contains the problem, otherwise the
     * first sheet (display order) that does; "mine" for own problems; null when none does.
     */
    suspend fun sheetFor(slug: String, currentSheetId: String): String? {
        val problem = contentDao.getConcept(slug) ?: return null
        if (currentSheetId == ALL_SHEET_ID) return ALL_SHEET_ID
        if (problem.userAdded == true) return MINE_SHEET_ID
        val ids = contentDao.sheetIdsFor(slug)
        if (currentSheetId in ids) return currentSheetId
        return ids.minWithOrNull(compareBy<String>({ sheetRank(it) }, { it }))
    }

    suspend fun toggleBookmark(slug: String) {
        val current = progressDao.get(slug) ?: UserProgressEntity(slug = slug)
        progressDao.upsert(
            current.copy(
                bookmarked = !current.bookmarked,
                updatedAt = System.currentTimeMillis()
            )
        )
    }

    /** Saves the note ("My notes"); blank clears it. */
    suspend fun saveNote(slug: String, note: String) {
        val value = note.takeIf { it.isNotBlank() }
        db.withTransaction {
            val current = progressDao.get(slug)
            val changed = if (current == null) value != null else current.note != value
            if (changed) {
                val row = current ?: UserProgressEntity(slug = slug)
                progressDao.upsert(row.copy(note = value, updatedAt = System.currentTimeMillis()))
            }
            Unit
        }
    }

    /** Saves the user's own solution; blank removes it. */
    suspend fun saveMySolution(slug: String, code: String) {
        val value = code.trimEnd().takeIf { it.isNotBlank() }
        db.withTransaction {
            val current = progressDao.get(slug)
            val changed = if (current == null) value != null else current.mySolution != value
            if (changed) {
                val row = current ?: UserProgressEntity(slug = slug)
                progressDao.upsert(
                    row.copy(mySolution = value, updatedAt = System.currentTimeMillis())
                )
            }
            Unit
        }
    }

    /**
     * Creates ([existingSlug] null) or updates an own problem; the draft's notes go to the
     * progress row. Returns the problem's slug.
     */
    suspend fun saveUserConcept(existingSlug: String?, draft: UserConceptDraft): String {
        val slug = existingSlug ?: (USER_SLUG_PREFIX + System.currentTimeMillis())
        db.withTransaction {
            // Never overwrite bundled content through the own-problem form.
            val existing = if (existingSlug != null) contentDao.getConcept(existingSlug) else null
            if (existing == null || existing.userAdded == true) {
                contentDao.insertConcept(draft.toEntity(slug))
                val current = progressDao.get(slug) ?: UserProgressEntity(slug = slug)
                progressDao.upsert(
                    current.copy(
                        note = draft.notes.takeIf { it.isNotBlank() },
                        updatedAt = System.currentTimeMillis()
                    )
                )
            }
            Unit
        }
        onUserDataChanged()
        return slug
    }

    /** Deletes an own problem with its progress and review state (bundled problems: no-op). */
    suspend fun deleteUserConcept(slug: String) {
        db.withTransaction {
            val problem = contentDao.getConcept(slug)
            if (problem != null && problem.userAdded == true) {
                contentDao.deleteUserConcept(slug)
                progressDao.delete(slug)
                reviewDao.delete(slug)
            }
            Unit
        }
        onUserDataChanged()
    }

    /** The edit form of an own problem, or null when [slug] isn't one. */
    suspend fun loadUserConceptDraft(slug: String): UserConceptDraft? {
        val problem = contentDao.getConcept(slug) ?: return null
        if (problem.userAdded != true) return null
        return problem.toDraft(progressDao.get(slug)?.note)
    }

    /**
     * Solved <-> unsolved. Marking solved stamps solvedAt, logs a SOLVED event and creates the
     * review state (due tomorrow); marking unsolved deletes the review state.
     */
    suspend fun toggleSolved(slug: String) {
        db.withTransaction {
            val current = progressDao.get(slug)
            val nowSolved = current?.status != STATUS_SOLVED
            applySolved(slug, nowSolved, System.currentTimeMillis(), tomorrowEpochDay())
        }
        onUserDataChanged()
    }

    /**
     * Bulk version of [toggleSolved] (the sheet's section actions, "Import solved problems").
     * When [SPREAD_MIN_COUNT] or more problems get their first review in one action, those
     * first reviews are spread evenly over the next [SPREAD_DAYS] days instead of all tomorrow.
     */
    suspend fun setSolved(slugs: List<String>, solved: Boolean) {
        db.withTransaction {
            val now = System.currentTimeMillis()
            val today = LocalDate.now().toEpochDay()
            val tomorrow = today + 1
            val unique = slugs.distinct()
            if (solved) {
                val fresh = unique.filter { reviewDao.get(it) == null }
                val firstDue = HashMap<String, Long>()
                if (fresh.size >= SPREAD_MIN_COUNT) {
                    fresh.forEachIndexed { index, slug ->
                        firstDue[slug] = tomorrow + index.toLong() * SPREAD_DAYS / fresh.size
                    }
                }
                for (slug in unique) applySolved(slug, true, now, firstDue[slug] ?: tomorrow)
            } else {
                for (slug in unique) applySolved(slug, false, now, tomorrow)
            }
        }
        onUserDataChanged()
    }

    private suspend fun applySolved(slug: String, solved: Boolean, now: Long, firstDueDay: Long) {
        val current = progressDao.get(slug) ?: UserProgressEntity(slug = slug)
        val wasSolved = current.status == STATUS_SOLVED
        if (wasSolved != solved) {
            progressDao.upsert(
                current.copy(
                    status = if (solved) STATUS_SOLVED else STATUS_UNSOLVED,
                    solvedAt = if (solved) now else null,
                    updatedAt = now
                )
            )
            if (solved) {
                activityDao.insert(
                    ActivityLogEntity(
                        slug = slug,
                        type = LOG_SOLVED,
                        timestamp = now,
                        durationMs = 0L
                    )
                )
            }
        }
        if (solved) {
            if (reviewDao.get(slug) == null) {
                reviewDao.upsert(newReviewState(slug, firstDueDay))
            }
        } else {
            reviewDao.delete(slug)
        }
    }

    /** Matches pasted URLs / slugs / titles against every problem (nothing is changed). */
    suspend fun previewImport(text: String): ImportPreview {
        val problems = contentDao.getConcepts()
        val solved = progressDao.getAll()
            .filter { it.status == STATUS_SOLVED }
            .map { it.slug }
            .toSet()
        return ImportMatcher.match(text, problems, solved)
    }

    /**
     * Applies [rating] to the problem's review state and logs a REVIEW event.
     * Returns the new interval in days.
     */
    suspend fun rate(slug: String, rating: Rating): Int {
        val days = db.withTransaction {
            val today = LocalDate.now()
            val state = reviewDao.get(slug)?.toDomain()
                ?: ReviewState(slug = slug, dueDay = today.toEpochDay())
            val next = scheduler.schedule(state, rating, today)
            reviewDao.upsert(next.toEntity())
            activityDao.insert(
                ActivityLogEntity(
                    slug = slug,
                    type = LOG_REVIEW,
                    timestamp = System.currentTimeMillis(),
                    durationMs = 0L,
                    rating = rating.name
                )
            )
            next.intervalDays
        }
        onUserDataChanged()
        return days
    }

    /** Startup repair: every solved problem without a review state becomes due today. */
    suspend fun repairReviewStates() {
        db.withTransaction {
            val missing = reviewDao.solvedSlugsWithoutReview()
            if (missing.isNotEmpty()) {
                val today = LocalDate.now().toEpochDay()
                reviewDao.insertAllIgnore(missing.map { newReviewState(it, today) })
            }
        }
    }

    suspend fun dueCount(todayEpochDay: Long): Int = reviewDao.countDue(todayEpochDay)

    suspend fun activitySince(since: Long): List<ActivityLogEntity> = activityDao.getSince(since)

    suspend fun logView(slug: String, durationMs: Long) {
        activityDao.insert(
            ActivityLogEntity(
                slug = slug,
                type = LOG_VIEW,
                timestamp = System.currentTimeMillis(),
                durationMs = durationMs
            )
        )
    }

    /** One QUIZ row per answered pattern quiz (rating CORRECT / WRONG). */
    suspend fun logQuiz(slug: String, correct: Boolean) {
        activityDao.insert(
            ActivityLogEntity(
                slug = slug,
                type = LOG_QUIZ,
                timestamp = System.currentTimeMillis(),
                durationMs = 0L,
                rating = if (correct) QUIZ_CORRECT else QUIZ_WRONG
            )
        )
    }

    fun observeActivitySince(since: Long): Flow<List<ActivityLogEntity>> =
        activityDao.observeSince(since)

    private fun newReviewState(slug: String, dueDay: Long): ReviewStateEntity =
        ReviewStateEntity(
            slug = slug,
            ease = DEFAULT_EASE,
            intervalDays = 0,
            reps = 0,
            lapses = 0,
            dueDay = dueDay
        )

    private fun tomorrowEpochDay(): Long = LocalDate.now().toEpochDay() + 1

    companion object {
        const val DEFAULT_SHEET_ID = "neetcode150"
        const val STATUS_SOLVED = "SOLVED"
        const val STATUS_UNSOLVED = "UNSOLVED"
        const val LOG_VIEW = "VIEW"
        const val LOG_SOLVED = "SOLVED"
        const val LOG_REVIEW = "REVIEW"
        const val LOG_QUIZ = "QUIZ"
        const val QUIZ_CORRECT = "CORRECT"
        const val QUIZ_WRONG = "WRONG"

        /** The virtual "My problems" sheet (computed from userAdded, not stored). */
        const val MINE_SHEET_ID = "mine"
        const val MINE_SHEET_NAME = "My cards"

        /** The virtual "All problems" sheet: every problem once (computed, not stored). */
        const val ALL_SHEET_ID = "all"
        const val ALL_SHEET_NAME = "All cards"
        const val ALL_OTHER_SECTION = "Other"
        const val USER_SLUG_PREFIX = "user-"

        /** Bulk solves of at least this many problems spread their first reviews. */
        const val SPREAD_MIN_COUNT = 5
        const val SPREAD_DAYS = 14

        /** Display order of the bundled sheets. */
        val SHEET_ORDER = listOf(
            "neetcode150", "blind75", "striver-sde", "top-interview-150", "grind75"
        )
    }
}
