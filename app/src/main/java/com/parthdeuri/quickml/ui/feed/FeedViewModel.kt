package com.parthdeuri.quickml.ui.feed

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.parthdeuri.quickml.AppContainer
import com.parthdeuri.quickml.data.repo.FeedData
import com.parthdeuri.quickml.data.repo.FeedItem
import com.parthdeuri.quickml.data.repo.PatternInfo
import com.parthdeuri.quickml.data.repo.ProblemRepository
import com.parthdeuri.quickml.data.repo.SheetInfo
import com.parthdeuri.quickml.data.repo.UserProblemDraft
import com.parthdeuri.quickml.data.settings.CardPrefs
import com.parthdeuri.quickml.domain.srs.Rating
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.time.LocalDate
import kotlin.random.Random

/**
 * LEARN = the selected sheet in order, SHUFFLE = the selected sheet in a random order (fixed per
 * session, the button reshuffles), REVISE = solved problems due today (all sheets),
 * QUIZ = full problems with a pattern in a daily shuffle (all sheets), PATTERNS = the pattern
 * library (pattern cards, not problems), BOOKMARKS = saved ones.
 */
enum class FeedMode { LEARN, SHUFFLE, REVISE, QUIZ, PATTERNS, BOOKMARKS }

/** A pattern reachable from a problem's pattern string (alias lookup). */
data class PatternRef(val id: String, val name: String)

/** A problem linked from a pattern card (Problems tab). */
data class PatternProblemRow(
    val slug: String,
    val title: String,
    val difficulty: String,
    val solved: Boolean
)

/** One card of Patterns mode. */
data class PatternCardUi(
    val pattern: PatternInfo,
    /** Linked problems that exist in the content, best first. */
    val problems: List<PatternProblemRow>
)

data class Filters(
    val difficulties: Set<String> = emptySet(),
    val topics: Set<String> = emptySet(),
    val onlyFull: Boolean = false,
    /** Subset of "SOLVED" / "UNSOLVED"; empty means no status filter. */
    val statuses: Set<String> = emptySet()
) {
    val isActive: Boolean
        get() = difficulties.isNotEmpty() || topics.isNotEmpty() || onlyFull || statuses.isNotEmpty()
}

data class FeedUiState(
    val loading: Boolean = true,
    val error: String? = null,
    val sheetName: String = "Revise",
    val sheetId: String = "",
    val sheets: List<SheetInfo> = emptyList(),
    val mode: FeedMode = FeedMode.LEARN,
    val filters: Filters = Filters(),
    val items: List<FeedItem> = emptyList(),
    val allTopics: List<String> = emptyList(),
    /** Solved problems due today or earlier (live: drops as cards are rated). */
    val dueCount: Int = 0,
    val solvedCount: Int = 0,
    val todayEpochDay: Long = 0L,
    /** Earliest due day after today and how many cards fall on it; null when nothing is scheduled. */
    val nextDueDay: Long? = null,
    val nextDueCount: Int = 0,
    /** Distinct patterns of all full problems (pattern-quiz distractors), sorted. */
    val patternPool: List<String> = emptyList(),
    val quizBeforeReveal: Boolean = true,
    val practiceMode: Boolean = false,
    /** Think timer (minutes) for the Problem tab in Revise / Quiz modes; 0 = off. */
    val thinkTimerMinutes: Int = 0,
    /** Pattern string / name / id (lowercase) -> pattern; empty when the content has none. */
    val patternIndex: Map<String, PatternRef> = emptyMap(),
    /** Patterns mode only: every pattern card. */
    val patternCards: List<PatternCardUi> = emptyList(),
    /** Bumped by "open this problem / pattern" requests so a fresh pager is built. */
    val openNonce: Int = 0
)

private sealed interface ImportStatus {
    object Loading : ImportStatus
    object Ready : ImportStatus
    class Failed(val message: String) : ImportStatus
}

@OptIn(ExperimentalCoroutinesApi::class)
class FeedViewModel(private val container: AppContainer) : ViewModel() {
    private val prefs = container.prefs
    private val repository = container.repository
    private val settings = container.studySettings

    private val importStatus = MutableStateFlow<ImportStatus>(ImportStatus.Loading)
    private val mode = MutableStateFlow(readSavedMode())
    private val filters = MutableStateFlow(Filters())
    private val reviseTick = MutableStateFlow(0)
    private val openNonce = MutableStateFlow(0)

    /** Last page while filters are active (not persisted); lets the feed survive tab switches. */
    private var filteredPage = 0

    /** Current page of the Revise queue (not persisted: the queue changes every session). */
    private var reviseCurrentPage = 0

    /** Seed of the Shuffle order: new per ViewModel (app session) and on every reshuffle. */
    private var shuffleSeed = Random.nextLong()

    /** Current page of the Shuffle deck (not persisted: the order changes every session). */
    private var shufflePage = 0

    /**
     * Slugs of the Revise queue, frozen when the queue is opened so rated cards stay in the list
     * (and the order doesn't reshuffle) until the feed is opened again.
     */
    private var frozenRevise: List<String>? = null

    /**
     * Set by the Sheet tab / widget / pattern cards; consumed by the next Learn pager that
     * contains the problem.
     */
    private var pendingOpenSlug: String? = null

    /** Set by "Review pattern" links; consumed by the next Patterns pager. */
    private var pendingPatternId: String? = null

    val uiState: StateFlow<FeedUiState> = importStatus
        .flatMapLatest { status ->
            when (status) {
                is ImportStatus.Loading -> flowOf(FeedUiState(loading = true))
                is ImportStatus.Failed -> flowOf(
                    FeedUiState(loading = false, error = status.message)
                )
                is ImportStatus.Ready -> readyFlow()
            }
        }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), FeedUiState())

    init {
        viewModelScope.launch {
            try {
                container.importer.importIfNeeded()
                repository.repairReviewStates()
                importStatus.value = ImportStatus.Ready
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                importStatus.value = ImportStatus.Failed(e.message ?: "Could not load problems")
            }
        }
    }

    private fun readyFlow(): Flow<FeedUiState> =
        combine(
            mode,
            filters,
            combine(reviseTick, openNonce) { _, nonce -> nonce },
            combine(
                repository.observeFeed(settings.selectedSheetId),
                repository.observePatterns()
            ) { data, patterns -> data to patterns },
            settings.cardPrefs
        ) { m, f, nonce, content, cardPrefs ->
            buildState(content.first, content.second, m, f, cardPrefs, nonce)
        }

    private fun buildState(
        data: FeedData,
        patterns: List<PatternInfo>,
        m: FeedMode,
        f: Filters,
        cardPrefs: CardPrefs,
        nonce: Int
    ): FeedUiState {
        // Learn follows the selected sheet; every other mode works across all sheets.
        val all = data.allItems
        val today = LocalDate.now().toEpochDay()
        val allTopics = all.flatMap { it.problem.topics }.distinct().sorted()

        val solved = all.filter { it.solved }
        val dueCount = solved.count { (it.dueDay ?: Long.MAX_VALUE) <= today }
        val futureDays = solved.mapNotNull { it.dueDay }.filter { it > today }
        val nextDueDay = futureDays.minOrNull()
        val nextDueCount = if (nextDueDay == null) 0 else futureDays.count { it == nextDueDay }

        val base = when (m) {
            FeedMode.LEARN -> data.items
            FeedMode.SHUFFLE ->
                data.items.sortedBy { it.problem.slug }.shuffled(Random(shuffleSeed))
            FeedMode.BOOKMARKS -> all.filter { it.bookmarked }
            FeedMode.REVISE -> reviseItems(all, today)
            FeedMode.QUIZ -> quizItems(all, today)
            FeedMode.PATTERNS -> emptyList()
        }
        val items = base.filter { item ->
            (f.difficulties.isEmpty() || item.problem.difficulty in f.difficulties) &&
                (f.topics.isEmpty() || item.problem.topics.any { it in f.topics }) &&
                (!f.onlyFull || item.isFull) &&
                (f.statuses.isEmpty() || item.status in f.statuses)
        }
        return FeedUiState(
            loading = false,
            error = null,
            sheetName = data.sheetName,
            sheetId = data.sheetId,
            sheets = data.sheets,
            mode = m,
            filters = f,
            items = items,
            allTopics = allTopics,
            dueCount = dueCount,
            solvedCount = solved.size,
            todayEpochDay = today,
            nextDueDay = nextDueDay,
            nextDueCount = nextDueCount,
            patternPool = all
                .filter { it.isFull }
                .flatMap { it.problem.patterns }
                .distinct()
                .sorted(),
            quizBeforeReveal = cardPrefs.quizBeforeReveal,
            practiceMode = cardPrefs.practiceMode,
            thinkTimerMinutes = cardPrefs.thinkTimerMinutes,
            patternIndex = patternIndex(patterns),
            patternCards = if (m == FeedMode.PATTERNS) patternCards(patterns, all) else emptyList(),
            openNonce = nonce
        )
    }

    /** Aliases first (they are what problems use), then names and ids. */
    private fun patternIndex(patterns: List<PatternInfo>): Map<String, PatternRef> {
        val index = HashMap<String, PatternRef>()
        fun put(key: String, ref: PatternRef) {
            val k = key.trim().lowercase()
            if (k.isNotEmpty() && k !in index) index[k] = ref
        }
        val refs = patterns.map { PatternRef(it.id, it.name) }
        patterns.forEachIndexed { i, p -> p.aliases.forEach { put(it, refs[i]) } }
        patterns.forEachIndexed { i, p ->
            put(p.name, refs[i])
            put(p.id, refs[i])
        }
        return index
    }

    private fun patternCards(patterns: List<PatternInfo>, all: List<FeedItem>): List<PatternCardUi> {
        val bySlug = all.associateBy { it.problem.slug }
        return patterns.map { pattern ->
            PatternCardUi(
                pattern = pattern,
                problems = pattern.problems.distinct().mapNotNull { slug ->
                    bySlug[slug]?.let { item ->
                        PatternProblemRow(
                            slug = slug,
                            title = item.problem.title,
                            difficulty = item.problem.difficulty,
                            solved = item.solved
                        )
                    }
                }
            )
        }
    }

    /** Full problems with a pattern, shuffled with a seed derived from the day (stable all day). */
    private fun quizItems(all: List<FeedItem>, today: Long): List<FeedItem> =
        all.filter { it.isFull && it.problem.patterns.isNotEmpty() }
            .sortedBy { it.problem.slug }
            .shuffled(Random(today))

    /** The frozen Revise queue, mapped onto the live items so ratings and bookmarks show up. */
    private fun reviseItems(all: List<FeedItem>, today: Long): List<FeedItem> {
        var slugs = frozenRevise
        if (slugs == null) {
            slugs = computeReviseOrder(all, today)
            if (all.isNotEmpty()) frozenRevise = slugs
        }
        val bySlug = all.associateBy { it.problem.slug }
        return slugs.mapNotNull { bySlug[it] }
    }

    /**
     * Due problems, most overdue first (ties shuffled with a seed derived from the day so the
     * order is stable all day), then re-arranged so that two consecutive cards don't share their
     * primary topic whenever that can be avoided.
     */
    private fun computeReviseOrder(all: List<FeedItem>, today: Long): List<String> {
        val due = all.filter { it.solved && (it.dueDay ?: Long.MAX_VALUE) <= today }
        if (due.isEmpty()) return emptyList()

        val random = Random(today)
        val ordered = due
            .groupBy { it.dueDay ?: today }
            .toSortedMap()
            .values
            .flatMap { group -> group.shuffled(random) }

        val remaining = ordered.toMutableList()
        val result = ArrayList<FeedItem>(remaining.size)
        var lastTopic: String? = null
        while (remaining.isNotEmpty()) {
            var index = 0
            if (result.isNotEmpty()) {
                val found = remaining.indexOfFirst { it.problem.topics.firstOrNull() != lastTopic }
                if (found >= 0) index = found
            }
            val next = remaining.removeAt(index)
            result.add(next)
            lastTopic = next.problem.topics.firstOrNull()
        }
        return result.map { it.problem.slug }
    }

    /** New Shuffle order, back at the first card. */
    fun reshuffle() {
        shuffleSeed = Random.nextLong()
        shufflePage = 0
        filteredPage = 0
        openNonce.value += 1
    }

    fun setMode(newMode: FeedMode) {
        filteredPage = 0
        pendingOpenSlug = null
        pendingPatternId = null
        if (newMode == FeedMode.REVISE) {
            frozenRevise = null
            reviseCurrentPage = 0
        }
        prefs.edit().putString(KEY_MODE, newMode.name).apply()
        mode.value = newMode
    }

    /** Called when the Feed tab is (re)opened: rebuilds the Revise queue from scratch. */
    fun onFeedOpened() {
        if (mode.value == FeedMode.REVISE) {
            frozenRevise = null
            reviseCurrentPage = 0
            reviseTick.value += 1
        }
    }

    /** Changes the selected sheet everywhere (Feed, Sheet tab, Dashboard). */
    fun selectSheet(id: String) {
        if (mode.value == FeedMode.LEARN || mode.value == FeedMode.SHUFFLE) filteredPage = 0
        settings.selectSheet(id)
    }

    fun setPracticeMode(on: Boolean) {
        settings.setPracticeMode(on)
    }

    /** Sheet tab -> Feed tab: Learn mode, no filters, scrolled to [slug]. */
    fun openProblem(slug: String) {
        pendingOpenSlug = slug
        pendingPatternId = null
        filteredPage = 0
        prefs.edit().putString(KEY_MODE, FeedMode.LEARN.name).apply()
        mode.value = FeedMode.LEARN
        filters.value = Filters()
        openNonce.value += 1
    }

    /**
     * Widget / pattern card -> [slug] in Learn mode; switches to a sheet that contains it when
     * the selected one doesn't ("My problems" for own problems).
     */
    fun openProblemAnywhere(slug: String) {
        viewModelScope.launch {
            val target = try {
                repository.sheetFor(slug, settings.selectedSheetId.value)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                null
            }
            if (target != null) settings.selectSheet(target)
            openProblem(slug)
        }
    }

    /** Quiz answer / pattern pill -> Patterns mode at that pattern's card. */
    fun openPattern(id: String) {
        pendingPatternId = id
        pendingOpenSlug = null
        prefs.edit().putString(KEY_MODE, FeedMode.PATTERNS.name).apply()
        mode.value = FeedMode.PATTERNS
        openNonce.value += 1
    }

    /** Fresh page to start the Patterns pager on. */
    fun restorePatternPage(cards: List<PatternCardUi>): Int {
        val pending = pendingPatternId
        if (pending != null && cards.isNotEmpty()) {
            pendingPatternId = null
            val index = cards.indexOfFirst { it.pattern.id == pending }
            if (index >= 0) return index
        }
        return prefs.getInt(pageKey(FeedMode.PATTERNS, ""), 0)
            .coerceIn(0, (cards.size - 1).coerceAtLeast(0))
    }

    fun onPatternPageChanged(page: Int) {
        prefs.edit().putInt(pageKey(FeedMode.PATTERNS, ""), page).apply()
    }

    fun applyFilters(newFilters: Filters) {
        filteredPage = 0
        filters.value = newFilters
    }

    fun clearFilters() {
        filteredPage = 0
        filters.value = Filters()
    }

    /**
     * [m], [f] and [sheetId] are the mode/filters/sheet the reporting pager was built for (it may
     * be stale).
     */
    fun onPageChanged(m: FeedMode, f: Filters, sheetId: String, page: Int) {
        when {
            m == FeedMode.REVISE -> reviseCurrentPage = page
            m == FeedMode.SHUFFLE -> shufflePage = page
            f.isActive -> filteredPage = page
            else -> prefs.edit().putInt(pageKey(m, sheetId), page).apply()
        }
    }

    /** Fresh page to start a pager on (never stale, unlike FeedUiState). */
    fun restorePage(m: FeedMode, f: Filters, sheetId: String, items: List<FeedItem>): Int {
        val pending = pendingOpenSlug
        if (pending != null && m == FeedMode.LEARN && !f.isActive) {
            // Not found: the sheet switch may still be on its way; keep it for the next pager.
            val index = items.indexOfFirst { it.problem.slug == pending }
            if (index >= 0) {
                pendingOpenSlug = null
                return index
            }
        }
        val saved = when {
            m == FeedMode.REVISE -> reviseCurrentPage
            m == FeedMode.SHUFFLE -> shufflePage
            f.isActive -> filteredPage
            else -> prefs.getInt(pageKey(m, sheetId), 0)
        }
        return saved.coerceIn(0, (items.size - 1).coerceAtLeast(0))
    }

    fun toggleBookmark(slug: String) {
        viewModelScope.launch { repository.toggleBookmark(slug) }
    }

    fun toggleSolved(slug: String) {
        viewModelScope.launch { repository.toggleSolved(slug) }
    }

    /** Saves the rating, then reports the new interval (days) back on the main thread. */
    fun rate(slug: String, rating: Rating, onDone: (Int) -> Unit) {
        viewModelScope.launch {
            val days = repository.rate(slug, rating)
            onDone(days)
        }
    }

    fun logView(slug: String, durationMs: Long) {
        viewModelScope.launch { repository.logView(slug, durationMs) }
    }

    fun logQuiz(slug: String, correct: Boolean) {
        viewModelScope.launch { repository.logQuiz(slug, correct) }
    }

    fun saveNote(slug: String, note: String) {
        viewModelScope.launch { repository.saveNote(slug, note) }
    }

    fun saveMySolution(slug: String, code: String) {
        viewModelScope.launch { repository.saveMySolution(slug, code) }
    }

    /** Edit of an own problem from its card. */
    fun saveUserProblem(slug: String, draft: UserProblemDraft) {
        viewModelScope.launch { repository.saveUserProblem(slug, draft) }
    }

    fun deleteUserProblem(slug: String) {
        viewModelScope.launch { repository.deleteUserProblem(slug) }
    }

    private fun readSavedMode(): FeedMode {
        val name = prefs.getString(KEY_MODE, null)
        // v0.1/v0.2 stored "ALL" for what is now Learn.
        if (name == "ALL") return FeedMode.LEARN
        return FeedMode.values().firstOrNull { it.name == name } ?: FeedMode.LEARN
    }

    /**
     * Learn saves one page per sheet; NeetCode 150 keeps the v0.1 key so an existing saved page
     * survives the upgrade. Other modes are cross-sheet and save one page each.
     */
    private fun pageKey(m: FeedMode, sheetId: String): String = when {
        m != FeedMode.LEARN -> "last_page_" + m.name
        sheetId.isEmpty() || sheetId == ProblemRepository.DEFAULT_SHEET_ID -> "last_page_ALL"
        else -> "last_page_LEARN_" + sheetId
    }

    private companion object {
        const val KEY_MODE = "last_mode"
    }
}

class FeedViewModelFactory(private val container: AppContainer) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T = FeedViewModel(container) as T
}
