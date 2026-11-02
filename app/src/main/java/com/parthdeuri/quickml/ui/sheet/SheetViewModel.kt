package com.parthdeuri.quickml.ui.sheet

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.parthdeuri.quickml.AppContainer
import com.parthdeuri.quickml.data.repo.FeedData
import com.parthdeuri.quickml.data.repo.ConceptRepository
import com.parthdeuri.quickml.data.repo.SheetInfo
import com.parthdeuri.quickml.data.repo.UserConceptDraft
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.time.LocalDate

/** Primitive-only so Compose can skip unchanged rows. */
data class SheetRow(
    val slug: String,
    val title: String,
    val difficulty: String,
    val solved: Boolean,
    val bookmarked: Boolean,
    val hasNotes: Boolean,
    val due: Boolean,
    /** The user wrote a note ("My notes"). */
    val hasNote: Boolean = false,
    /** One of the user's own problems (long-press: edit / delete). */
    val userAdded: Boolean = false
)

data class SheetSectionUi(
    val name: String,
    val solved: Int,
    val total: Int,
    val collapsed: Boolean,
    val rows: List<SheetRow>
)

data class SheetUiState(
    val loading: Boolean = true,
    val sheetId: String = "",
    val sheetName: String = "",
    val sheets: List<SheetInfo> = emptyList(),
    val solved: Int = 0,
    val total: Int = 0,
    val sections: List<SheetSectionUi> = emptyList()
)

class SheetViewModel(container: AppContainer) : ViewModel() {
    private val repository = container.repository
    private val settings = container.studySettings

    private val query = MutableStateFlow("")

    /**
     * Collapsed sections as "sheetId/section" (names repeat across sheets); kept for the
     * lifetime of this ViewModel (in-session).
     */
    private val collapsed = MutableStateFlow<Set<String>>(emptySet())

    /** Slugs per section of the whole sheet (ignores the search), for the bulk actions. */
    private var sectionSlugs: Map<String, List<String>> = emptyMap()

    /** Sheet the current state was built for. */
    private var currentSheetId: String = ""

    val state: StateFlow<SheetUiState> = combine(
        repository.observeFeed(settings.selectedSheetId),
        query,
        collapsed
    ) { feed, q, collapsedNames ->
        build(feed, q, collapsedNames)
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), SheetUiState())

    private fun build(feed: FeedData, rawQuery: String, collapsedNames: Set<String>): SheetUiState {
        val items = feed.items
        // No sheets yet = content still importing.
        if (feed.sheets.isEmpty()) return SheetUiState(loading = true, sheetName = feed.sheetName)
        currentSheetId = feed.sheetId

        val today = LocalDate.now().toEpochDay()
        val q = rawQuery.trim()
        val groups = items.groupBy { it.section }
        sectionSlugs = groups.mapValues { entry -> entry.value.map { it.problem.slug } }

        val sections = groups.mapNotNull { (name, group) ->
            val rows = group
                .filter { q.isEmpty() || it.problem.title.contains(q, ignoreCase = true) }
                .map { item ->
                    SheetRow(
                        slug = item.problem.slug,
                        title = item.problem.title,
                        difficulty = item.problem.difficulty,
                        solved = item.solved,
                        bookmarked = item.bookmarked,
                        hasNotes = item.isFull,
                        due = item.solved && (item.dueDay ?: Long.MAX_VALUE) <= today,
                        hasNote = item.hasNote,
                        userAdded = item.userAdded
                    )
                }
            if (q.isNotEmpty() && rows.isEmpty()) {
                null
            } else {
                SheetSectionUi(
                    name = name,
                    solved = group.count { it.solved },
                    total = group.size,
                    // Matches are always shown while searching.
                    collapsed = q.isEmpty() && collapseKey(feed.sheetId, name) in collapsedNames,
                    rows = rows
                )
            }
        }
        return SheetUiState(
            loading = false,
            sheetId = feed.sheetId,
            sheetName = feed.sheetName,
            sheets = feed.sheets,
            solved = items.count { it.solved },
            total = items.size,
            sections = sections
        )
    }

    fun setQuery(value: String) {
        query.value = value
    }

    fun toggleCollapsed(section: String) {
        val key = collapseKey(currentSheetId, section)
        val current = collapsed.value
        collapsed.value = if (key in current) current - key else current + key
    }

    /** Changes the selected sheet everywhere (Feed, Sheet tab, Dashboard). */
    fun selectSheet(id: String) {
        settings.selectSheet(id)
    }

    private fun collapseKey(sheetId: String, section: String): String = "$sheetId/$section"

    fun toggleSolved(slug: String) {
        viewModelScope.launch { repository.toggleSolved(slug) }
    }

    fun markSection(section: String, solved: Boolean) {
        val slugs = sectionSlugs[section] ?: return
        viewModelScope.launch { repository.setSolved(slugs, solved) }
    }

    /** The edit form of an own problem (null when it isn't one). */
    suspend fun loadDraft(slug: String): UserConceptDraft? = repository.loadUserConceptDraft(slug)

    /**
     * Adds ([slug] null) or updates an own problem. A new problem switches to "My problems" so
     * it shows up right away; [onDone] gets whether it was new.
     */
    fun saveConcept(slug: String?, draft: UserConceptDraft, onDone: (Boolean) -> Unit) {
        viewModelScope.launch {
            repository.saveUserConcept(slug, draft)
            if (slug == null) settings.selectSheet(ConceptRepository.MINE_SHEET_ID)
            onDone(slug == null)
        }
    }

    fun deleteConcept(slug: String) {
        viewModelScope.launch { repository.deleteUserConcept(slug) }
    }
}

class SheetViewModelFactory(private val container: AppContainer) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T = SheetViewModel(container) as T
}
