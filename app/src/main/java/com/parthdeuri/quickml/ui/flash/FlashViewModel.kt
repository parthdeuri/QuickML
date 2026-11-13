package com.parthdeuri.quickml.ui.flash

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.parthdeuri.quickml.AppContainer
import com.parthdeuri.quickml.data.db.FlashcardEntity
import com.parthdeuri.quickml.data.db.SheetEntity
import com.parthdeuri.quickml.data.repo.FlashCounts
import com.parthdeuri.quickml.domain.srs.Rating
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/** The card types shown as filter chips; each covers one or more flash card kinds. */
enum class FlashGroup(val label: String, val kinds: Set<String>) {
    QUESTIONS("Questions", setOf("qa")),
    CONCEPTS("Concepts", setOf("define", "idea")),
    USE("Use and pitfalls", setOf("use", "pitfalls")),
    FORMULAS("Formulas", setOf("math")),
    COMPARE("Compare and families", setOf("compare", "family"))
}

/** One study session: the queue is fixed when it starts. */
data class FlashSession(
    val queue: List<FlashcardEntity>,
    val index: Int = 0,
    val flipped: Boolean = false,
    /** Interval in days each rating would give for the current card. */
    val preview: Map<Rating, Int> = emptyMap(),
    /** How many cards got each rating so far. */
    val tally: Map<Rating, Int> = emptyMap()
) {
    val current: FlashcardEntity?
        get() = queue.getOrNull(index)
}

data class FlashUiState(
    val decks: List<SheetEntity> = emptyList(),
    /** A sheet id, or null for all decks. */
    val deck: String? = null,
    val groups: Set<FlashGroup> = FlashGroup.values().toSet(),
    val newLimit: Int = 20,
    val counts: FlashCounts = FlashCounts(due = 0, newToday = 0, total = 0),
    val session: FlashSession? = null
)

class FlashViewModel(private val container: AppContainer) : ViewModel() {
    private val repository = container.flashRepository
    private val study = container.studySettings

    private val mutable = MutableStateFlow(FlashUiState(newLimit = study.flashNewPerDay.value))
    val state: StateFlow<FlashUiState> = mutable.asStateFlow()

    /** Guards against a double tap rating the same card twice. */
    private var busy = false

    init {
        viewModelScope.launch {
            repository.observeDecks().collect { decks ->
                mutable.update { it.copy(decks = decks) }
                refreshCounts()
            }
        }
    }

    private fun kindsOf(groups: Set<FlashGroup>): Set<String> = groups.flatMap { it.kinds }.toSet()

    private fun refreshCounts() {
        viewModelScope.launch {
            val s = mutable.value
            val counts = repository.counts(s.deck, kindsOf(s.groups), s.newLimit)
            mutable.update { it.copy(counts = counts) }
        }
    }

    fun selectDeck(deck: String?) {
        mutable.update { it.copy(deck = deck) }
        refreshCounts()
    }

    fun toggleGroup(group: FlashGroup) {
        mutable.update {
            val next = if (group in it.groups) it.groups - group else it.groups + group
            it.copy(groups = next)
        }
        refreshCounts()
    }

    fun setNewLimit(value: Int) {
        study.setFlashNewPerDay(value)
        mutable.update { it.copy(newLimit = study.flashNewPerDay.value) }
        refreshCounts()
    }

    fun start() {
        viewModelScope.launch {
            val s = mutable.value
            val queue = repository.buildQueue(s.deck, kindsOf(s.groups), s.newLimit)
            if (queue.isEmpty()) {
                refreshCounts()
                return@launch
            }
            val preview = repository.preview(queue[0].id)
            mutable.update { it.copy(session = FlashSession(queue = queue, preview = preview)) }
        }
    }

    fun flip() {
        mutable.update { s ->
            val session = s.session ?: return@update s
            s.copy(session = session.copy(flipped = !session.flipped))
        }
    }

    fun rate(rating: Rating) {
        val session = mutable.value.session ?: return
        val card = session.current ?: return
        if (!session.flipped || busy) return
        busy = true
        viewModelScope.launch {
            try {
                repository.rate(card, rating)
                val nextIndex = session.index + 1
                val next = session.queue.getOrNull(nextIndex)
                val preview = if (next != null) repository.preview(next.id) else emptyMap()
                val tally = session.tally + (rating to ((session.tally[rating] ?: 0) + 1))
                mutable.update {
                    it.copy(
                        session = session.copy(
                            index = nextIndex,
                            flipped = false,
                            preview = preview,
                            tally = tally
                        )
                    )
                }
            } finally {
                busy = false
            }
        }
    }

    /** Back to the deck picker (also after the last card); refreshes the counts. */
    fun exitSession() {
        mutable.update { it.copy(session = null) }
        refreshCounts()
    }
}

class FlashViewModelFactory(private val container: AppContainer) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T = FlashViewModel(container) as T
}
