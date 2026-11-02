package com.parthdeuri.quickml.ui.feed

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.pager.VerticalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FilterList
import androidx.compose.material.icons.filled.Shuffle
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import android.os.SystemClock
import android.widget.Toast
import androidx.compose.ui.platform.LocalContext
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.Locale
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.parthdeuri.quickml.QuickMLApp
import com.parthdeuri.quickml.data.repo.FeedItem
import com.parthdeuri.quickml.data.repo.ConceptRepository
import com.parthdeuri.quickml.data.repo.SheetInfo
import com.parthdeuri.quickml.data.repo.toDraft
import com.parthdeuri.quickml.speech.LocalSpeaker
import com.parthdeuri.quickml.ui.common.UserConceptDialog
import com.parthdeuri.quickml.ui.sheet.SheetPicker
import com.parthdeuri.quickml.ui.theme.QuickMLTheme
import kotlinx.coroutines.launch

@Composable
fun FeedScreen(viewModel: FeedViewModel, onOpenSettings: () -> Unit = {}) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    var showFilters by rememberSaveable { mutableStateOf(false) }
    val error = state.error

    Box(modifier = Modifier.fillMaxSize()) {
        when {
            state.loading -> {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator(color = QuickMLTheme.colors.accent)
                }
            }

            error != null -> {
                Box(Modifier.fillMaxSize().padding(32.dp), contentAlignment = Alignment.Center) {
                    Text(
                        text = "Could not load cards.\n$error",
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            state.mode == FeedMode.PATTERNS -> PatternsContent(
                state = state,
                viewModel = viewModel
            )

            else -> FeedContent(
                state = state,
                viewModel = viewModel,
                onOpenFilters = { showFilters = true },
                onOpenSettings = onOpenSettings
            )
        }
    }

    if (showFilters && !state.loading && error == null) {
        FilterSheet(
            current = state.filters,
            allTopics = state.allTopics,
            onApply = { filters ->
                viewModel.applyFilters(filters)
                showFilters = false
            },
            onClear = {
                viewModel.clearFilters()
                showFilters = false
            },
            onDismiss = { showFilters = false }
        )
    }
}

@Composable
private fun FeedContent(
    state: FeedUiState,
    viewModel: FeedViewModel,
    onOpenFilters: () -> Unit,
    onOpenSettings: () -> Unit
) {
    // AI master toggle: the Ask tab exists only while it is on.
    val aiSettings = (LocalContext.current.applicationContext as QuickMLApp).container.aiSettings
    val aiEnabled by aiSettings.enabled.collectAsStateWithLifecycle()
    // Own problem being edited / about to be deleted (from a card's overflow menu).
    var editing by remember { mutableStateOf<FeedItem?>(null) }
    var deleting by remember { mutableStateOf<FeedItem?>(null) }

    // A fresh pager for each mode / filter combination (and sheet, for Learn), and for every
    // "open this problem" request.
    val sheetMode = state.mode == FeedMode.LEARN || state.mode == FeedMode.SHUFFLE
    val pagerSheet = if (sheetMode) state.sheetId else ""
    key(state.mode, state.filters, pagerSheet, state.openNonce) {
        val items = state.items
        val pagerMode = state.mode
        val pagerFilters = state.filters
        val startPage = remember {
            viewModel.restorePage(pagerMode, pagerFilters, pagerSheet, items)
        }
        val pagerState = rememberPagerState(initialPage = startPage) { items.size }
        val scope = rememberCoroutineScope()
        val context = LocalContext.current
        val speaker = LocalSpeaker.current

        LaunchedEffect(pagerState) {
            snapshotFlow { pagerState.currentPage }.collect { page ->
                // Another card: stop reading the previous one.
                speaker?.stop()
                viewModel.onPageChanged(pagerMode, pagerFilters, pagerSheet, page)
            }
        }
        // Mode / filter / sheet change or leaving the Feed tab: stop reading.
        DisposableEffect(speaker) {
            onDispose { speaker?.stop() }
        }
        val quizEnabled = pagerMode == FeedMode.QUIZ ||
            (pagerMode == FeedMode.REVISE && state.quizBeforeReveal)

        ViewTimeTracker(
            slug = items.getOrNull(pagerState.currentPage)?.problem?.slug,
            onLog = { slug, ms -> viewModel.logView(slug, ms) }
        )

        val titles = remember(items) {
            items.associate { it.problem.slug to it.problem.title }
        }
        val indexBySlug = remember(items) {
            items.withIndex().associate { it.value.problem.slug to it.index }
        }

        Column(modifier = Modifier.fillMaxSize()) {
            FeedTopBar(
                sheetName = state.sheetName,
                sheets = state.sheets,
                sheetId = state.sheetId,
                onSelectSheet = { viewModel.selectSheet(it) },
                position = if (items.isEmpty()) 0 else pagerState.currentPage.coerceIn(0, items.size - 1) + 1,
                total = items.size,
                mode = state.mode,
                dueCount = state.dueCount,
                filtersActive = state.filters.isActive,
                onModeChange = { viewModel.setMode(it) },
                onOpenFilters = onOpenFilters,
                showPatternsChip = state.patternIndex.isNotEmpty(),
                onReshuffle = { viewModel.reshuffle() }
            )

            if (items.isEmpty()) {
                EmptyState(
                    state = state,
                    onClearFilters = { viewModel.clearFilters() },
                    modifier = Modifier.weight(1f)
                )
            } else {
                VerticalPager(
                    state = pagerState,
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    beyondViewportPageCount = 1,
                    key = { page -> items.getOrNull(page)?.problem?.slug ?: page }
                ) { page ->
                    val item = items.getOrNull(page)
                    if (item != null) {
                        ConceptCard(
                            item = item,
                            titles = titles,
                            quizEnabled = quizEnabled,
                            patternPool = state.patternPool,
                            practiceMode = state.practiceMode,
                            onPracticeModeChange = { viewModel.setPracticeMode(it) },
                            onQuizAnswered = { correct ->
                                viewModel.logQuiz(item.problem.slug, correct)
                            },
                            onToggleBookmark = { viewModel.toggleBookmark(item.problem.slug) },
                            onToggleSolved = { viewModel.toggleSolved(item.problem.slug) },
                            onRate = { rating ->
                                viewModel.rate(item.problem.slug, rating) { days ->
                                    val unit = if (days == 1) " day" else " days"
                                    Toast.makeText(
                                        context,
                                        "Next review in " + days + unit,
                                        Toast.LENGTH_SHORT
                                    ).show()
                                    // Revise: move on to the next card (unless the user already did).
                                    if (pagerMode == FeedMode.REVISE &&
                                        pagerState.currentPage == page &&
                                        page + 1 < items.size
                                    ) {
                                        scope.launch { pagerState.animateScrollToPage(page + 1) }
                                    }
                                }
                            },
                            onRelatedClick = { slug ->
                                val target = indexBySlug[slug]
                                if (target != null) {
                                    scope.launch { pagerState.animateScrollToPage(target) }
                                }
                            },
                            isCurrent = pagerState.currentPage == page,
                            thinkTimerMinutes = if (pagerMode == FeedMode.REVISE || pagerMode == FeedMode.QUIZ) {
                                state.thinkTimerMinutes
                            } else {
                                0
                            },
                            patternIndex = state.patternIndex,
                            onOpenPattern = { id -> viewModel.openPattern(id) },
                            onSaveNote = { note -> viewModel.saveNote(item.problem.slug, note) },
                            onSaveMySolution = { code ->
                                viewModel.saveMySolution(item.problem.slug, code)
                            },
                            onEditConcept = { editing = item },
                            onDeleteConcept = { deleting = item },
                            aiEnabled = aiEnabled,
                            onOpenSettings = onOpenSettings
                        )
                    }
                }
            }
            Spacer(Modifier.navigationBarsPadding())
        }
    }

    val edit = editing
    if (edit != null) {
        UserConceptDialog(
            initial = remember(edit.problem.slug) { edit.problem.toDraft(edit.note) },
            onSave = { draft ->
                viewModel.saveUserConcept(edit.problem.slug, draft)
                editing = null
            },
            onDismiss = { editing = null }
        )
    }
    val delete = deleting
    if (delete != null) {
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("Delete card?") },
            text = {
                Text("\"" + delete.problem.title + "\" and its progress will be deleted.")
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        viewModel.deleteUserConcept(delete.problem.slug)
                        deleting = null
                    }
                ) { Text("Delete") }
            },
            dismissButton = {
                TextButton(onClick = { deleting = null }) { Text("Cancel") }
            }
        )
    }
}

/** Patterns mode: a vertical pager of pattern cards (no filters). */
@Composable
private fun PatternsContent(state: FeedUiState, viewModel: FeedViewModel) {
    key(state.openNonce) {
        val cards = state.patternCards
        val startPage = remember { viewModel.restorePatternPage(cards) }
        val pagerState = rememberPagerState(initialPage = startPage) { cards.size }
        LaunchedEffect(pagerState) {
            snapshotFlow { pagerState.currentPage }.collect { page ->
                if (cards.isNotEmpty()) viewModel.onPatternPageChanged(page)
            }
        }
        Column(modifier = Modifier.fillMaxSize()) {
            FeedTopBar(
                sheetName = state.sheetName,
                sheets = state.sheets,
                sheetId = state.sheetId,
                onSelectSheet = { viewModel.selectSheet(it) },
                position = if (cards.isEmpty()) 0 else pagerState.currentPage.coerceIn(0, cards.size - 1) + 1,
                total = cards.size,
                mode = state.mode,
                dueCount = state.dueCount,
                filtersActive = false,
                onModeChange = { viewModel.setMode(it) },
                onOpenFilters = null,
                showPatternsChip = true
            )
            if (cards.isEmpty()) {
                EmptyState(
                    state = state,
                    onClearFilters = { viewModel.clearFilters() },
                    modifier = Modifier.weight(1f)
                )
            } else {
                VerticalPager(
                    state = pagerState,
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    beyondViewportPageCount = 1,
                    key = { page -> cards.getOrNull(page)?.pattern?.id ?: page }
                ) { page ->
                    val card = cards.getOrNull(page)
                    if (card != null) {
                        PatternCard(
                            card = card,
                            onOpenConcept = { slug -> viewModel.openConceptAnywhere(slug) }
                        )
                    }
                }
            }
            Spacer(Modifier.navigationBarsPadding())
        }
    }
}

private val NextDueFormat: DateTimeFormatter =
    DateTimeFormatter.ofPattern("EEE, d MMM", Locale.getDefault())

private const val MIN_VIEW_MS = 3_000L
private const val MAX_VIEW_MS = 10 * 60 * 1_000L

/**
 * Times how long [slug] stays on screen and reports one VIEW per visit: when the slug changes,
 * the composable leaves composition, or the app stops. Visits under 3 s are dropped, long ones
 * are capped at 10 min.
 */
@Composable
private fun ViewTimeTracker(slug: String?, onLog: (String, Long) -> Unit) {
    val currentOnLog by rememberUpdatedState(onLog)
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(slug, lifecycleOwner) {
        var startedAt = -1L
        if (slug != null && lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) {
            startedAt = SystemClock.elapsedRealtime()
        }
        fun flush() {
            if (slug != null && startedAt >= 0L) {
                val elapsed = SystemClock.elapsedRealtime() - startedAt
                startedAt = -1L
                if (elapsed >= MIN_VIEW_MS) {
                    currentOnLog(slug, minOf(elapsed, MAX_VIEW_MS))
                }
            }
        }
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_STOP -> flush()
                Lifecycle.Event.ON_START -> {
                    if (slug != null && startedAt < 0L) startedAt = SystemClock.elapsedRealtime()
                }
                else -> Unit
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            flush()
        }
    }
}

@Composable
private fun FeedTopBar(
    sheetName: String,
    sheets: List<SheetInfo>,
    sheetId: String,
    onSelectSheet: (String) -> Unit,
    position: Int,
    total: Int,
    mode: FeedMode,
    dueCount: Int,
    filtersActive: Boolean,
    onModeChange: (FeedMode) -> Unit,
    /** Null hides the filter button (Patterns mode). */
    onOpenFilters: (() -> Unit)?,
    /** The Patterns chip shows only when the content has a pattern library. */
    showPatternsChip: Boolean,
    /** Shown only in Shuffle mode. */
    onReshuffle: () -> Unit = {}
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .statusBarsPadding()
            .padding(horizontal = 16.dp, vertical = 4.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            SheetPicker(
                name = sheetName,
                sheets = sheets,
                selectedId = sheetId,
                onSelect = onSelectSheet,
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.weight(1f)
            )
            Text(
                text = "$position / $total",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            if (mode == FeedMode.SHUFFLE) {
                IconButton(onClick = onReshuffle) {
                    Icon(
                        imageVector = Icons.Filled.Shuffle,
                        contentDescription = "Reshuffle",
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            if (onOpenFilters != null) {
                IconButton(onClick = onOpenFilters) {
                    Icon(
                        imageVector = Icons.Filled.FilterList,
                        contentDescription = "Filters",
                        tint = if (filtersActive) {
                            QuickMLTheme.colors.accent
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        }
                    )
                }
            } else {
                Spacer(Modifier.height(48.dp))
            }
        }
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            FilterChip(
                selected = mode == FeedMode.LEARN,
                onClick = { onModeChange(FeedMode.LEARN) },
                label = { Text("Learn") }
            )
            FilterChip(
                selected = mode == FeedMode.SHUFFLE,
                onClick = { onModeChange(FeedMode.SHUFFLE) },
                label = { Text("Shuffle") }
            )
            FilterChip(
                selected = mode == FeedMode.REVISE,
                onClick = { onModeChange(FeedMode.REVISE) },
                label = { Text("Revise · " + dueCount) }
            )
            FilterChip(
                selected = mode == FeedMode.QUIZ,
                onClick = { onModeChange(FeedMode.QUIZ) },
                label = { Text("Quiz") }
            )
            if (showPatternsChip || mode == FeedMode.PATTERNS) {
                FilterChip(
                    selected = mode == FeedMode.PATTERNS,
                    onClick = { onModeChange(FeedMode.PATTERNS) },
                    label = { Text("Families") }
                )
            }
            FilterChip(
                selected = mode == FeedMode.BOOKMARKS,
                onClick = { onModeChange(FeedMode.BOOKMARKS) },
                label = { Text("Bookmarks") }
            )
        }
        Spacer(Modifier.height(4.dp))
    }
}

@Composable
private fun EmptyState(
    state: FeedUiState,
    onClearFilters: () -> Unit,
    modifier: Modifier = Modifier
) {
    // Filters don't apply to Patterns mode.
    val filtersActive = state.filters.isActive && state.mode != FeedMode.PATTERNS
    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        val message = when {
            filtersActive -> "No cards match your filters."
            state.mode == FeedMode.BOOKMARKS -> "No bookmarks yet. Double-tap a card to save it."
            state.mode == FeedMode.REVISE && state.solvedCount == 0 ->
                "Mark cards learned to start revising"
            state.mode == FeedMode.REVISE -> "All caught up 🎉"
            state.mode == FeedMode.QUIZ -> "No cards with full notes to quiz on yet."
            state.mode == FeedMode.PATTERNS -> "No technique families in this version of the content yet."
            (state.mode == FeedMode.LEARN || state.mode == FeedMode.SHUFFLE) &&
                state.sheetId == ConceptRepository.MINE_SHEET_ID ->
                "No cards of your own yet. Add one from the Sheet tab with +."
            else -> "No cards found."
        }
        Text(
            text = message,
            textAlign = TextAlign.Center,
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        val nextDay = state.nextDueDay
        if (!filtersActive && state.mode == FeedMode.REVISE && state.solvedCount > 0 && nextDay != null) {
            val cards = if (state.nextDueCount == 1) "1 card" else state.nextDueCount.toString() + " cards"
            val whenText = if (nextDay == state.todayEpochDay + 1) {
                "tomorrow"
            } else {
                "on " + LocalDate.ofEpochDay(nextDay).format(NextDueFormat)
            }
            Spacer(Modifier.height(8.dp))
            Text(
                text = "Next: $cards $whenText",
                textAlign = TextAlign.Center,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        if (filtersActive) {
            Spacer(Modifier.height(16.dp))
            Button(onClick = onClearFilters) { Text("Clear filters") }
        }
    }
}
