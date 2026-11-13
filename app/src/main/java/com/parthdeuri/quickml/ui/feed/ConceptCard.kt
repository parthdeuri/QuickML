package com.parthdeuri.quickml.ui.feed

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.widget.Toast
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.Image
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.OpenInNew
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.filled.Bookmark
import androidx.compose.material.icons.filled.BookmarkBorder
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.parthdeuri.quickml.data.content.ExampleJson
import com.parthdeuri.quickml.data.repo.FeedItem
import com.parthdeuri.quickml.domain.srs.Rating
import com.parthdeuri.quickml.domain.srs.Sm2Scheduler
import com.parthdeuri.quickml.domain.srs.formatInterval
import com.parthdeuri.quickml.domain.srs.label
import com.parthdeuri.quickml.speech.LocalSpeaker
import com.parthdeuri.quickml.ui.common.MathText
import com.parthdeuri.quickml.speech.approachSpeech
import com.parthdeuri.quickml.speech.intuitionSpeech
import com.parthdeuri.quickml.ui.common.SolutionEditorDialog
import com.parthdeuri.quickml.ui.theme.QuickMLTheme
import java.time.LocalDate
import kotlin.random.Random
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private enum class CardSection(val label: String) {
    PROBLEM("Concept"),
    INTUITION("Intuition"),
    APPROACH("Approach"),
    CODE("Code"),
    EDGE("Pitfalls"),
    ASK("Ask")
}

private const val NOTE_SAVE_DELAY_MS = 600L

/**
 * Tabs shown for [item], driven by its content: Concept always; Intuition / Approach / Code
 * when they have something to show; Edge cases when there are edge cases, or the problem is
 * solved / has a note (so "My notes" is reachable), or the user asked for the notes. Ask comes
 * last, on full problems, and only while the AI toggle is on.
 */
private fun cardSections(
    item: FeedItem,
    notesRequested: Boolean,
    aiEnabled: Boolean
): List<CardSection> {
    val p = item.problem
    return buildList {
        add(CardSection.PROBLEM)
        if (!p.bruteForce.isNullOrBlank() || !p.intuition.isNullOrBlank() ||
            !p.whyItWorks.isNullOrBlank()
        ) {
            add(CardSection.INTUITION)
        }
        if (p.approach.isNotEmpty() || item.dryRun.isNotEmpty()) add(CardSection.APPROACH)
        if (!p.solution.isNullOrBlank() || item.hasMySolution) add(CardSection.CODE)
        if (p.edgeCases.isNotEmpty() || item.solved || item.hasNote || notesRequested) {
            add(CardSection.EDGE)
        }
        if (aiEnabled && item.isFull) add(CardSection.ASK)
    }
}

@Composable
fun ConceptCard(
    item: FeedItem,
    titles: Map<String, String>,
    /** Ask "Which pattern fits?" before showing topics and patterns (Quiz / Revise modes). */
    quizEnabled: Boolean,
    /** Distinct patterns of all full problems: the distractor pool. */
    patternPool: List<String>,
    practiceMode: Boolean,
    onPracticeModeChange: (Boolean) -> Unit,
    /** Called once per card visit with whether the pattern quiz was answered correctly. */
    onQuizAnswered: (Boolean) -> Unit,
    onToggleBookmark: () -> Unit,
    onToggleSolved: () -> Unit,
    onRate: (Rating) -> Unit,
    onRelatedClick: (String) -> Unit,
    /** True while this card is the pager's current page (starts the think timer). */
    isCurrent: Boolean,
    /** Think timer on the Concept tab, in minutes; 0 = off. */
    thinkTimerMinutes: Int,
    /** Pattern string (lowercase) -> pattern of the library. */
    patternIndex: Map<String, PatternRef>,
    onOpenPattern: (String) -> Unit,
    onSaveNote: (String) -> Unit,
    onSaveMySolution: (String) -> Unit,
    /** Own problems only. */
    onEditConcept: () -> Unit,
    onDeleteConcept: () -> Unit,
    /** AI master toggle: adds the Ask tab to full problems. */
    aiEnabled: Boolean = false,
    /** Ask tab without a key: opens Settings. */
    onOpenSettings: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    val colors = QuickMLTheme.colors
    val slug = item.problem.slug
    var notesRequested by remember(slug) { mutableStateOf(false) }
    val sections = remember(item, notesRequested, aiEnabled) {
        cardSections(item, notesRequested, aiEnabled)
    }
    val pagerState = rememberPagerState { sections.size }
    val scope = rememberCoroutineScope()
    val currentToggle by rememberUpdatedState(onToggleBookmark)
    val currentOnQuizAnswered by rememberUpdatedState(onQuizAnswered)
    val currentOnSaveNote by rememberUpdatedState(onSaveNote)

    // Pattern quiz state lives here (not in the Concept tab) so it survives tab swipes.
    val patterns = item.problem.patterns
    val quizOptions = remember(slug, patterns, patternPool) {
        patternQuizOptions(slug, patterns, patternPool)
    }
    val showQuiz = quizEnabled && item.isFull && quizOptions.size >= 2
    var quizPick by remember(slug) { mutableStateOf<String?>(null) }

    // "My notes": edited locally, saved debounced and when the card leaves composition.
    var noteText by remember(slug) { mutableStateOf(item.note.orEmpty()) }
    var savedNote by remember(slug) { mutableStateOf(item.note.orEmpty()) }
    LaunchedEffect(slug, noteText) {
        if (noteText != savedNote) {
            delay(NOTE_SAVE_DELAY_MS)
            currentOnSaveNote(noteText)
            savedNote = noteText
        }
    }
    DisposableEffect(slug) {
        onDispose {
            if (noteText != savedNote) {
                currentOnSaveNote(noteText)
                savedNote = noteText
            }
        }
    }

    // Code tab: Reference / Mine, and the full-screen editor of the user's own solution.
    var showMine by remember(slug) { mutableStateOf(false) }
    var editingSolution by remember(slug) { mutableStateOf(false) }

    // Think timer: restarts each time the card becomes the current page.
    val timerTotal = thinkTimerMinutes.coerceAtLeast(0) * 60
    var timerLeft by remember(slug) { mutableIntStateOf(timerTotal) }
    LaunchedEffect(slug, isCurrent, timerTotal) {
        timerLeft = timerTotal
        if (isCurrent && timerTotal > 0) {
            while (timerLeft > 0) {
                delay(1_000L)
                timerLeft -= 1
            }
        }
    }

    // "My notes" from the overflow menu / note icon: show the Edge tab and go there.
    var notesTick by remember(slug) { mutableIntStateOf(0) }
    LaunchedEffect(notesTick) {
        if (notesTick > 0) {
            val index = sections.indexOf(CardSection.EDGE)
            if (index >= 0) pagerState.animateScrollToPage(index)
        }
    }
    fun openNotes() {
        notesRequested = true
        notesTick++
    }

    // Changing tabs stops reading this card aloud.
    val speaker = LocalSpeaker.current
    LaunchedEffect(pagerState, speaker, slug) {
        snapshotFlow { pagerState.currentPage }.collect {
            speaker?.stopIfSpeaking("$slug/")
        }
    }

    var heartTick by remember { mutableIntStateOf(0) }
    val heart = remember { Animatable(0f) }
    LaunchedEffect(heartTick) {
        if (heartTick > 0) {
            heart.snapTo(0f)
            heart.animateTo(1f, tween(180))
            delay(250)
            heart.animateTo(0f, tween(220))
        }
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .padding(horizontal = 12.dp, vertical = 8.dp)
            .clip(RoundedCornerShape(24.dp))
            .background(colors.card)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(Unit) {
                    detectTapGestures(
                        onDoubleTap = {
                            currentToggle()
                            heartTick++
                        }
                    )
                }
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (sections.size > 1) {
                    ScrollableTabRow(
                        selectedTabIndex = pagerState.currentPage.coerceIn(0, sections.size - 1),
                        modifier = Modifier.weight(1f),
                        containerColor = Color.Transparent,
                        contentColor = colors.accent,
                        edgePadding = 8.dp,
                        divider = {}
                    ) {
                        sections.forEachIndexed { index, section ->
                            Tab(
                                selected = pagerState.currentPage == index,
                                onClick = {
                                    scope.launch { pagerState.animateScrollToPage(index) }
                                },
                                text = { Text(section.label, maxLines = 1) },
                                selectedContentColor = colors.accent,
                                unselectedContentColor = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                } else {
                    Spacer(Modifier.weight(1f))
                }
                if (item.hasNote) {
                    Icon(
                        imageVector = Icons.Filled.Edit,
                        contentDescription = "Has a note",
                        tint = colors.accent,
                        modifier = Modifier
                            .clip(CircleShape)
                            .clickable { openNotes() }
                            .padding(4.dp)
                            .size(16.dp)
                    )
                }
                IconButton(onClick = onToggleSolved) {
                    Icon(
                        imageVector = if (item.solved) Icons.Filled.CheckCircle else Icons.Outlined.CheckCircle,
                        contentDescription = if (item.solved) "Mark not learned" else "Mark learned",
                        tint = if (item.solved) colors.accent else MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                IconButton(onClick = onToggleBookmark) {
                    Icon(
                        imageVector = if (item.bookmarked) Icons.Filled.Bookmark else Icons.Filled.BookmarkBorder,
                        contentDescription = if (item.bookmarked) "Remove bookmark" else "Bookmark",
                        tint = if (item.bookmarked) colors.accent else MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                CardMenu(
                    hasNote = item.hasNote,
                    hasMySolution = item.hasMySolution,
                    userAdded = item.userAdded,
                    onNotes = { openNotes() },
                    onMySolution = { editingSolution = true },
                    onEditConcept = onEditConcept,
                    onDeleteConcept = onDeleteConcept
                )
            }

            HorizontalPager(
                state = pagerState,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            ) { index ->
                val problem = item.problem
                // getOrNull: the tab list can shrink (e.g. unsolve) while a page is composed.
                when (sections.getOrNull(index)) {
                    null -> Unit
                    CardSection.PROBLEM -> ConceptSection(
                        item = item,
                        onToggleSolved = onToggleSolved,
                        quiz = if (showQuiz) {
                            QuizUi(
                                options = quizOptions,
                                answer = patterns.first(),
                                pick = quizPick,
                                onPick = { option ->
                                    if (quizPick == null) {
                                        quizPick = option
                                        currentOnQuizAnswered(option == patterns.first())
                                    }
                                },
                                review = lookupPattern(patternIndex, patterns.first()),
                                onReview = onOpenPattern
                            )
                        } else {
                            null
                        },
                        timerSeconds = if (timerTotal > 0) timerLeft else null,
                        patternIndex = patternIndex,
                        onOpenPattern = onOpenPattern
                    )
                    CardSection.INTUITION -> IntuitionSection(
                        slug = problem.slug,
                        bruteForce = problem.bruteForce,
                        insight = problem.intuition.orEmpty(),
                        whyItWorks = problem.whyItWorks
                    )
                    CardSection.APPROACH -> ApproachSection(
                        slug = problem.slug,
                        steps = problem.approach,
                        diagram = item.diagram,
                        dryRun = item.dryRun,
                        time = problem.time,
                        space = problem.space
                    )
                    CardSection.CODE -> CodeSection(
                        reference = problem.solution,
                        mine = item.mySolution,
                        showMine = showMine,
                        onShowMineChange = { showMine = it },
                        onEditMine = { editingSolution = true },
                        keyLines = item.keyLines,
                        practiceMode = practiceMode,
                        onPracticeModeChange = onPracticeModeChange
                    )
                    CardSection.EDGE -> EdgeSection(
                        note = noteText,
                        onNoteChange = { noteText = it },
                        edgeCases = problem.edgeCases,
                        related = problem.related,
                        titles = titles,
                        onRelatedClick = onRelatedClick
                    )
                    CardSection.ASK -> AskSection(item = item, onOpenSettings = onOpenSettings)
                }
            }

            // Rating / "mark solved" prompt: Code / Edge cases tabs, the last tab, or the only
            // tab of a notes-less card (whose Concept tab already has "Mark solved").
            val currentIndex = pagerState.currentPage.coerceIn(0, sections.size - 1)
            val currentSection = sections[currentIndex]
            // The Ask tab (chat) never shows the row and does not count as the last tab.
            val lastContentIndex = sections.indexOfLast { it != CardSection.ASK }
            val onLastTabs = currentSection != CardSection.ASK && (
                sections.size == 1 ||
                    currentSection == CardSection.CODE ||
                    currentSection == CardSection.EDGE ||
                    currentIndex == lastContentIndex
                )
            if (onLastTabs) {
                if (item.solved) {
                    val today = remember { LocalDate.now() }
                    val preview = remember(item.review) {
                        Sm2Scheduler.preview(item.reviewState, today)
                    }
                    RatingRow(preview = preview, onRate = onRate)
                } else if (currentSection != CardSection.PROBLEM) {
                    MarkSolvedPrompt(onToggleSolved)
                }
            }
        }

        if (heart.value > 0f) {
            Icon(
                imageVector = Icons.Filled.Favorite,
                contentDescription = null,
                tint = colors.heart,
                modifier = Modifier
                    .align(Alignment.Center)
                    .size(110.dp)
                    .graphicsLayer {
                        val s = 0.5f + heart.value * 0.7f
                        scaleX = s
                        scaleY = s
                        alpha = heart.value
                    }
            )
        }
    }

    if (editingSolution) {
        SolutionEditorDialog(
            initial = item.mySolution.orEmpty(),
            onSave = { code ->
                editingSolution = false
                onSaveMySolution(code)
                showMine = code.isNotBlank()
            },
            onDismiss = { editingSolution = false }
        )
    }
}

/** Overflow menu of a card: notes, own solution, and edit / delete for own problems. */
@Composable
private fun CardMenu(
    hasNote: Boolean,
    hasMySolution: Boolean,
    userAdded: Boolean,
    onNotes: () -> Unit,
    onMySolution: () -> Unit,
    onEditConcept: () -> Unit,
    onDeleteConcept: () -> Unit
) {
    var open by remember { mutableStateOf(false) }
    Box {
        IconButton(onClick = { open = true }) {
            Icon(
                imageVector = Icons.Filled.MoreVert,
                contentDescription = "More",
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            DropdownMenuItem(
                text = { Text(if (hasNote) "My notes" else "Add a note") },
                onClick = {
                    open = false
                    onNotes()
                }
            )
            DropdownMenuItem(
                text = { Text(if (hasMySolution) "Edit my implementation" else "Add my implementation") },
                onClick = {
                    open = false
                    onMySolution()
                }
            )
            if (userAdded) {
                DropdownMenuItem(
                    text = { Text("Edit card") },
                    onClick = {
                        open = false
                        onEditConcept()
                    }
                )
                DropdownMenuItem(
                    text = { Text("Delete card", color = QuickMLTheme.colors.danger) },
                    onClick = {
                        open = false
                        onDeleteConcept()
                    }
                )
            }
        }
    }
}

private fun lookupPattern(index: Map<String, PatternRef>, pattern: String): PatternRef? =
    index[pattern.trim().lowercase()]

@Composable
internal fun RatingRow(preview: Map<Rating, Int>, onRate: (Rating) -> Unit) {
    val colors = QuickMLTheme.colors
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 12.dp, end = 12.dp, top = 4.dp, bottom = 10.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Text(
            text = "How well did you remember?",
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Rating.values().forEach { rating ->
                FilledTonalButton(
                    onClick = { onRate(rating) },
                    modifier = Modifier.weight(1f),
                    contentPadding = PaddingValues(horizontal = 4.dp, vertical = 8.dp)
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = rating.label,
                            color = colors.onTint(colors.rating(rating)),
                            style = MaterialTheme.typography.labelLarge,
                            fontWeight = FontWeight.SemiBold,
                            maxLines = 1
                        )
                        Text(
                            text = formatInterval(preview[rating] ?: 1),
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            style = MaterialTheme.typography.labelSmall,
                            maxLines = 1
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun MarkSolvedPrompt(onToggleSolved: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 16.dp, end = 12.dp, top = 4.dp, bottom = 10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = "Mark learned to start revising it",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f)
        )
        Spacer(Modifier.width(8.dp))
        OutlinedButton(onClick = onToggleSolved) {
            Icon(
                imageVector = Icons.Outlined.CheckCircle,
                contentDescription = null,
                modifier = Modifier.size(18.dp)
            )
            Spacer(Modifier.width(8.dp))
            Text("Mark learned")
        }
    }
}

@Composable
internal fun Pill(
    text: String,
    color: Color,
    modifier: Modifier = Modifier,
    onClick: (() -> Unit)? = null
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(50))
            .background(color.copy(alpha = 0.16f))
            .then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier)
            .padding(horizontal = 12.dp, vertical = 5.dp)
    ) {
        Text(
            text = text,
            color = QuickMLTheme.colors.onTint(color),
            style = MaterialTheme.typography.labelMedium
        )
    }
}

private fun platformLabel(platform: String): String = when (platform) {
    "leetcode" -> "LeetCode"
    "gfg" -> "GFG"
    "codingninjas" -> "Coding Ninjas"
    else -> "Web"
}

private fun openUrl(context: Context, url: String) {
    try {
        val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url)).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        context.startActivity(intent)
    } catch (e: ActivityNotFoundException) {
        Toast.makeText(context, "No app found to open this link", Toast.LENGTH_SHORT).show()
    }
}

/** Pattern quiz shown on the Concept tab; [pick] is null until answered. */
private class QuizUi(
    val options: List<String>,
    val answer: String,
    val pick: String?,
    val onPick: (String) -> Unit,
    /** The answer's pattern card ("Review pattern"), when the library has it. */
    val review: PatternRef?,
    val onReview: (String) -> Unit
)

/**
 * The problem's first pattern plus up to 3 distractors from [pool] (never another pattern of the
 * same problem), shuffled. Deterministic per slug. Empty when the problem has no pattern.
 */
private fun patternQuizOptions(
    slug: String,
    patterns: List<String>,
    pool: List<String>
): List<String> {
    val answer = patterns.firstOrNull() ?: return emptyList()
    val random = Random(slug.hashCode())
    val distractors = pool
        .filter { it !in patterns }
        .distinct()
        .shuffled(random)
        .take(3)
    return (distractors + answer).shuffled(random)
}

@Composable
private fun PatternQuizCard(quiz: QuizUi) {
    val colors = QuickMLTheme.colors
    val pick = quiz.pick
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = "Which family fits?",
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.SemiBold
        )
        quiz.options.forEach { option ->
            val highlight: Color? = when {
                pick == null -> null
                option == quiz.answer -> colors.success
                option == pick -> colors.danger
                else -> null
            }
            val shape = RoundedCornerShape(12.dp)
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(shape)
                    .background(highlight?.copy(alpha = 0.16f) ?: colors.card)
                    .border(1.dp, highlight ?: MaterialTheme.colorScheme.outlineVariant, shape)
                    .clickable(enabled = pick == null) { quiz.onPick(option) }
                    .padding(horizontal = 14.dp, vertical = 12.dp)
            ) {
                Text(
                    text = option,
                    style = MaterialTheme.typography.bodyLarge,
                    color = if (highlight != null) {
                        colors.onTint(highlight)
                    } else {
                        MaterialTheme.colorScheme.onSurface
                    }
                )
            }
        }
        if (pick != null) {
            val correct = pick == quiz.answer
            Text(
                text = if (correct) "Correct!" else "It's " + quiz.answer,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                color = colors.onTint(if (correct) colors.success else colors.danger)
            )
            val review = quiz.review
            if (review != null) {
                TextButton(
                    onClick = { quiz.onReview(review.id) },
                    contentPadding = PaddingValues(horizontal = 0.dp, vertical = 4.dp)
                ) {
                    Text("Review family: " + review.name, color = colors.accent)
                }
            }
        }
    }
}

/** "Think · 1:23" while running; accent "Time's up" at zero. Never blocks anything. */
@Composable
private fun ThinkTimerChip(secondsLeft: Int) {
    val done = secondsLeft <= 0
    val accent = QuickMLTheme.colors.accent
    val color = if (done) accent else MaterialTheme.colorScheme.onSurfaceVariant
    val textColor = if (done) QuickMLTheme.colors.onTint(accent) else color
    val text = if (done) {
        "Time's up — check the intuition"
    } else {
        "Think · " + (secondsLeft / 60) + ":" + (secondsLeft % 60).toString().padStart(2, '0')
    }
    Text(
        text = text,
        color = textColor,
        style = MaterialTheme.typography.labelLarge,
        fontWeight = if (done) FontWeight.SemiBold else FontWeight.Normal,
        modifier = Modifier
            .clip(RoundedCornerShape(50))
            .background(color.copy(alpha = if (done) 0.16f else 0.10f))
            .padding(horizontal = 12.dp, vertical = 5.dp)
    )
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ConceptSection(
    item: FeedItem,
    onToggleSolved: () -> Unit,
    quiz: QuizUi?,
    /** Think timer seconds left; null when the timer is off. */
    timerSeconds: Int?,
    patternIndex: Map<String, PatternRef>,
    onOpenPattern: (String) -> Unit
) {
    val colors = QuickMLTheme.colors
    val problem = item.problem
    val context = LocalContext.current
    val summary = problem.summary
    val description = problem.description
    val examples = item.examples
    val constraints = item.constraints
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Text(
            text = problem.title,
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold
        )
        if (timerSeconds != null) ThinkTimerChip(timerSeconds)
        if (quiz != null) PatternQuizCard(quiz)
        // Topics, patterns and the section name would give the quiz away.
        val revealed = quiz == null || quiz.pick != null
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Pill(problem.difficulty, colors.difficulty(problem.difficulty))
            if (revealed) {
                problem.topics.forEach { Pill(it, colors.accent) }
                problem.patterns.forEach { pattern ->
                    val ref = lookupPattern(patternIndex, pattern)
                    Pill(
                        text = pattern,
                        color = MaterialTheme.colorScheme.tertiary,
                        onClick = ref?.let { r -> { onOpenPattern(r.id) } }
                    )
                }
            }
            if (item.solved) Pill("Learned", colors.success)
        }
        if (revealed && item.section.isNotEmpty()) {
            Text(
                text = item.section + "  #" + item.order,
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        if (description != null) {
            if (summary != null) {
                MathText(
                    text = summary,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            SectionHeading("When to use")
            MathText(
                text = description.trim(),
                style = MaterialTheme.typography.bodyLarge,
                lineHeight = 26.sp
            )
        } else if (summary != null) {
            if (item.isFull) SectionHeading("Concept")
            MathText(
                text = summary,
                style = MaterialTheme.typography.bodyLarge,
                lineHeight = 26.sp
            )
        } else if (!item.userAdded) {
            Text(
                text = "Reference only: open the link below",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        if (examples.isNotEmpty()) {
            SectionHeading("Interview questions")
            examples.forEachIndexed { index, example ->
                ExampleCard(number = index + 1, example = example)
            }
        }
        if (constraints.isNotEmpty()) {
            SectionHeading("Hyperparameters")
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                constraints.forEach { constraint ->
                    Row {
                        Text(
                            text = "•",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Spacer(Modifier.width(8.dp))
                        MathText(
                            text = constraint,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
            }
        }
        if (item.solved) {
            FilledTonalButton(
                onClick = onToggleSolved,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(
                    imageVector = Icons.Filled.CheckCircle,
                    contentDescription = null,
                    tint = colors.accent,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(Modifier.width(8.dp))
                Text("Learned (tap to undo)")
            }
        } else {
            OutlinedButton(
                onClick = onToggleSolved,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(
                    imageVector = Icons.Outlined.CheckCircle,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(Modifier.width(8.dp))
                Text("Mark learned")
            }
        }
        if (problem.url.isNotBlank()) {
            Button(
                onClick = { openUrl(context, problem.url) },
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.OpenInNew,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(Modifier.width(8.dp))
                Text("Open reference")
            }
        }
    }
}

@Composable
internal fun SectionHeading(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.titleMedium,
        fontWeight = FontWeight.SemiBold
    )
}

@Composable
private fun ExampleCard(number: Int, example: ExampleJson) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = "Question " + number,
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.SemiBold
        )
        MonoLine(label = "Q", value = example.input)
        MonoLine(label = "A", value = example.output)
        if (example.explanation.isNotBlank()) {
            MathText(
                text = example.explanation.trim(),
                style = MaterialTheme.typography.bodyMedium,
                lineHeight = 22.sp
            )
        }
    }
}

@Composable
private fun MonoLine(label: String, value: String) {
    val code = QuickMLTheme.colors.code
    Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        val hasMath = value.contains('$')
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(8.dp))
                .background(code.background)
                .then(if (hasMath) Modifier else Modifier.horizontalScroll(rememberScrollState()))
                .padding(horizontal = 10.dp, vertical = 8.dp)
        ) {
            if (hasMath) {
                // Formulas need real typesetting and normal wrapping, not a one-line mono box.
                MathText(
                    text = value,
                    style = MaterialTheme.typography.bodyMedium,
                    color = code.text
                )
            } else {
                Text(
                    text = value,
                    style = TextStyle(
                        fontFamily = FontFamily.Monospace,
                        fontSize = 13.sp,
                        lineHeight = 18.sp
                    ),
                    color = code.text,
                    softWrap = false
                )
            }
        }
    }
}

/** Listen / Stop button reading [text] aloud; hidden when no Speaker is provided. */
@Composable
private fun ListenRow(utteranceId: String, text: () -> String) {
    val speaker = LocalSpeaker.current ?: return
    val speakingId by speaker.speakingId.collectAsState()
    val speaking = speakingId == utteranceId
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Spacer(Modifier.weight(1f))
        IconButton(
            onClick = {
                if (speaking) speaker.stop() else speaker.speak(utteranceId, text())
            }
        ) {
            Icon(
                imageVector = if (speaking) Icons.Filled.Stop else Icons.AutoMirrored.Filled.VolumeUp,
                contentDescription = if (speaking) "Stop reading" else "Listen",
                tint = if (speaking) {
                    QuickMLTheme.colors.accent
                } else {
                    MaterialTheme.colorScheme.onSurfaceVariant
                }
            )
        }
    }
}

/**
 * Intuition tab: brute force (why it's slow) -> key insight (highlighted) -> why it works.
 * Older content without the extra fields falls back to plain paragraphs.
 */
@Composable
private fun IntuitionSection(
    slug: String,
    bruteForce: String?,
    insight: String,
    whyItWorks: String?
) {
    val colors = QuickMLTheme.colors
    val plain = bruteForce.isNullOrBlank() && whyItWorks.isNullOrBlank()
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(if (plain) 12.dp else 16.dp)
    ) {
        ListenRow(utteranceId = "$slug/intuition") {
            intuitionSpeech(bruteForce, insight, whyItWorks)
        }
        if (plain) {
            insight.split("\n\n").filter { it.isNotBlank() }.forEach { paragraph ->
                MathText(
                    text = paragraph.trim(),
                    style = MaterialTheme.typography.bodyLarge,
                    lineHeight = 26.sp
                )
            }
        } else {
            if (!bruteForce.isNullOrBlank()) {
                ExplanationBlock(
                    title = "When not to use",
                    body = bruteForce,
                    titleColor = MaterialTheme.colorScheme.onSurfaceVariant,
                    background = MaterialTheme.colorScheme.surfaceVariant
                )
            }
            if (insight.isNotBlank()) {
                ExplanationBlock(
                    title = "Key insight",
                    body = insight,
                    titleColor = colors.onTint(colors.accent),
                    background = colors.accent.copy(alpha = 0.12f)
                )
            }
            if (!whyItWorks.isNullOrBlank()) {
                ExplanationBlock(
                    title = "Why it works",
                    body = whyItWorks,
                    titleColor = MaterialTheme.colorScheme.onSurfaceVariant,
                    background = Color.Transparent
                )
            }
        }
    }
}

@Composable
private fun ExplanationBlock(title: String, body: String, titleColor: Color, background: Color) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(background)
            .then(
                // A transparent block still reads as a card: a thin outline, same padding.
                if (background == Color.Transparent) {
                    Modifier.border(
                        1.dp,
                        MaterialTheme.colorScheme.outlineVariant,
                        RoundedCornerShape(14.dp)
                    )
                } else {
                    Modifier
                }
            )
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.SemiBold,
            color = titleColor
        )
        body.split("\n\n").filter { it.isNotBlank() }.forEach { paragraph ->
            MathText(
                text = paragraph.trim(),
                style = MaterialTheme.typography.bodyLarge,
                lineHeight = 26.sp
            )
        }
    }
}

/** Loading state of a PNG shipped in assets. */
private sealed interface AssetPic {
    data object Loading : AssetPic
    data object Missing : AssetPic
    class Ready(val bitmap: ImageBitmap) : AssetPic
}

@Composable
private fun rememberAssetPic(dir: String, name: String): AssetPic {
    val context = LocalContext.current
    val theme = if (QuickMLTheme.colors.isDark) "dark" else "light"
    val pic by produceState<AssetPic>(AssetPic.Loading, dir, name, theme) {
        value = withContext(Dispatchers.IO) {
            runCatching {
                context.assets.open("$dir/${name}_$theme.png").use {
                    BitmapFactory.decodeStream(it)?.asImageBitmap()
                }
            }.getOrNull()?.let { AssetPic.Ready(it) } ?: AssetPic.Missing
        }
    }
    return pic
}

/** A pre-rendered picture on a code-coloured card; tap to enlarge. */
@Composable
private fun PictureBox(title: String, bitmap: ImageBitmap) {
    val colors = QuickMLTheme.colors
    var enlarged by remember { mutableStateOf(false) }
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        SectionHeading(title)
        Image(
            bitmap = bitmap,
            contentDescription = title,
            contentScale = ContentScale.FillWidth,
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(12.dp))
                .background(colors.code.background)
                .clickable { enlarged = true }
                .padding(8.dp)
        )
    }
    if (enlarged) {
        Dialog(
            onDismissRequest = { enlarged = false },
            properties = DialogProperties(usePlatformDefaultWidth = false)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(colors.code.background)
                    .clickable { enlarged = false },
                contentAlignment = Alignment.Center
            ) {
                Image(
                    bitmap = bitmap,
                    contentDescription = title,
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxSize().padding(12.dp)
                )
            }
        }
    }
}

/** Pre-drawn diagram (assets/diagrams, light or dark variant). */
@Composable
private fun DiagramBox(lines: List<String>) {
    val name = lines.firstOrNull() ?: return
    val pic = rememberAssetPic("diagrams", name)
    if (pic is AssetPic.Ready) PictureBox("Diagram", pic.bitmap)
}

/** Formulas, typeset live; a line that cannot be typeset shows as its LaTeX source. */
@Composable
private fun DryRunBox(lines: List<String>) {
    val code = QuickMLTheme.colors.code
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        SectionHeading("Math")
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(12.dp))
                .background(code.background)
                .padding(horizontal = 14.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            lines.forEach { line ->
                MathText(
                    text = line,
                    style = MaterialTheme.typography.bodyLarge,
                    color = code.text
                )
            }
        }
    }
}

@Composable
private fun ApproachSection(
    slug: String,
    steps: List<String>,
    diagram: List<String>,
    dryRun: List<String>,
    time: String?,
    space: String?
) {
    val colors = QuickMLTheme.colors
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        ListenRow(utteranceId = "$slug/approach") {
            approachSpeech(steps, time, space)
        }
        steps.forEachIndexed { index, step ->
            Row(verticalAlignment = Alignment.Top) {
                Box(
                    modifier = Modifier
                        .size(26.dp)
                        .clip(CircleShape)
                        .background(colors.accent.copy(alpha = 0.18f)),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = (index + 1).toString(),
                        color = colors.onTint(colors.accent),
                        style = MaterialTheme.typography.labelLarge
                    )
                }
                Spacer(Modifier.width(12.dp))
                MathText(
                    text = step,
                    style = MaterialTheme.typography.bodyLarge,
                    lineHeight = 24.sp,
                    modifier = Modifier.weight(1f)
                )
            }
        }
        if (diagram.isNotEmpty()) {
            DiagramBox(diagram)
        }
        if (dryRun.isNotEmpty()) {
            DryRunBox(dryRun)
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            ComplexityBox("Compute", time ?: "-", Modifier.weight(1f))
            ComplexityBox("Memory", space ?: "-", Modifier.weight(1f))
        }
    }
}

@Composable
private fun ComplexityBox(label: String, value: String, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(14.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .padding(horizontal = 14.dp, vertical = 10.dp)
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.titleMedium,
            fontFamily = FontFamily.Monospace,
            color = QuickMLTheme.colors.accent
        )
    }
}

/**
 * Code tab: the reference solution and/or the user's own ("Mine"), with a Reference · Mine
 * toggle when both exist. Practice mode masks key lines of the reference only.
 */
@Composable
private fun CodeSection(
    reference: String?,
    mine: String?,
    showMine: Boolean,
    onShowMineChange: (Boolean) -> Unit,
    onEditMine: () -> Unit,
    keyLines: List<Int>,
    practiceMode: Boolean,
    onPracticeModeChange: (Boolean) -> Unit
) {
    val hasReference = !reference.isNullOrBlank()
    val hasMine = !mine.isNullOrBlank()
    val viewingMine = hasMine && (showMine || !hasReference)
    val code = if (viewingMine) mine.orEmpty() else reference.orEmpty()
    val activeKeyLines = if (viewingMine) emptyList() else keyLines
    val context = LocalContext.current
    val clipboard = LocalClipboardManager.current
    val palette = QuickMLTheme.colors.code
    val trimmed = remember(code) { code.trimEnd('\n') }
    val highlighted = remember(trimmed, palette) { PythonHighlighter.highlight(trimmed, palette) }
    val lineNumbers = remember(trimmed) {
        (1..trimmed.lines().size).joinToString("\n")
    }
    val codeLines = remember(trimmed) { trimmed.split('\n') }
    // 0-based indices of the key lines that can be masked (in range and not blank).
    val maskable = remember(codeLines, activeKeyLines) {
        activeKeyLines
            .map { it - 1 }
            .filter { it in codeLines.indices && codeLines[it].isNotBlank() }
            .toSet()
    }
    val practice = practiceMode && maskable.isNotEmpty()
    var revealed by remember(trimmed, practice) { mutableStateOf(emptySet<Int>()) }
    val codeStyle = remember {
        TextStyle(
            fontFamily = FontFamily.Monospace,
            fontSize = 13.sp,
            lineHeight = 20.sp
        )
    }
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 12.dp, vertical = 4.dp)
    ) {
        val bothViews = hasReference && hasMine
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (bothViews) {
                Row(
                    modifier = Modifier.weight(1f),
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    FilterChip(
                        selected = !viewingMine,
                        onClick = { onShowMineChange(false) },
                        label = { Text("Reference") }
                    )
                    FilterChip(
                        selected = viewingMine,
                        onClick = { onShowMineChange(true) },
                        label = { Text("Mine") }
                    )
                }
            } else {
                Text(
                    text = if (viewingMine) "My implementation" else "Python",
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier
                        .weight(1f)
                        .padding(start = 4.dp)
                )
            }
            if (maskable.isNotEmpty()) {
                FilterChip(
                    selected = practiceMode,
                    onClick = { onPracticeModeChange(!practiceMode) },
                    label = { Text("Practice") }
                )
                Spacer(Modifier.width(4.dp))
            }
            val copy = {
                // Always the full, real code (also in Practice mode).
                clipboard.setText(AnnotatedString(trimmed))
                Toast.makeText(context, "Code copied", Toast.LENGTH_SHORT).show()
            }
            if (bothViews) {
                IconButton(onClick = copy) {
                    Icon(
                        imageVector = Icons.Filled.ContentCopy,
                        contentDescription = "Copy code",
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            } else {
                FilledTonalButton(onClick = copy) {
                    Icon(
                        imageVector = Icons.Filled.ContentCopy,
                        contentDescription = null,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(Modifier.width(6.dp))
                    Text("Copy")
                }
            }
        }
        if (practice) {
            val hidden = maskable.count { it !in revealed }
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = when (hidden) {
                        0 -> "All key lines revealed"
                        1 -> "1 key line hidden"
                        else -> hidden.toString() + " key lines hidden"
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier
                        .weight(1f)
                        .padding(start = 4.dp)
                )
                TextButton(onClick = { revealed = if (hidden > 0) maskable else emptySet() }) {
                    Text(if (hidden > 0) "Reveal all" else "Hide again")
                }
            }
        } else {
            Spacer(Modifier.size(8.dp))
        }
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .clip(RoundedCornerShape(14.dp))
                .background(palette.background)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
            ) {
                if (practice) {
                    PracticeCode(
                        lines = codeLines,
                        highlighted = highlighted,
                        masked = maskable - revealed,
                        revealedKeys = maskable.intersect(revealed),
                        style = codeStyle,
                        onReveal = { index -> revealed = revealed + index }
                    )
                } else {
                    Row(
                        modifier = Modifier
                            .horizontalScroll(rememberScrollState())
                            .padding(12.dp)
                    ) {
                        Text(
                            text = lineNumbers,
                            style = codeStyle,
                            color = palette.lineNumber,
                            textAlign = TextAlign.End
                        )
                        Spacer(Modifier.width(14.dp))
                        Text(
                            text = highlighted,
                            style = codeStyle,
                            color = palette.text,
                            softWrap = false
                        )
                    }
                }
            }
        }
        TextButton(onClick = onEditMine) {
            Icon(
                imageVector = Icons.Filled.Edit,
                contentDescription = null,
                modifier = Modifier.size(16.dp)
            )
            Spacer(Modifier.width(6.dp))
            Text(if (hasMine) "Edit my implementation" else "Add my implementation")
        }
    }
}

/**
 * Code rendered line by line so key lines can be masked: a masked line keeps its indentation and
 * shows a "tap to reveal" block instead of the code. Revealed key lines get a faint highlight.
 */
@Composable
private fun PracticeCode(
    lines: List<String>,
    highlighted: AnnotatedString,
    masked: Set<Int>,
    revealedKeys: Set<Int>,
    style: TextStyle,
    onReveal: (Int) -> Unit
) {
    // Start offset of every line in [highlighted] (same text as the joined lines).
    val starts = remember(lines) {
        val result = IntArray(lines.size)
        var offset = 0
        lines.forEachIndexed { index, line ->
            result[index] = offset
            offset += line.length + 1
        }
        result
    }
    val digits = lines.size.toString().length
    val colors = QuickMLTheme.colors
    Column(
        modifier = Modifier
            .horizontalScroll(rememberScrollState())
            .padding(12.dp)
    ) {
        lines.forEachIndexed { index, line ->
            Row(
                modifier = if (index in revealedKeys) {
                    Modifier
                        .clip(RoundedCornerShape(4.dp))
                        .background(colors.accent.copy(alpha = 0.10f))
                } else {
                    Modifier
                },
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = (index + 1).toString().padStart(digits),
                    style = style,
                    color = colors.code.lineNumber,
                    softWrap = false
                )
                Spacer(Modifier.width(14.dp))
                if (index in masked) {
                    val indent = line.takeWhile { it == ' ' || it == '\t' }
                    if (indent.isNotEmpty()) {
                        Text(text = indent, style = style, softWrap = false)
                    }
                    Text(
                        text = "tap to reveal",
                        style = style,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        softWrap = false,
                        modifier = Modifier
                            .clip(RoundedCornerShape(6.dp))
                            .background(MaterialTheme.colorScheme.surfaceVariant)
                            .clickable { onReveal(index) }
                            .padding(horizontal = 12.dp)
                    )
                } else {
                    val start = starts[index]
                    val end = (start + line.length).coerceAtMost(highlighted.length)
                    Text(
                        text = if (line.isEmpty() || start >= end) {
                            AnnotatedString(" ")
                        } else {
                            highlighted.subSequence(start, end)
                        },
                        style = style,
                        color = colors.code.text,
                        softWrap = false
                    )
                }
            }
        }
    }
}

@Composable
private fun EdgeSection(
    note: String,
    onNoteChange: (String) -> Unit,
    edgeCases: List<String>,
    related: List<String>,
    titles: Map<String, String>,
    onRelatedClick: (String) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Text(
            text = "My notes",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold
        )
        OutlinedTextField(
            value = note,
            onValueChange = onNoteChange,
            placeholder = { Text("Your own notes: tricks, mistakes, what to remember") },
            minLines = 3,
            modifier = Modifier.fillMaxWidth()
        )
        Text(
            text = "Saved automatically",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        if (edgeCases.isNotEmpty()) {
            Spacer(Modifier.size(6.dp))
            Text(
                text = "Pitfalls",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )
            edgeCases.forEach { edge ->
                Row {
                    Text(
                        text = "•",
                        color = QuickMLTheme.colors.accent,
                        style = MaterialTheme.typography.bodyLarge
                    )
                    Spacer(Modifier.width(10.dp))
                    MathText(
                        text = edge,
                        style = MaterialTheme.typography.bodyLarge,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }
        if (related.isNotEmpty()) {
            Spacer(Modifier.size(6.dp))
            Text(
                text = "Related concepts",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )
            related.forEach { slug ->
                val title = titles[slug]
                if (title != null) {
                    Text(
                        text = title,
                        color = QuickMLTheme.colors.accent,
                        style = MaterialTheme.typography.bodyLarge,
                        modifier = Modifier
                            .clip(RoundedCornerShape(8.dp))
                            .clickable { onRelatedClick(slug) }
                            .padding(vertical = 6.dp, horizontal = 4.dp)
                    )
                } else {
                    Text(
                        text = slug.replace('-', ' '),
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        style = MaterialTheme.typography.bodyLarge,
                        modifier = Modifier.padding(vertical = 6.dp, horizontal = 4.dp)
                    )
                }
            }
        }
    }
}
