package com.parthdeuri.quickml.ui.dashboard

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.parthdeuri.quickml.ui.theme.QuickMLTheme
import kotlin.math.roundToInt

private const val EMPTY_HINT = "Scroll the feed to start building your history"

@Composable
fun DashboardScreen(
    viewModel: DashboardViewModel,
    onOpenSettings: () -> Unit,
    modifier: Modifier = Modifier
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    LaunchedEffect(Unit) { viewModel.refresh() }

    Column(modifier = modifier.fillMaxSize().statusBarsPadding()) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(start = 16.dp, end = 4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Dashboard",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.weight(1f).padding(vertical = 12.dp)
            )
            IconButton(onClick = onOpenSettings) {
                Icon(
                    imageVector = Icons.Filled.Settings,
                    contentDescription = "Settings",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
        if (state.loading) {
            Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator(color = QuickMLTheme.colors.accent)
            }
        } else {
            DashboardContent(state, Modifier.weight(1f))
        }
    }
}

@Composable
private fun DashboardContent(state: DashboardState, modifier: Modifier = Modifier) {
    val percent = if (state.total == 0) 0 else (state.solved * 100f / state.total).roundToInt()
    Column(
        modifier = modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // 1. Stat tiles
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            StatTile(
                value = state.currentStreak.toString(),
                label = "Day streak",
                subtext = "Best: " + state.bestStreak + (if (state.bestStreak == 1) " day" else " days"),
                modifier = Modifier.weight(1f)
            )
            StatTile(
                value = state.solved.toString() + " / " + state.total,
                label = "Learned",
                subtext = "$percent% of the sheet",
                modifier = Modifier.weight(1f)
            )
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            StatTile(
                value = state.dueToday.toString(),
                label = "Due today",
                subtext = "across all sheets",
                modifier = Modifier.weight(1f)
            )
            StatTile(
                value = if (state.recallPercent != null) state.recallPercent.toString() + "%" else "—",
                label = "Recall rate",
                subtext = state.reviewCount30.toString() + (if (state.reviewCount30 == 1) " review" else " reviews"),
                modifier = Modifier.weight(1f)
            )
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            StatTile(
                value = state.flashDue.toString(),
                label = "Flash cards due",
                subtext = "open the Flash tab",
                modifier = Modifier.weight(1f)
            )
            Spacer(Modifier.weight(1f))
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            StatTile(
                value = formatDuration(state.weekMillis),
                label = "Revised this week",
                subtext = null,
                modifier = Modifier.weight(1f)
            )
            StatTile(
                value = state.cardsToday.toString(),
                label = "Cards today",
                subtext = null,
                modifier = Modifier.weight(1f)
            )
        }

        // 1b. Pattern quiz (last 30 days)
        val quizPercent = if (state.quizTotal30 == 0) {
            null
        } else {
            (state.quizCorrect30 * 100f / state.quizTotal30).roundToInt()
        }
        StatTile(
            value = if (quizPercent == null) "—" else quizPercent.toString() + "%",
            label = "Family quiz",
            subtext = if (quizPercent == null) {
                "Try Quiz mode in the feed"
            } else {
                state.quizCorrect30.toString() + " / " + state.quizTotal30 + " correct · last 30 days"
            },
            modifier = Modifier.fillMaxWidth()
        )

        // 2. By difficulty
        DashCard(title = sheetTitle("By difficulty", state.sheetName)) {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                state.byDifficulty.forEach { row ->
                    DifficultyRow(row)
                }
            }
        }

        // 3. Sections
        if (state.sections.isNotEmpty()) {
            DashCard(title = sheetTitle("Sections", state.sheetName)) {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    state.sections.forEach { row ->
                        StackedProgressRow(
                            label = row.label,
                            trailing = row.solved.toString() + " / " + row.total,
                            fraction = row.fraction
                        )
                    }
                }
            }
        }

        // 4. Heatmap
        DashCard(title = "Activity") {
            if (state.hasActivity) {
                Heatmap(state.heatmap)
            } else {
                EmptyHint()
            }
        }

        // 5. Daily revision time
        DashCard(title = "Daily revision time (last 14 days)") {
            if (state.hasActivity) {
                DailyTimeChart(state.daily)
            } else {
                EmptyHint()
            }
        }

        // 5b. Reviews by rating
        DashCard(title = "Reviews (last 14 days)") {
            if (state.reviewDaily.any { it.total > 0 }) {
                ReviewsChart(state.reviewDaily)
            } else {
                Text(
                    text = "Rate cards in Revise mode to see your recall history",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 12.dp)
                )
            }
        }

        // 6. Focus next
        if (state.focus.isNotEmpty()) {
            DashCard(title = "Focus next") {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    state.focus.forEach { row ->
                        StackedProgressRow(
                            label = row.label + " · " + row.solved + "/" + row.total,
                            trailing = null,
                            fraction = row.fraction
                        )
                    }
                }
            }
        }

        // 7. Footer
        Text(
            text = "Full notes available for " + state.fullCount + " of " + state.total + " cards",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(bottom = 16.dp)
        )
    }
}

/** "By difficulty · NeetCode 150". */
private fun sheetTitle(title: String, sheetName: String): String =
    if (sheetName.isEmpty()) title else "$title · $sheetName"

@Composable
private fun DashCard(title: String, content: @Composable () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(QuickMLTheme.colors.card)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold
        )
        content()
    }
}

@Composable
private fun StatTile(
    value: String,
    label: String,
    subtext: String?,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(20.dp))
            .background(QuickMLTheme.colors.card)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp)
    ) {
        Text(
            text = value,
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
        Text(
            text = label,
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        // Keeps all tiles in a row the same height whether or not they have a subtext.
        Text(
            text = subtext ?: " ",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun Bar(fraction: Float, color: Color, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .height(6.dp)
            .clip(RoundedCornerShape(3.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
    ) {
        val f = fraction.coerceIn(0f, 1f)
        if (f > 0f) {
            Box(
                Modifier
                    .fillMaxHeight()
                    .fillMaxWidth(f)
                    .clip(RoundedCornerShape(3.dp))
                    .background(color)
            )
        }
    }
}

@Composable
private fun DifficultyRow(row: ProgressRow) {
    val colors = QuickMLTheme.colors
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(
            text = row.label,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.width(68.dp)
        )
        Bar(
            fraction = row.fraction,
            color = colors.difficulty(row.label, fallback = colors.accent),
            modifier = Modifier.weight(1f)
        )
        Text(
            text = row.solved.toString() + " / " + row.total,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.padding(start = 12.dp)
        )
    }
}

@Composable
private fun StackedProgressRow(label: String, trailing: String?, fraction: Float) {
    Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f)
            )
            if (trailing != null) {
                Text(
                    text = trailing,
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(start = 12.dp)
                )
            }
        }
        Bar(fraction = fraction, color = QuickMLTheme.colors.accent, modifier = Modifier.fillMaxWidth())
    }
}

@Composable
private fun EmptyHint() {
    Text(
        text = EMPTY_HINT,
        style = MaterialTheme.typography.bodyMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 12.dp)
    )
}

/** "2h 15m" / "40m" / "<1m". */
internal fun formatDuration(ms: Long): String {
    if (ms <= 0L) return "0m"
    val totalMinutes = ms / 60_000L
    if (totalMinutes == 0L) return "<1m"
    val hours = totalMinutes / 60
    val minutes = totalMinutes % 60
    return if (hours > 0) hours.toString() + "h " + minutes + "m" else minutes.toString() + "m"
}

/** "23 min" / "<1 min" / "0 min". */
internal fun formatMinutes(ms: Long): String {
    val minutes = (ms / 60_000f).roundToInt()
    return when {
        ms <= 0L -> "0 min"
        minutes == 0 -> "<1 min"
        else -> "$minutes min"
    }
}
