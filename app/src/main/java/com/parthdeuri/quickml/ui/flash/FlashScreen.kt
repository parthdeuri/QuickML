package com.parthdeuri.quickml.ui.flash

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.parthdeuri.quickml.data.repo.FlashCounts
import com.parthdeuri.quickml.data.settings.StudySettings
import com.parthdeuri.quickml.domain.srs.Rating
import com.parthdeuri.quickml.domain.srs.label
import com.parthdeuri.quickml.ui.common.MathText
import com.parthdeuri.quickml.ui.feed.RatingRow
import com.parthdeuri.quickml.ui.feed.SectionHeading
import com.parthdeuri.quickml.ui.theme.QuickMLTheme

/** The Flash tab: a deck picker, then a study session (tap to flip, rate, next). */
@Composable
fun FlashScreen(viewModel: FlashViewModel, onOpenConcept: (String) -> Unit) {
    val state by viewModel.state.collectAsState()
    val session = state.session
    BackHandler(enabled = session != null) { viewModel.exitSession() }
    if (session == null) {
        DeckPicker(state, viewModel)
    } else {
        StudyView(session, viewModel, onOpenConcept)
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun DeckPicker(state: FlashUiState, viewModel: FlashViewModel) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Text(
            text = "Flash cards",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = "Quick recall questions made from every card, scheduled with spaced repetition.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        CountsCard(state.counts)

        SectionHeading("Deck")
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            FilterChip(
                selected = state.deck == null,
                onClick = { viewModel.selectDeck(null) },
                label = { Text("All") }
            )
            state.decks.forEach { deck ->
                FilterChip(
                    selected = state.deck == deck.id,
                    onClick = { viewModel.selectDeck(deck.id) },
                    label = { Text(deck.name) }
                )
            }
        }

        SectionHeading("Card types")
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            FlashGroup.values().forEach { group ->
                FilterChip(
                    selected = group in state.groups,
                    onClick = { viewModel.toggleGroup(group) },
                    label = { Text(group.label) }
                )
            }
        }

        SectionHeading("New cards per day")
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            StudySettings.FLASH_NEW_CHOICES.forEach { n ->
                FilterChip(
                    selected = state.newLimit == n,
                    onClick = { viewModel.setNewLimit(n) },
                    label = { Text(n.toString()) }
                )
            }
        }

        val session = state.counts.session
        Button(
            onClick = { viewModel.start() },
            enabled = session > 0,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(if (session > 0) "Start (" + session + " cards)" else "Nothing due right now")
        }
        if (state.counts.total == 0) {
            Text(
                text = "No flash cards match this selection.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        } else if (session == 0) {
            Text(
                text = "All caught up. Come back tomorrow, or pick another deck or a higher daily limit.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun CountsCard(counts: FlashCounts) {
    val colors = QuickMLTheme.colors
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .padding(16.dp),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        CountItem("Due", counts.due.toString(), colors.accent)
        CountItem("New today", counts.newToday.toString(), colors.accent)
        CountItem("In selection", counts.total.toString(), MaterialTheme.colorScheme.onSurface)
    }
}

@Composable
private fun CountItem(label: String, value: String, valueColor: androidx.compose.ui.graphics.Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = value,
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = valueColor
        )
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

private fun kindLabel(kind: String): String = when (kind) {
    "define" -> "Definition"
    "idea" -> "Key insight"
    "qa" -> "Interview question"
    "use" -> "When to use"
    "pitfalls" -> "Pitfalls"
    "math" -> "Formulas"
    "compare" -> "Compare"
    "family" -> "Technique family"
    else -> "Card"
}

@Composable
private fun StudyView(
    session: FlashSession,
    viewModel: FlashViewModel,
    onOpenConcept: (String) -> Unit
) {
    val card = session.current
    if (card == null) {
        SummaryView(session, viewModel)
        return
    }
    Column(modifier = Modifier.fillMaxSize()) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 16.dp, end = 4.dp, top = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = (session.index + 1).toString() + " / " + session.queue.size,
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.weight(1f)
            )
            IconButton(onClick = { viewModel.exitSession() }) {
                Icon(Icons.Filled.Close, contentDescription = "End session")
            }
        }
        LinearProgressIndicator(
            progress = { session.index.toFloat() / session.queue.size },
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
        )
        Column(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(16.dp)
                .clip(RoundedCornerShape(20.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant)
                .clickable { viewModel.flip() }
                .verticalScroll(rememberScrollState())
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Text(
                text = kindLabel(card.kind).uppercase(),
                style = MaterialTheme.typography.labelMedium,
                color = QuickMLTheme.colors.accent,
                fontWeight = FontWeight.SemiBold
            )
            MathText(
                text = card.front,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
                lineHeight = 28.sp
            )
            if (session.flipped) {
                HorizontalDivider()
                MathText(
                    text = card.back,
                    style = MaterialTheme.typography.bodyLarge,
                    lineHeight = 26.sp
                )
            } else {
                Text(
                    text = "Tap the card to show the answer",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
        if (session.flipped) {
            RatingRow(preview = session.preview, onRate = { viewModel.rate(it) })
            TextButton(
                onClick = { onOpenConcept(card.slug) },
                modifier = Modifier.padding(start = 8.dp, bottom = 8.dp)
            ) {
                Text("Open the full card")
            }
        } else {
            Spacer(Modifier.height(24.dp))
        }
    }
}

@Composable
private fun SummaryView(session: FlashSession, viewModel: FlashViewModel) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Session complete",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = session.queue.size.toString() + " cards reviewed",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Rating.values().forEach { rating ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(text = rating.label, style = MaterialTheme.typography.bodyLarge)
                Text(
                    text = (session.tally[rating] ?: 0).toString(),
                    style = MaterialTheme.typography.bodyLarge,
                    fontWeight = FontWeight.SemiBold,
                    color = QuickMLTheme.colors.rating(rating)
                )
            }
        }
        Spacer(Modifier.height(8.dp))
        Button(onClick = { viewModel.exitSession() }, modifier = Modifier.fillMaxWidth()) {
            Text("Back to decks")
        }
        OutlinedButton(
            onClick = {
                viewModel.exitSession()
                viewModel.start()
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Study more")
        }
    }
}
