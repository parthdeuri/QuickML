package com.parthdeuri.quickml.ui.feed

import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
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
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.parthdeuri.quickml.ui.theme.QuickMLTheme
import kotlinx.coroutines.launch

private enum class PatternSection(val label: String) {
    OVERVIEW("Overview"),
    TEMPLATE("Template"),
    PITFALLS("Pitfalls"),
    PROBLEMS("Problems")
}

/** One card of Patterns mode: Overview · Template · Pitfalls · Problems. */
@Composable
fun PatternCard(
    card: PatternCardUi,
    onOpenProblem: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    val sections = PatternSection.values()
    val pagerState = rememberPagerState { sections.size }
    val scope = rememberCoroutineScope()
    val pattern = card.pattern
    val colors = QuickMLTheme.colors
    Box(
        modifier = modifier
            .fillMaxSize()
            .padding(horizontal = 12.dp, vertical = 8.dp)
            .clip(RoundedCornerShape(24.dp))
            .background(colors.card)
    ) {
        Column(modifier = Modifier.fillMaxSize()) {
            ScrollableTabRow(
                selectedTabIndex = pagerState.currentPage.coerceIn(0, sections.size - 1),
                containerColor = Color.Transparent,
                contentColor = colors.accent,
                edgePadding = 8.dp,
                divider = {}
            ) {
                sections.forEachIndexed { index, section ->
                    val label = if (section == PatternSection.PROBLEMS && card.problems.isNotEmpty()) {
                        section.label + " · " + card.problems.size
                    } else {
                        section.label
                    }
                    Tab(
                        selected = pagerState.currentPage == index,
                        onClick = { scope.launch { pagerState.animateScrollToPage(index) } },
                        text = { Text(label, maxLines = 1) },
                        selectedContentColor = colors.accent,
                        unselectedContentColor = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            HorizontalPager(
                state = pagerState,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            ) { index ->
                when (sections[index]) {
                    PatternSection.OVERVIEW -> OverviewTab(card)
                    PatternSection.TEMPLATE -> TemplateTab(pattern.template)
                    PatternSection.PITFALLS -> PitfallsTab(pattern.pitfalls)
                    PatternSection.PROBLEMS -> ProblemsTab(card.problems, onOpenProblem)
                }
            }
        }
    }
}

@Composable
private fun OverviewTab(card: PatternCardUi) {
    val pattern = card.pattern
    val colors = QuickMLTheme.colors
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Text(
            text = pattern.name,
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold
        )
        if (pattern.summary.isNotBlank()) {
            Text(
                text = pattern.summary,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                lineHeight = 24.sp
            )
        }
        if (pattern.complexity.isNotBlank()) {
            Pill(pattern.complexity, colors.accent)
        }
        if (pattern.whenToUse.isNotEmpty()) {
            SectionHeading("When to use")
            Bullets(pattern.whenToUse)
        }
        if (pattern.keyIdea.isNotBlank()) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(14.dp))
                    .background(colors.accent.copy(alpha = 0.12f))
                    .padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "Key idea",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.SemiBold,
                    color = colors.onTint(colors.accent)
                )
                pattern.keyIdea.split("\n\n").filter { it.isNotBlank() }.forEach { paragraph ->
                    Text(
                        text = paragraph.trim(),
                        style = MaterialTheme.typography.bodyLarge,
                        lineHeight = 26.sp
                    )
                }
            }
        }
    }
}

@Composable
private fun Bullets(lines: List<String>) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        lines.forEach { line ->
            Row {
                Text(
                    text = "•",
                    color = QuickMLTheme.colors.accent,
                    style = MaterialTheme.typography.bodyLarge
                )
                Spacer(Modifier.width(10.dp))
                Text(
                    text = line,
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}

@Composable
private fun TemplateTab(template: String) {
    val context = LocalContext.current
    val clipboard = LocalClipboardManager.current
    val palette = QuickMLTheme.colors.code
    val trimmed = remember(template) { template.trimEnd('\n') }
    val highlighted = remember(trimmed, palette) { PythonHighlighter.highlight(trimmed, palette) }
    val lineNumbers = remember(trimmed) { (1..trimmed.lines().size).joinToString("\n") }
    val codeStyle = remember {
        TextStyle(fontFamily = FontFamily.Monospace, fontSize = 13.sp, lineHeight = 20.sp)
    }
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 12.dp, vertical = 4.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Python template",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier
                    .weight(1f)
                    .padding(start = 4.dp)
            )
            if (trimmed.isNotBlank()) {
                FilledTonalButton(
                    onClick = {
                        clipboard.setText(AnnotatedString(trimmed))
                        Toast.makeText(context, "Template copied", Toast.LENGTH_SHORT).show()
                    }
                ) {
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
        Spacer(Modifier.size(8.dp))
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(bottom = 10.dp)
                .clip(RoundedCornerShape(14.dp))
                .background(palette.background)
        ) {
            if (trimmed.isBlank()) {
                Text(
                    text = "No template for this pattern.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(16.dp)
                )
            } else {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(rememberScrollState())
                ) {
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
    }
}

@Composable
private fun PitfallsTab(pitfalls: List<String>) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        SectionHeading("Common pitfalls")
        if (pitfalls.isEmpty()) {
            Text(
                text = "None listed.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        } else {
            Bullets(pitfalls)
        }
    }
}

@Composable
private fun ProblemsTab(problems: List<PatternProblemRow>, onOpenProblem: (String) -> Unit) {
    val colors = QuickMLTheme.colors
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 8.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp)
    ) {
        if (problems.isEmpty()) {
            Text(
                text = "No linked problems yet.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(8.dp)
            )
        }
        problems.forEach { row ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(10.dp))
                    .clickable { onOpenProblem(row.slug) }
                    .padding(horizontal = 8.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = if (row.solved) Icons.Filled.CheckCircle else Icons.Outlined.CheckCircle,
                    contentDescription = if (row.solved) "Solved" else "Not solved",
                    tint = if (row.solved) colors.accent else MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(Modifier.width(12.dp))
                Text(
                    text = row.title,
                    style = MaterialTheme.typography.bodyLarge,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f)
                )
                Spacer(Modifier.width(8.dp))
                Box(
                    Modifier
                        .size(8.dp)
                        .clip(CircleShape)
                        .background(colors.difficulty(row.difficulty))
                )
                Spacer(Modifier.width(6.dp))
                Text(
                    text = row.difficulty,
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}
