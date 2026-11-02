package com.parthdeuri.quickml.ui.sheet

import android.widget.Toast
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Bookmark
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CheckboxDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.parthdeuri.quickml.data.repo.ProblemRepository
import com.parthdeuri.quickml.data.repo.UserProblemDraft
import com.parthdeuri.quickml.ui.common.UserProblemDialog
import com.parthdeuri.quickml.ui.theme.QuickMLTheme
import kotlinx.coroutines.launch

/** A pending "mark all" action awaiting confirmation. */
private data class BulkAction(val section: String, val solved: Boolean, val count: Int)

/** The own-problem form: [slug] null = add. */
private data class ProblemEditor(val slug: String?, val draft: UserProblemDraft?)

@Composable
fun SheetScreen(
    viewModel: SheetViewModel,
    onOpenProblem: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    var query by rememberSaveable { mutableStateOf("") }
    var pending by remember { mutableStateOf<BulkAction?>(null) }
    var editor by remember { mutableStateOf<ProblemEditor?>(null) }
    var deleting by remember { mutableStateOf<SheetRow?>(null) }
    val scope = rememberCoroutineScope()
    val context = LocalContext.current

    LaunchedEffect(query) { viewModel.setQuery(query) }

    Box(modifier = modifier.fillMaxSize()) {
    Column(modifier = Modifier.fillMaxSize().statusBarsPadding()) {
        Column(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                SheetPicker(
                    name = state.sheetName.ifEmpty { "Sheet" },
                    sheets = state.sheets,
                    selectedId = state.sheetId,
                    onSelect = viewModel::selectSheet,
                    style = MaterialTheme.typography.titleLarge,
                    modifier = Modifier.weight(1f)
                )
                Text(
                    text = state.solved.toString() + " / " + state.total,
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            ThinProgress(
                fraction = if (state.total == 0) 0f else state.solved.toFloat() / state.total,
                modifier = Modifier.fillMaxWidth()
            )
            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                placeholder = { Text("Search by title") },
                leadingIcon = { Icon(Icons.Filled.Search, contentDescription = null) },
                trailingIcon = {
                    if (query.isNotEmpty()) {
                        IconButton(onClick = { query = "" }) {
                            Icon(Icons.Filled.Close, contentDescription = "Clear search")
                        }
                    }
                }
            )
        }

        when {
            state.loading -> {
                Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator(color = QuickMLTheme.colors.accent)
                }
            }

            state.sections.isEmpty() -> {
                Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
                    Text(
                        text = when {
                            query.isNotBlank() -> "No cards match your search."
                            state.sheetId == ProblemRepository.MINE_SHEET_ID ->
                                "No cards of your own yet.\nTap + to add one."
                            else -> "No cards in this sheet."
                        },
                        textAlign = TextAlign.Center,
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            else -> {
                LazyColumn(
                    modifier = Modifier.weight(1f).fillMaxWidth(),
                    // Room for the + button.
                    contentPadding = PaddingValues(bottom = 88.dp)
                ) {
                    state.sections.forEach { section ->
                        item(key = "section:" + section.name, contentType = "header") {
                            SectionHeader(
                                section = section,
                                onToggleCollapsed = { viewModel.toggleCollapsed(section.name) },
                                onMarkAll = { solved ->
                                    pending = BulkAction(section.name, solved, section.total)
                                }
                            )
                        }
                        if (!section.collapsed) {
                            items(
                                items = section.rows,
                                key = { row -> row.slug },
                                contentType = { "row" }
                            ) { row ->
                                ProblemRow(
                                    row = row,
                                    onToggleSolved = viewModel::toggleSolved,
                                    onOpen = onOpenProblem,
                                    onEdit = { slug ->
                                        scope.launch {
                                            val draft = viewModel.loadDraft(slug)
                                            if (draft != null) editor = ProblemEditor(slug, draft)
                                        }
                                    },
                                    onDelete = { deleting = it }
                                )
                            }
                        }
                    }
                }
            }
        }
    }
        if (!state.loading) {
            FloatingActionButton(
                onClick = { editor = ProblemEditor(slug = null, draft = null) },
                containerColor = QuickMLTheme.colors.accent,
                contentColor = MaterialTheme.colorScheme.onPrimary,
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(16.dp)
            ) {
                Icon(Icons.Filled.Add, contentDescription = "Add your own card")
            }
        }
    }

    val edit = editor
    if (edit != null) {
        UserProblemDialog(
            initial = edit.draft,
            onSave = { draft ->
                editor = null
                viewModel.saveProblem(edit.slug, draft) { added ->
                    if (added) {
                        Toast.makeText(context, "Added to My cards", Toast.LENGTH_SHORT).show()
                    }
                }
            },
            onDismiss = { editor = null }
        )
    }

    val toDelete = deleting
    if (toDelete != null) {
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("Delete card?") },
            text = { Text("\"" + toDelete.title + "\" and its progress will be deleted.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        viewModel.deleteProblem(toDelete.slug)
                        deleting = null
                    }
                ) { Text("Delete") }
            },
            dismissButton = {
                TextButton(onClick = { deleting = null }) { Text("Cancel") }
            }
        )
    }

    val action = pending
    if (action != null) {
        val title = if (action.solved) "Mark all learned?" else "Mark all not learned?"
        val body = if (action.solved) {
            "All " + action.count + " cards in " + action.section +
                " will be marked learned and scheduled for revision."
        } else {
            "All " + action.count + " cards in " + action.section +
                " will be marked not learned and lose their revision schedule."
        }
        AlertDialog(
            onDismissRequest = { pending = null },
            title = { Text(title) },
            text = { Text(body) },
            confirmButton = {
                TextButton(
                    onClick = {
                        viewModel.markSection(action.section, action.solved)
                        pending = null
                    }
                ) { Text("Confirm") }
            },
            dismissButton = {
                TextButton(onClick = { pending = null }) { Text("Cancel") }
            }
        )
    }
}

@Composable
private fun ThinProgress(fraction: Float, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .height(4.dp)
            .clip(RoundedCornerShape(2.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
    ) {
        val f = fraction.coerceIn(0f, 1f)
        if (f > 0f) {
            Box(
                Modifier
                    .fillMaxHeight()
                    .fillMaxWidth(f)
                    .clip(RoundedCornerShape(2.dp))
                    .background(QuickMLTheme.colors.accent)
            )
        }
    }
}

@Composable
private fun SectionHeader(
    section: SheetSectionUi,
    onToggleCollapsed: () -> Unit,
    onMarkAll: (Boolean) -> Unit
) {
    var menuOpen by remember { mutableStateOf(false) }
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onToggleCollapsed)
            .padding(start = 16.dp, end = 4.dp, top = 10.dp, bottom = 6.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                imageVector = if (section.collapsed) Icons.Filled.ExpandMore else Icons.Filled.ExpandLess,
                contentDescription = if (section.collapsed) "Expand" else "Collapse",
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(20.dp)
            )
            Spacer(Modifier.width(8.dp))
            Text(
                text = section.name,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f)
            )
            Text(
                text = section.solved.toString() + " / " + section.total,
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Box {
                IconButton(onClick = { menuOpen = true }) {
                    Icon(
                        imageVector = Icons.Filled.MoreVert,
                        contentDescription = "Section actions",
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                    DropdownMenuItem(
                        text = { Text("Mark all learned") },
                        onClick = {
                            menuOpen = false
                            onMarkAll(true)
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Mark all not learned") },
                        onClick = {
                            menuOpen = false
                            onMarkAll(false)
                        }
                    )
                }
            }
        }
        Spacer(Modifier.height(4.dp))
        ThinProgress(
            fraction = if (section.total == 0) 0f else section.solved.toFloat() / section.total,
            modifier = Modifier.fillMaxWidth().padding(end = 12.dp)
        )
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun ProblemRow(
    row: SheetRow,
    onToggleSolved: (String) -> Unit,
    onOpen: (String) -> Unit,
    /** Own problems only (long-press menu). */
    onEdit: (String) -> Unit,
    onDelete: (SheetRow) -> Unit
) {
    var menuOpen by remember { mutableStateOf(false) }
    val openMenu: () -> Unit = { menuOpen = true }
    val colors = QuickMLTheme.colors
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 8.dp, end = 16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Checkbox(
            checked = row.solved,
            onCheckedChange = { onToggleSolved(row.slug) },
            colors = CheckboxDefaults.colors(
                checkedColor = colors.accent,
                checkmarkColor = MaterialTheme.colorScheme.onPrimary
            )
        )
        Box(modifier = Modifier.weight(1f)) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(8.dp))
                .combinedClickable(
                    onClick = { onOpen(row.slug) },
                    onLongClick = if (row.userAdded) openMenu else null
                )
                .padding(horizontal = 6.dp, vertical = 6.dp),
            verticalArrangement = Arrangement.spacedBy(3.dp)
        ) {
            Text(
                text = row.title,
                style = MaterialTheme.typography.bodyLarge,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Box(
                    Modifier
                        .size(8.dp)
                        .clip(CircleShape)
                        .background(colors.difficulty(row.difficulty))
                )
                Text(
                    text = row.difficulty,
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                if (row.hasNotes) {
                    Icon(
                        imageVector = Icons.Filled.Description,
                        contentDescription = "Has full notes",
                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.size(14.dp)
                    )
                }
                if (row.hasNote) {
                    Icon(
                        imageVector = Icons.Filled.Edit,
                        contentDescription = "Has a note",
                        tint = colors.accent,
                        modifier = Modifier.size(14.dp)
                    )
                }
                if (row.bookmarked) {
                    Icon(
                        imageVector = Icons.Filled.Bookmark,
                        contentDescription = "Bookmarked",
                        tint = colors.accent,
                        modifier = Modifier.size(14.dp)
                    )
                }
                if (row.due) {
                    Text(
                        text = "due",
                        style = MaterialTheme.typography.labelSmall,
                        color = colors.onTint(colors.accent),
                        modifier = Modifier
                            .clip(RoundedCornerShape(50))
                            .background(colors.accent.copy(alpha = 0.16f))
                            .padding(horizontal = 8.dp, vertical = 2.dp)
                    )
                }
            }
        }
        DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
            DropdownMenuItem(
                text = { Text("Edit card") },
                onClick = {
                    menuOpen = false
                    onEdit(row.slug)
                }
            )
            DropdownMenuItem(
                text = { Text("Delete card", color = colors.danger) },
                onClick = {
                    menuOpen = false
                    onDelete(row)
                }
            )
        }
        }
    }
}
