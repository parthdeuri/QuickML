package com.parthdeuri.quickml.ui.settings

import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.parthdeuri.quickml.AppContainer
import com.parthdeuri.quickml.data.backup.BackupException
import com.parthdeuri.quickml.data.backup.BackupFile
import com.parthdeuri.quickml.data.repo.ImportPreview
import com.parthdeuri.quickml.data.repo.ProblemRepository
import com.parthdeuri.quickml.ui.theme.QuickMLTheme
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.time.LocalDate

/**
 * Settings card: back up / restore progress (Storage Access Framework, JSON) and import a list
 * of solved problems. [onRestored] lets the screen re-read the restored settings.
 */
@Composable
fun DataSettingsCard(container: AppContainer, onRestored: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val backupManager = container.backupManager
    var busy by remember { mutableStateOf(false) }
    var restoreCandidate by remember { mutableStateOf<BackupFile?>(null) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var showImport by remember { mutableStateOf(false) }

    fun toast(message: String) {
        Toast.makeText(context, message, Toast.LENGTH_LONG).show()
    }

    val createLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("application/json")
    ) { uri ->
        if (uri != null) {
            scope.launch {
                busy = true
                try {
                    val summary = backupManager.export(uri)
                    toast("Backed up " + summary.describe())
                } catch (e: CancellationException) {
                    throw e
                } catch (e: BackupException) {
                    errorMessage = e.message
                } catch (e: Exception) {
                    errorMessage = "Could not write the backup file."
                } finally {
                    busy = false
                }
            }
        }
    }

    val openLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            scope.launch {
                busy = true
                try {
                    restoreCandidate = backupManager.read(uri)
                } catch (e: CancellationException) {
                    throw e
                } catch (e: BackupException) {
                    errorMessage = e.message
                } catch (e: Exception) {
                    errorMessage = "This file isn't a QuickML backup."
                } finally {
                    busy = false
                }
            }
        }
    }

    Column(
        modifier = Modifier
            .padding(horizontal = 16.dp, vertical = 8.dp)
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(QuickMLTheme.colors.card)
            .padding(vertical = 8.dp)
    ) {
        ActionRow(
            title = "Back up progress",
            subtitle = "Save solved problems, reviews, notes, own problems and settings to a file",
            enabled = !busy,
            onClick = { createLauncher.launch("quickml-backup-" + LocalDate.now() + ".json") }
        )
        ActionRow(
            title = "Restore from backup",
            subtitle = "Replace your current progress with a backup file",
            enabled = !busy,
            onClick = {
                openLauncher.launch(arrayOf("application/json", "text/plain", "application/octet-stream"))
            }
        )
        ActionRow(
            title = "Import solved problems",
            subtitle = "Paste LeetCode / GFG links, slugs or titles to mark them solved",
            enabled = !busy,
            onClick = { showImport = true }
        )
    }

    val candidate = restoreCandidate
    if (candidate != null) {
        val summary = remember(candidate) { backupManager.summaryOf(candidate) }
        AlertDialog(
            onDismissRequest = { restoreCandidate = null },
            title = { Text("Restore backup?") },
            text = {
                Text(
                    "This replaces your current progress (" + summary.describe() + ")" +
                        if (candidate.exportedAt.isNotBlank()) {
                            ", backed up " + candidate.exportedAt.take(10) + "."
                        } else {
                            "."
                        }
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        restoreCandidate = null
                        scope.launch {
                            busy = true
                            try {
                                backupManager.restore(candidate)
                                onRestored()
                                toast("Restored " + summary.describe())
                            } catch (e: CancellationException) {
                                throw e
                            } catch (e: BackupException) {
                                errorMessage = e.message
                            } catch (e: Exception) {
                                errorMessage = "Could not restore the backup."
                            } finally {
                                busy = false
                            }
                        }
                    }
                ) { Text("Restore") }
            },
            dismissButton = {
                TextButton(onClick = { restoreCandidate = null }) { Text("Cancel") }
            }
        )
    }

    val error = errorMessage
    if (error != null) {
        AlertDialog(
            onDismissRequest = { errorMessage = null },
            title = { Text("Something went wrong") },
            text = { Text(error) },
            confirmButton = {
                TextButton(onClick = { errorMessage = null }) { Text("OK") }
            }
        )
    }

    if (showImport) {
        ImportSolvedDialog(
            repository = container.repository,
            onDone = { count ->
                showImport = false
                val noun = if (count == 1) " problem" else " problems"
                toast("Marked " + count + noun + " solved")
            },
            onDismiss = { showImport = false }
        )
    }
}

@Composable
private fun ActionRow(title: String, subtitle: String, enabled: Boolean, onClick: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 10.dp)
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.bodyLarge,
            color = if (enabled) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = subtitle,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

/** Paste -> preview (matched / already solved / not found) -> mark the matches solved. */
@Composable
private fun ImportSolvedDialog(
    repository: ProblemRepository,
    onDone: (Int) -> Unit,
    onDismiss: () -> Unit
) {
    val scope = rememberCoroutineScope()
    var text by remember { mutableStateOf("") }
    var preview by remember { mutableStateOf<ImportPreview?>(null) }
    var busy by remember { mutableStateOf(false) }
    val current = preview

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Import solved problems") },
        text = {
            Column(
                modifier = Modifier
                    .heightIn(max = 420.dp)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                if (current == null) {
                    Text(
                        text = "LeetCode / GFG links, slugs or exact titles, one per line or " +
                            "separated by commas.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    OutlinedTextField(
                        value = text,
                        onValueChange = { text = it },
                        minLines = 5,
                        maxLines = 10,
                        placeholder = { Text("https://leetcode.com/problems/two-sum/\nvalid-anagram\nContains Duplicate") },
                        modifier = Modifier.fillMaxWidth()
                    )
                } else {
                    Text(
                        text = "Matched " + current.toMark.size +
                            " · Already solved " + current.alreadySolved +
                            " · Not found " + current.notFound.size,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold
                    )
                    if (current.toMark.size >= ProblemRepository.SPREAD_MIN_COUNT) {
                        Text(
                            text = "Their first reviews will be spread over the next " +
                                ProblemRepository.SPREAD_DAYS + " days.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    if (current.notFound.isNotEmpty()) {
                        Text(
                            text = "Not found:",
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        current.notFound.forEach { line ->
                            Text(
                                text = "• $line",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }
        },
        confirmButton = {
            if (current == null) {
                TextButton(
                    enabled = text.isNotBlank() && !busy,
                    onClick = {
                        scope.launch {
                            busy = true
                            try {
                                val input = text
                                preview = withContext(Dispatchers.Default) {
                                    repository.previewImport(input)
                                }
                            } finally {
                                busy = false
                            }
                        }
                    }
                ) { Text("Preview") }
            } else {
                val count = current.toMark.size
                TextButton(
                    enabled = count > 0 && !busy,
                    onClick = {
                        scope.launch {
                            busy = true
                            try {
                                repository.setSolved(current.toMark, true)
                                onDone(count)
                            } finally {
                                busy = false
                            }
                        }
                    }
                ) { Text("Mark " + count + " solved", color = if (count > 0) QuickMLTheme.colors.accent else MaterialTheme.colorScheme.onSurfaceVariant) }
            }
        },
        dismissButton = {
            TextButton(
                onClick = { if (current != null) preview = null else onDismiss() }
            ) { Text(if (current != null) "Back" else "Cancel") }
        }
    )
}
