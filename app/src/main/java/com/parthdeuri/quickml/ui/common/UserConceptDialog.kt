package com.parthdeuri.quickml.ui.common

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
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.parthdeuri.quickml.data.repo.USER_DIFFICULTIES
import com.parthdeuri.quickml.data.repo.UserConceptDraft
import com.parthdeuri.quickml.data.repo.detectPlatform
import com.parthdeuri.quickml.data.repo.normalizeLink
import com.parthdeuri.quickml.ui.theme.QuickMLTheme

/** Monospace style shared by the code fields. */
val CodeFieldStyle = TextStyle(
    fontFamily = FontFamily.Monospace,
    fontSize = 13.sp,
    lineHeight = 19.sp
)

/** Keyboard for code: no auto-capitalisation, plain ASCII layout. */
val CodeKeyboard = KeyboardOptions(
    capitalization = KeyboardCapitalization.None,
    keyboardType = KeyboardType.Ascii
)

/** Add / edit form of an own problem ([initial] null = add). */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun UserConceptDialog(
    initial: UserConceptDraft?,
    onSave: (UserConceptDraft) -> Unit,
    onDismiss: () -> Unit
) {
    var draft by remember { mutableStateOf(initial ?: UserConceptDraft()) }
    FullScreenDialog(
        title = if (initial == null) "Add card" else "Edit card",
        onDismiss = onDismiss,
        onConfirm = { onSave(draft) },
        confirmEnabled = draft.isValid
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            OutlinedTextField(
                value = draft.title,
                onValueChange = { draft = draft.copy(title = it) },
                label = { Text("Title (required)") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth()
            )
            OutlinedTextField(
                value = draft.link,
                onValueChange = { draft = draft.copy(link = it) },
                label = { Text("Link (optional)") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                supportingText = {
                    if (draft.link.isNotBlank()) {
                        Text("Platform: " + platformName(detectPlatform(normalizeLink(draft.link))))
                    }
                },
                modifier = Modifier.fillMaxWidth()
            )
            Text(
                text = "Difficulty",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                USER_DIFFICULTIES.forEach { difficulty ->
                    FilterChip(
                        selected = draft.difficulty == difficulty,
                        onClick = { draft = draft.copy(difficulty = difficulty) },
                        label = { Text(difficulty) }
                    )
                }
            }
            OutlinedTextField(
                value = draft.topics,
                onValueChange = { draft = draft.copy(topics = it) },
                label = { Text("Topics (comma-separated)") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth()
            )
            val topics = draft.topicList
            if (topics.isNotEmpty()) {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    topics.forEach { topic ->
                        Text(
                            text = topic,
                            color = QuickMLTheme.colors.accent,
                            style = MaterialTheme.typography.labelMedium,
                            modifier = Modifier
                                .padding(end = 2.dp)
                        )
                    }
                }
            }
            OutlinedTextField(
                value = draft.description,
                onValueChange = { draft = draft.copy(description = it) },
                label = { Text("Description") },
                minLines = 3,
                modifier = Modifier.fillMaxWidth()
            )
            OutlinedTextField(
                value = draft.intuition,
                onValueChange = { draft = draft.copy(intuition = it) },
                label = { Text("Intuition") },
                minLines = 2,
                modifier = Modifier.fillMaxWidth()
            )
            OutlinedTextField(
                value = draft.approach,
                onValueChange = { draft = draft.copy(approach = it) },
                label = { Text("Approach (one step per line)") },
                minLines = 3,
                modifier = Modifier.fillMaxWidth()
            )
            OutlinedTextField(
                value = draft.solution,
                onValueChange = { draft = draft.copy(solution = it) },
                label = { Text("Code") },
                minLines = 6,
                textStyle = CodeFieldStyle,
                keyboardOptions = CodeKeyboard,
                modifier = Modifier.fillMaxWidth()
            )
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = draft.time,
                    onValueChange = { draft = draft.copy(time = it) },
                    label = { Text("Time") },
                    singleLine = true,
                    modifier = Modifier.weight(1f)
                )
                OutlinedTextField(
                    value = draft.space,
                    onValueChange = { draft = draft.copy(space = it) },
                    label = { Text("Space") },
                    singleLine = true,
                    modifier = Modifier.weight(1f)
                )
            }
            OutlinedTextField(
                value = draft.notes,
                onValueChange = { draft = draft.copy(notes = it) },
                label = { Text("Notes") },
                minLines = 2,
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(Modifier.height(24.dp))
        }
    }
}

/** Full-screen monospace editor for the user's own solution ("Mine"). */
@Composable
fun SolutionEditorDialog(
    initial: String,
    onSave: (String) -> Unit,
    onDismiss: () -> Unit
) {
    var code by remember { mutableStateOf(initial) }
    FullScreenDialog(
        title = "My implementation",
        onDismiss = onDismiss,
        onConfirm = { onSave(code) }
    ) {
        OutlinedTextField(
            value = code,
            onValueChange = { code = it },
            placeholder = { Text("Paste or type your implementation", style = CodeFieldStyle) },
            textStyle = CodeFieldStyle,
            keyboardOptions = CodeKeyboard,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(start = 12.dp, end = 12.dp, bottom = 12.dp)
        )
    }
}

private fun platformName(platform: String): String = when (platform) {
    "leetcode" -> "LeetCode"
    "gfg" -> "GFG"
    "codingninjas" -> "Coding Ninjas"
    else -> "Other"
}
