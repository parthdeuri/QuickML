package com.parthdeuri.quickml.ui.settings

import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import com.parthdeuri.quickml.AppContainer
import com.parthdeuri.quickml.ai.AiError
import com.parthdeuri.quickml.ai.AiSettings
import com.parthdeuri.quickml.ai.ChatTurn
import com.parthdeuri.quickml.ui.theme.QuickMLTheme
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

private const val KEY_PAGE_URL = "https://aistudio.google.com/apikey"

/** Settings card for the AI master toggle, Gemini key and model; key and model only when on. */
@Composable
fun AiSettingsCard(container: AppContainer) {
    val settings = container.aiSettings
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val enabled by settings.enabled.collectAsState()
    val hasKey by settings.hasKey.collectAsState()
    val model by settings.model.collectAsState()

    var keyInput by remember { mutableStateOf("") }
    var customMode by remember { mutableStateOf(model !in AiSettings.MODEL_PRESETS) }
    var customText by remember {
        mutableStateOf(if (model in AiSettings.MODEL_PRESETS) "" else model)
    }
    var menuOpen by remember { mutableStateOf(false) }
    var testing by remember { mutableStateOf(false) }
    var testResult by remember { mutableStateOf<String?>(null) }
    var testOk by remember { mutableStateOf(false) }
    var confirmDelete by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .padding(horizontal = 16.dp, vertical = 8.dp)
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(QuickMLTheme.colors.card)
            .padding(horizontal = 16.dp, vertical = 16.dp)
    ) {
        Text(
            text = "AI (Gemini)",
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(bottom = 8.dp)
        )
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(modifier = Modifier.weight(1f)) {
                Text("AI features", style = MaterialTheme.typography.bodyLarge)
                Text(
                    text = "Off: the app works as before and makes no network requests",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Switch(checked = enabled, onCheckedChange = { settings.setEnabled(it) })
        }

        if (enabled) {
            Text(
                text = "Gemini API key",
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(top = 14.dp)
            )
            Text(
                text = if (hasKey) "Key saved ✓" else "No key",
                style = MaterialTheme.typography.bodySmall,
                color = if (hasKey) QuickMLTheme.colors.accent else {
                    MaterialTheme.colorScheme.onSurfaceVariant
                }
            )
            OutlinedTextField(
                value = keyInput,
                onValueChange = { keyInput = it },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 6.dp),
                singleLine = true,
                placeholder = { Text("Paste your key") },
                visualTransformation = PasswordVisualTransformation(),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password)
            )
            Row(
                modifier = Modifier.padding(top = 6.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Button(
                    onClick = {
                        if (settings.setKey(keyInput)) {
                            keyInput = ""
                            testResult = null
                        } else {
                            testOk = false
                            testResult = "Could not store the key on this phone"
                        }
                    },
                    enabled = keyInput.isNotBlank()
                ) { Text("Save key") }
                TextButton(
                    onClick = {
                        settings.clearKey()
                        testResult = null
                    },
                    enabled = hasKey
                ) { Text("Clear key") }
            }

            Text(
                text = "Model",
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(top = 14.dp)
            )
            Box {
                OutlinedButton(onClick = { menuOpen = true }) {
                    Text(if (customMode) "Custom…" else model)
                }
                DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                    AiSettings.MODEL_PRESETS.forEach { preset ->
                        DropdownMenuItem(
                            text = { Text(preset) },
                            onClick = {
                                settings.setModel(preset)
                                customMode = false
                                menuOpen = false
                            }
                        )
                    }
                    DropdownMenuItem(
                        text = { Text("Custom…") },
                        onClick = {
                            customMode = true
                            menuOpen = false
                        }
                    )
                }
            }
            if (customMode) {
                OutlinedTextField(
                    value = customText,
                    onValueChange = {
                        customText = it
                        settings.setModel(it)
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 6.dp),
                    singleLine = true,
                    placeholder = { Text("Model name, e.g. gemini-3.5-flash-lite") }
                )
            }

            Row(
                modifier = Modifier.padding(top = 12.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedButton(
                    onClick = {
                        testing = true
                        testResult = null
                        scope.launch {
                            val message = try {
                                container.createAiProvider()
                                    .ask(
                                        "Reply with one short word.",
                                        listOf(ChatTurn(ChatTurn.ROLE_USER, "Say OK"))
                                    )
                                    .first()
                                null
                            } catch (e: CancellationException) {
                                throw e
                            } catch (e: AiError) {
                                e.message ?: AiError.Unknown.message
                            } catch (e: Exception) {
                                AiError.Unknown.message
                            }
                            testOk = message == null
                            testResult = message ?: "Key works ✓"
                            testing = false
                        }
                    },
                    enabled = hasKey && !testing
                ) { Text(if (testing) "Testing…" else "Test key") }
                val result = testResult
                if (result != null) {
                    Text(
                        text = result,
                        style = MaterialTheme.typography.bodySmall,
                        color = if (testOk) {
                            QuickMLTheme.colors.success
                        } else {
                            QuickMLTheme.colors.danger
                        },
                        modifier = Modifier.weight(1f)
                    )
                }
            }

            Text(
                text = "Get a free key at aistudio.google.com/apikey",
                style = MaterialTheme.typography.bodyMedium,
                color = QuickMLTheme.colors.accent,
                textDecoration = TextDecoration.Underline,
                modifier = Modifier
                    .padding(top = 14.dp)
                    .clickable {
                        try {
                            context.startActivity(
                                Intent(Intent.ACTION_VIEW, Uri.parse(KEY_PAGE_URL))
                            )
                        } catch (e: ActivityNotFoundException) {
                            Toast.makeText(context, "No browser found", Toast.LENGTH_SHORT).show()
                        }
                    }
            )
            Text(
                text = "Your questions and the card text are sent to Google. Free-tier " +
                    "prompts may be used to improve Google products. Your key stays encrypted " +
                    "on this phone and is not included in backups.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 8.dp)
            )
            OutlinedButton(
                onClick = { confirmDelete = true },
                modifier = Modifier.padding(top = 10.dp)
            ) { Text("Delete all chats") }
        }
    }

    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text("Delete all chats?") },
            text = { Text("The Ask conversations of every card will be deleted.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        confirmDelete = false
                        scope.launch {
                            container.chatDao.deleteAll()
                            Toast.makeText(context, "Chats deleted", Toast.LENGTH_SHORT).show()
                        }
                    }
                ) { Text("Delete") }
            },
            dismissButton = {
                TextButton(onClick = { confirmDelete = false }) { Text("Cancel") }
            }
        )
    }
}
