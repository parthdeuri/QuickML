package com.parthdeuri.quickml.ui.settings

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import com.parthdeuri.quickml.data.settings.StudySettings
import com.parthdeuri.quickml.speech.Speaker
import com.parthdeuri.quickml.ui.theme.QuickMLTheme

private const val SAMPLE_TEXT =
    "This is how QuickML reads a card aloud. Choose the voice and speed you like best."

/** Settings card for read-aloud speed and voice; hidden when there is no [speaker]. */
@Composable
fun ReadAloudCard(study: StudySettings, speaker: Speaker?) {
    if (speaker == null) return
    val rate by study.speechRate.collectAsState()
    val voiceName by study.voiceName.collectAsState()
    val voices by speaker.voices.collectAsState()
    val speakingId by speaker.speakingId.collectAsState()
    var picking by remember { mutableStateOf(false) }

    // Start the engine (silently) so the voice list is there when the card is opened.
    LaunchedEffect(speaker) { speaker.prepare() }

    // An unknown stored voice reads with the language default, so show it as such.
    val current = voices.firstOrNull { it.name == voiceName }
    val playing = speakingId == "voice-preview"

    Column(
        modifier = Modifier
            .padding(horizontal = 16.dp, vertical = 8.dp)
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(QuickMLTheme.colors.card)
            .padding(horizontal = 16.dp, vertical = 16.dp)
    ) {
        Text("Read aloud speed", style = MaterialTheme.typography.bodyLarge)
        Row(
            modifier = Modifier.padding(top = 6.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            StudySettings.SPEECH_RATE_CHOICES.forEach { choice ->
                FilterChip(
                    selected = rate == choice,
                    onClick = { study.setSpeechRate(choice) },
                    label = { Text("${choice}x") }
                )
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 14.dp)
                .clickable { picking = true },
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text("Voice", style = MaterialTheme.typography.bodyLarge)
                Text(
                    text = current?.label ?: "System default",
                    style = MaterialTheme.typography.bodySmall,
                    color = QuickMLTheme.colors.accent
                )
            }
            OutlinedButton(onClick = { speaker.preview(SAMPLE_TEXT) }) {
                Text(if (playing) "Stop" else "Test voice")
            }
        }
        Text(
            text = "More voices come from your phone's text-to-speech settings",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 6.dp)
        )
    }

    if (picking) {
        AlertDialog(
            onDismissRequest = { picking = false },
            title = { Text("Voice") },
            text = {
                LazyColumn(modifier = Modifier.heightIn(max = 360.dp)) {
                    item {
                        VoiceRow(
                            label = "System default",
                            selected = current == null,
                            onClick = {
                                study.setVoiceName(null)
                                picking = false
                            }
                        )
                    }
                    items(voices, key = { it.name }) { option ->
                        VoiceRow(
                            label = option.label,
                            selected = option.name == current?.name,
                            onClick = {
                                study.setVoiceName(option.name)
                                picking = false
                            }
                        )
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { picking = false }) { Text("Close") }
            }
        )
    }
}

@Composable
private fun VoiceRow(label: String, selected: Boolean, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        RadioButton(selected = selected, onClick = onClick)
        Text(text = label, style = MaterialTheme.typography.bodyMedium)
    }
}
