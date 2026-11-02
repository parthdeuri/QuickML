package com.parthdeuri.quickml.ui.settings

import android.Manifest
import android.app.TimePickerDialog
import android.content.pm.PackageManager
import android.os.Build
import android.text.format.DateFormat
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.filled.Check
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.Role
import com.parthdeuri.quickml.data.settings.AccentChoice
import com.parthdeuri.quickml.speech.LocalSpeaker
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.parthdeuri.quickml.AppContainer
import com.parthdeuri.quickml.data.settings.StudySettings
import com.parthdeuri.quickml.data.settings.ThemeMode
import com.parthdeuri.quickml.reminder.ReminderScheduler
import com.parthdeuri.quickml.ui.theme.QuickMLTheme
import java.time.LocalTime
import java.time.format.DateTimeFormatter
import java.time.format.FormatStyle

@Composable
fun SettingsScreen(
    container: AppContainer,
    onBack: () -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val settings = container.reminderSettings

    var enabled by remember { mutableStateOf(settings.enabled) }
    var hour by remember { mutableIntStateOf(settings.hour) }
    var minute by remember { mutableIntStateOf(settings.minute) }
    var denied by remember { mutableStateOf(false) }

    val study = container.studySettings
    val cardPrefs by study.cardPrefs.collectAsState()
    val themeMode by study.themeMode.collectAsState()
    val accent by study.accent.collectAsState()
    val cardColor = QuickMLTheme.colors.card

    fun turnOn() {
        settings.enabled = true
        enabled = true
        denied = false
        ReminderScheduler.schedule(context, hour, minute)
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            turnOn()
        } else {
            settings.enabled = false
            enabled = false
            denied = true
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .statusBarsPadding()
            .verticalScroll(rememberScrollState())
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 4.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = onBack) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
            }
            Text(
                text = "Settings",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )
        }

        Column(
            modifier = Modifier
                .padding(horizontal = 16.dp, vertical = 8.dp)
                .fillMaxWidth()
                .clip(RoundedCornerShape(20.dp))
                .background(cardColor)
                .padding(horizontal = 16.dp, vertical = 16.dp)
        ) {
            Text("Theme", style = MaterialTheme.typography.bodyLarge)
            Text(
                text = "System follows your phone's light / dark setting",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Row(
                modifier = Modifier.padding(top = 6.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                ThemeMode.values().forEach { mode ->
                    FilterChip(
                        selected = themeMode == mode,
                        onClick = { study.setThemeMode(mode) },
                        label = { Text(mode.label) }
                    )
                }
            }

            Text(
                text = "Accent colour",
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(top = 14.dp)
            )
            Text(
                text = "Tabs, chips, links and the heatmap; the app icon and widget stay teal",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Row(
                modifier = Modifier.padding(top = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                AccentChoice.values().forEach { choice ->
                    AccentSwatch(
                        choice = choice,
                        selected = accent == choice,
                        onClick = { study.setAccent(choice) }
                    )
                }
            }
        }

        ReadAloudCard(study = study, speaker = LocalSpeaker.current)

        AiSettingsCard(container = container)

        Column(
            modifier = Modifier
                .padding(horizontal = 16.dp, vertical = 8.dp)
                .fillMaxWidth()
                .clip(RoundedCornerShape(20.dp))
                .background(cardColor)
                .padding(vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(0.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("Daily reminder", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        text = "A notification when problems are due or you haven't revised today",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Switch(
                    checked = enabled,
                    onCheckedChange = { wantOn ->
                        if (!wantOn) {
                            settings.enabled = false
                            enabled = false
                            ReminderScheduler.cancel(context)
                        } else if (
                            Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
                            ContextCompat.checkSelfPermission(
                                context,
                                Manifest.permission.POST_NOTIFICATIONS
                            ) != PackageManager.PERMISSION_GRANTED
                        ) {
                            permissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                        } else {
                            turnOn()
                        }
                    }
                )
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        TimePickerDialog(
                            context,
                            { _, pickedHour, pickedMinute ->
                                hour = pickedHour
                                minute = pickedMinute
                                settings.hour = pickedHour
                                settings.minute = pickedMinute
                                if (enabled) {
                                    ReminderScheduler.schedule(context, pickedHour, pickedMinute)
                                }
                            },
                            hour,
                            minute,
                            DateFormat.is24HourFormat(context)
                        ).show()
                    }
                    .padding(horizontal = 16.dp, vertical = 14.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Reminder time",
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier.weight(1f)
                )
                Text(
                    text = LocalTime.of(hour, minute)
                        .format(DateTimeFormatter.ofLocalizedTime(FormatStyle.SHORT)),
                    style = MaterialTheme.typography.bodyLarge,
                    color = QuickMLTheme.colors.accent
                )
            }

            if (denied) {
                Text(
                    text = "Notifications are blocked for QuickML, so the reminder stays off. " +
                        "Allow notifications in the system settings to use it.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                )
            }
        }

        Column(
            modifier = Modifier
                .padding(horizontal = 16.dp, vertical = 8.dp)
                .fillMaxWidth()
                .clip(RoundedCornerShape(20.dp))
                .background(cardColor)
                .padding(vertical = 8.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        "Family quiz before revealing (Revise mode)",
                        style = MaterialTheme.typography.bodyLarge
                    )
                    Text(
                        text = "Guess the technique family before the topics and families are shown",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Switch(
                    checked = cardPrefs.quizBeforeReveal,
                    onCheckedChange = { study.setQuizBeforeReveal(it) }
                )
            }

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp)
            ) {
                Text("Think timer before revealing", style = MaterialTheme.typography.bodyLarge)
                Text(
                    text = "A countdown on the Problem tab in Revise and Quiz modes",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Row(
                    modifier = Modifier.padding(top = 6.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    StudySettings.THINK_TIMER_CHOICES.forEach { minutes ->
                        FilterChip(
                            selected = cardPrefs.thinkTimerMinutes == minutes,
                            onClick = { study.setThinkTimerMinutes(minutes) },
                            label = { Text(if (minutes == 0) "Off" else "$minutes min") }
                        )
                    }
                }
            }
        }

        DataSettingsCard(
            container = container,
            onRestored = {
                // The backup may carry other reminder settings.
                enabled = settings.enabled
                hour = settings.hour
                minute = settings.minute
                denied = false
            }
        )

        Spacer(Modifier.height(16.dp))
        Spacer(Modifier.navigationBarsPadding())
    }
}

/** Round colour swatch; the selected one is ringed and shows a check. */
@Composable
private fun AccentSwatch(choice: AccentChoice, selected: Boolean, onClick: () -> Unit) {
    val color = Color(if (QuickMLTheme.colors.isDark) choice.dark else choice.light)
    Box(
        modifier = Modifier
            .size(36.dp)
            .clip(CircleShape)
            .background(color)
            .then(
                if (selected) {
                    Modifier.border(3.dp, MaterialTheme.colorScheme.onSurface, CircleShape)
                } else {
                    Modifier
                }
            )
            .clickable(role = Role.RadioButton, onClick = onClick)
            .semantics {
                contentDescription = choice.label
                this.selected = selected
            },
        contentAlignment = Alignment.Center
    ) {
        if (selected) {
            Icon(
                imageVector = Icons.Filled.Check,
                contentDescription = null,
                tint = if (color.luminance() > 0.5f) Color.Black else Color.White,
                modifier = Modifier.size(18.dp)
            )
        }
    }
}
