package com.parthdeuri.quickml.ui.feed

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

private val DIFFICULTIES = listOf("Easy", "Medium", "Hard")
private val STATUS_OPTIONS = listOf("SOLVED" to "Solved", "UNSOLVED" to "Unsolved")

private fun Set<String>.toggled(value: String): Set<String> =
    if (contains(value)) this - value else this + value

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun FilterSheet(
    current: Filters,
    allTopics: List<String>,
    onApply: (Filters) -> Unit,
    onClear: () -> Unit,
    onDismiss: () -> Unit
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    var difficulties by remember { mutableStateOf(current.difficulties) }
    var topics by remember { mutableStateOf(current.topics) }
    var onlyFull by remember { mutableStateOf(current.onlyFull) }
    var statuses by remember { mutableStateOf(current.statuses) }

    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = sheetState) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp)
                .navigationBarsPadding()
                .padding(bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Filters",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )

            Text("Difficulty", style = MaterialTheme.typography.titleSmall)
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                DIFFICULTIES.forEach { d ->
                    FilterChip(
                        selected = d in difficulties,
                        onClick = { difficulties = difficulties.toggled(d) },
                        label = { Text(d) }
                    )
                }
            }

            Text("Status", style = MaterialTheme.typography.titleSmall)
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                STATUS_OPTIONS.forEach { (value, label) ->
                    FilterChip(
                        selected = value in statuses,
                        onClick = { statuses = statuses.toggled(value) },
                        label = { Text(label) }
                    )
                }
            }

            Text("Topics", style = MaterialTheme.typography.titleSmall)
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                allTopics.forEach { t ->
                    FilterChip(
                        selected = t in topics,
                        onClick = { topics = topics.toggled(t) },
                        label = { Text(t) }
                    )
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Only problems with full notes",
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier.weight(1f)
                )
                Switch(checked = onlyFull, onCheckedChange = { onlyFull = it })
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                OutlinedButton(
                    onClick = {
                        difficulties = emptySet()
                        topics = emptySet()
                        onlyFull = false
                        statuses = emptySet()
                        onClear()
                    },
                    modifier = Modifier.weight(1f)
                ) { Text("Clear") }
                Button(
                    onClick = { onApply(
                            Filters(
                                difficulties = difficulties,
                                topics = topics,
                                onlyFull = onlyFull,
                                statuses = statuses
                            )
                        ) },
                    modifier = Modifier.weight(1f)
                ) { Text("Apply") }
            }
        }
    }
}
