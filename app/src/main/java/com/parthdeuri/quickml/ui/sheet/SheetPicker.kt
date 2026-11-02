package com.parthdeuri.quickml.ui.sheet

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.parthdeuri.quickml.data.repo.SheetInfo
import com.parthdeuri.quickml.ui.theme.QuickMLTheme

/**
 * The selected sheet's name as a dropdown listing every sheet with solved / total.
 * Used by the Feed top bar and the Sheet tab header (both write the same setting).
 */
@Composable
fun SheetPicker(
    name: String,
    sheets: List<SheetInfo>,
    selectedId: String,
    onSelect: (String) -> Unit,
    style: TextStyle,
    modifier: Modifier = Modifier
) {
    var open by remember { mutableStateOf(false) }
    val canPick = sheets.size > 1
    Box(modifier = modifier) {
        Row(
            modifier = Modifier
                .clip(RoundedCornerShape(8.dp))
                .clickable(enabled = canPick) { open = true }
                .padding(vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = name,
                style = style,
                fontWeight = FontWeight.Bold,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f, fill = false)
            )
            if (canPick) {
                Icon(
                    imageVector = Icons.Filled.ArrowDropDown,
                    contentDescription = "Choose sheet",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            sheets.forEach { sheet ->
                val selected = sheet.id == selectedId
                DropdownMenuItem(
                    text = {
                        Text(
                            text = sheet.name,
                            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
                            color = if (selected) QuickMLTheme.colors.accent else Color.Unspecified
                        )
                    },
                    leadingIcon = {
                        if (selected) {
                            Icon(
                                Icons.Filled.Check,
                                contentDescription = null,
                                tint = QuickMLTheme.colors.accent
                            )
                        } else {
                            Spacer(Modifier.size(24.dp))
                        }
                    },
                    trailingIcon = {
                        Text(
                            text = sheet.solved.toString() + "/" + sheet.total,
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    },
                    onClick = {
                        open = false
                        onSelect(sheet.id)
                    }
                )
            }
        }
    }
}
