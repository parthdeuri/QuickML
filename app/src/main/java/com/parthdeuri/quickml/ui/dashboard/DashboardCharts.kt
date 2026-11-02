package com.parthdeuri.quickml.ui.dashboard

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
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
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.RoundRect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.parthdeuri.quickml.domain.srs.Rating
import com.parthdeuri.quickml.domain.srs.label
import com.parthdeuri.quickml.ui.theme.QuickMLTheme
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.Locale
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** 0 | 1-2 | 3-5 | 6-10 | 11+ cards -> level 0..4. */
private fun heatLevel(count: Int): Int = when {
    count <= 0 -> 0
    count <= 2 -> 1
    count <= 5 -> 2
    count <= 10 -> 3
    else -> 4
}

@Composable
internal fun Heatmap(data: HeatmapData) {
    val weeks = data.weeks
    if (weeks.isEmpty()) return

    var selectedDate by remember { mutableStateOf<LocalDate?>(null) }
    // Level 0 (no activity) sits close to the card surface; level 4 is the deepest teal.
    val levels = QuickMLTheme.colors.heatmap
    val outline = MaterialTheme.colorScheme.onSurface
    val cells = weeks.flatMap { column -> column.filterNotNull() }
    val totalCards = cells.sumOf { it.count }
    val activeDays = cells.count { it.count > 0 }
    val summary = "Activity heatmap, last " + weeks.size + " weeks: " + totalCards +
        " cards viewed on " + activeDays + " days"
    val selectedCell = selectedDate?.let { d -> cells.firstOrNull { it.date == d } }

    BoxWithConstraints(Modifier.fillMaxWidth()) {
        val gap = 3.dp
        val cols = weeks.size
        val cell = (maxWidth - gap * (cols - 1)) / cols
        val stride = cell + gap

        Column {
            Box(Modifier.fillMaxWidth().height(16.dp)) {
                data.months.forEach { label ->
                    Text(
                        text = label.text,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        softWrap = false,
                        modifier = Modifier.offset(x = stride * label.week)
                    )
                }
            }
            Canvas(
                modifier = Modifier
                    .size(width = stride * cols - gap, height = stride * 7 - gap)
                    .semantics { contentDescription = summary }
                    .pointerInput(weeks, cell) {
                        detectTapGestures { offset ->
                            val step = (cell + gap).toPx()
                            val w = (offset.x / step).toInt()
                            val d = (offset.y / step).toInt()
                            selectedDate = weeks.getOrNull(w)?.getOrNull(d)?.date
                        }
                    }
            ) {
                val cellPx = cell.toPx()
                val stepPx = cellPx + gap.toPx()
                val radius = CornerRadius(cellPx * 0.22f)
                weeks.forEachIndexed { w, column ->
                    column.forEachIndexed { d, item ->
                        if (item != null) {
                            val topLeft = Offset(w * stepPx, d * stepPx)
                            drawRoundRect(
                                color = levels[heatLevel(item.count)],
                                topLeft = topLeft,
                                size = Size(cellPx, cellPx),
                                cornerRadius = radius
                            )
                            if (item.date == selectedDate) {
                                drawRoundRect(
                                    color = outline,
                                    topLeft = topLeft,
                                    size = Size(cellPx, cellPx),
                                    cornerRadius = radius,
                                    style = Stroke(width = 1.5.dp.toPx())
                                )
                            }
                        }
                    }
                }
            }
            Spacer(Modifier.height(8.dp))
            Text(
                text = if (selectedCell != null) heatmapDetail(selectedCell) else "Tap a day for details",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Less",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                levels.forEach { color ->
                    Box(
                        Modifier
                            .padding(start = 4.dp)
                            .size(10.dp)
                            .clip(RoundedCornerShape(2.dp))
                            .background(color)
                    )
                }
                Text(
                    text = "More",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(start = 6.dp)
                )
            }
        }
    }
}

private val DetailDateFormat: DateTimeFormatter =
    DateTimeFormatter.ofPattern("EEE, d MMM", Locale.getDefault())

private fun heatmapDetail(cell: HeatmapCell): String {
    val date = cell.date.format(DetailDateFormat)
    if (cell.count == 0) return "$date · no cards"
    val cards = if (cell.count == 1) "1 card" else cell.count.toString() + " cards"
    return date + " · " + cards + " · " + formatMinutes(cell.durationMs)
}

private val GridSteps = listOf(10f, 20f, 30f, 60f, 90f, 120f, 180f, 240f)

private fun niceTop(maxMinutes: Float): Float =
    GridSteps.firstOrNull { maxMinutes <= it } ?: (ceil(maxMinutes / 60f) * 60f)

@Composable
internal fun DailyTimeChart(daily: List<DayStat>) {
    if (daily.isEmpty()) return

    var selected by remember { mutableStateOf<Int?>(null) }
    val top = niceTop(daily.maxOf { it.minutes })
    val gridColor = QuickMLTheme.colors.chartGrid
    val barColor = QuickMLTheme.colors.accent
    val chartHeight = 140.dp
    val hasSelection = selected != null

    val totalMillis = daily.sumOf { it.millis }
    val best = daily.maxByOrNull { it.millis }
    val summary = "Revision time last " + daily.size + " days, total " + formatDuration(totalMillis) +
        if (best != null && best.millis > 0L) {
            ", best day " + best.shortName + " " + formatMinutes(best.millis)
        } else {
            ""
        }
    val selectedDay = selected?.let { daily.getOrNull(it) }

    Column {
        Text(
            text = if (selectedDay != null) {
                selectedDay.shortName + ": " + formatMinutes(selectedDay.millis)
            } else {
                "Tap a bar for details"
            },
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 12.dp)
                .semantics { contentDescription = summary }
        ) {
            Box(Modifier.width(36.dp).height(chartHeight)) {
                listOf(top, top / 2f).forEach { value ->
                    Text(
                        text = value.roundToInt().toString() + "m",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(end = 6.dp)
                            .offset(y = chartHeight * (1f - value / top) - 7.dp)
                    )
                }
            }
            Column(Modifier.weight(1f)) {
                Canvas(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(chartHeight)
                        .pointerInput(daily) {
                            detectTapGestures { offset ->
                                val columnWidth = size.width.toFloat() / daily.size
                                val index = (offset.x / columnWidth).toInt()
                                    .coerceIn(0, daily.size - 1)
                                selected = if (selected == index) null else index
                            }
                        }
                ) {
                    val columnWidth = size.width / daily.size
                    val barWidth = min(columnWidth * 0.5f, 14.dp.toPx())
                    val lineWidth = 1.dp.toPx()

                    // Recessive gridlines: baseline, half, top.
                    listOf(0f, 0.5f, 1f).forEach { fraction ->
                        val y = (size.height * (1f - fraction))
                            .coerceIn(lineWidth / 2f, size.height - lineWidth / 2f)
                        drawLine(
                            color = gridColor,
                            start = Offset(0f, y),
                            end = Offset(size.width, y),
                            strokeWidth = lineWidth
                        )
                    }

                    daily.forEachIndexed { index, day ->
                        if (day.millis > 0L) {
                            val barHeight = max(
                                size.height * (day.minutes / top).coerceIn(0f, 1f),
                                2.dp.toPx()
                            )
                            val left = index * columnWidth + (columnWidth - barWidth) / 2f
                            val radius = min(4.dp.toPx(), min(barWidth / 2f, barHeight))
                            val corner = CornerRadius(radius)
                            val path = Path().apply {
                                addRoundRect(
                                    RoundRect(
                                        left = left,
                                        top = size.height - barHeight,
                                        right = left + barWidth,
                                        bottom = size.height,
                                        topLeftCornerRadius = corner,
                                        topRightCornerRadius = corner,
                                        bottomRightCornerRadius = CornerRadius.Zero,
                                        bottomLeftCornerRadius = CornerRadius.Zero
                                    )
                                )
                            }
                            val dim = hasSelection && selected != index
                            drawPath(path, color = barColor.copy(alpha = if (dim) 0.4f else 1f))
                        }
                    }
                }
                Row(Modifier.fillMaxWidth().padding(top = 4.dp)) {
                    daily.forEach { day ->
                        Text(
                            text = day.initial,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
            }
        }
    }
}

private val CountSteps = listOf(4, 10, 20, 50, 100, 200, 500, 1000)

private fun niceCountTop(maxCount: Int): Int =
    CountSteps.firstOrNull { maxCount <= it } ?: (ceil(maxCount / 100f).toInt() * 100)

private fun reviewDetail(day: ReviewDayStat): String =
    day.shortName + ": " + Rating.values().joinToString(" · ") { rating ->
        day.counts[rating.ordinal].toString() + " " + rating.label
    }

/** Stacked bars per day: Forgot at the bottom, then Hard, Good, Easy, with a 2 dp gap between segments. */
@Composable
internal fun ReviewsChart(days: List<ReviewDayStat>) {
    if (days.isEmpty()) return

    var selected by remember { mutableStateOf<Int?>(null) }
    val ratings = Rating.values()
    val top = niceCountTop(days.maxOf { it.total })
    val gridColor = QuickMLTheme.colors.chartGrid
    val chartHeight = 140.dp
    val hasSelection = selected != null
    val theme = QuickMLTheme.colors
    val colors = ratings.map { theme.rating(it) }

    val totals = ratings.map { rating -> days.sumOf { it.counts[rating.ordinal] } }
    val summary = "Reviews last " + days.size + " days: " +
        ratings.joinToString(", ") { totals[it.ordinal].toString() + " " + it.label }
    val selectedDay = selected?.let { days.getOrNull(it) }

    Column {
        Text(
            text = if (selectedDay != null) reviewDetail(selectedDay) else "Tap a bar for details",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 12.dp)
                .semantics { contentDescription = summary }
        ) {
            Box(Modifier.width(36.dp).height(chartHeight)) {
                listOf(top, top / 2).forEach { value ->
                    Text(
                        text = value.toString(),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(end = 6.dp)
                            .offset(y = chartHeight * (1f - value.toFloat() / top) - 7.dp)
                    )
                }
            }
            Column(Modifier.weight(1f)) {
                Canvas(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(chartHeight)
                        .pointerInput(days) {
                            detectTapGestures { offset ->
                                val columnWidth = size.width.toFloat() / days.size
                                val index = (offset.x / columnWidth).toInt()
                                    .coerceIn(0, days.size - 1)
                                selected = if (selected == index) null else index
                            }
                        }
                ) {
                    val columnWidth = size.width / days.size
                    val barWidth = min(columnWidth * 0.5f, 14.dp.toPx())
                    val lineWidth = 1.dp.toPx()
                    val gap = 2.dp.toPx()
                    val minSegment = 2.dp.toPx()
                    val unit = size.height / top

                    listOf(0f, 0.5f, 1f).forEach { fraction ->
                        val y = (size.height * (1f - fraction))
                            .coerceIn(lineWidth / 2f, size.height - lineWidth / 2f)
                        drawLine(
                            color = gridColor,
                            start = Offset(0f, y),
                            end = Offset(size.width, y),
                            strokeWidth = lineWidth
                        )
                    }

                    days.forEachIndexed { index, day ->
                        if (day.total > 0) {
                            val left = index * columnWidth + (columnWidth - barWidth) / 2f
                            val dim = hasSelection && selected != index
                            val alpha = if (dim) 0.4f else 1f
                            val topmost = ratings.indices.last { day.counts[it] > 0 }
                            var bottom = size.height
                            ratings.indices.forEach { i ->
                                val count = day.counts[i]
                                if (count > 0) {
                                    val full = count * unit
                                    val color = colors[i].copy(alpha = alpha)
                                    if (i == topmost) {
                                        val h = max(full, minSegment)
                                        val radius = min(4.dp.toPx(), min(barWidth / 2f, h))
                                        val corner = CornerRadius(radius)
                                        val path = Path().apply {
                                            addRoundRect(
                                                RoundRect(
                                                    left = left,
                                                    top = bottom - h,
                                                    right = left + barWidth,
                                                    bottom = bottom,
                                                    topLeftCornerRadius = corner,
                                                    topRightCornerRadius = corner,
                                                    bottomRightCornerRadius = CornerRadius.Zero,
                                                    bottomLeftCornerRadius = CornerRadius.Zero
                                                )
                                            )
                                        }
                                        drawPath(path, color = color)
                                    } else {
                                        val h = max(full - gap, minSegment)
                                        drawRect(
                                            color = color,
                                            topLeft = Offset(left, bottom - h),
                                            size = Size(barWidth, h)
                                        )
                                        bottom -= h + gap
                                    }
                                }
                            }
                        }
                    }
                }
                Row(Modifier.fillMaxWidth().padding(top = 4.dp)) {
                    days.forEach { day ->
                        Text(
                            text = day.initial,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
            }
        }
        // Four series: the legend is always shown.
        Row(
            modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            ratings.forEach { rating ->
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        Modifier
                            .size(8.dp)
                            .clip(RoundedCornerShape(2.dp))
                            .background(theme.rating(rating))
                    )
                    Text(
                        text = rating.label,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(start = 6.dp)
                    )
                }
            }
        }
    }
}
