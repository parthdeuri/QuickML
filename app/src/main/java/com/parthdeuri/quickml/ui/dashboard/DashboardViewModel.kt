package com.parthdeuri.quickml.ui.dashboard

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.parthdeuri.quickml.AppContainer
import com.parthdeuri.quickml.data.db.ActivityLogEntity
import com.parthdeuri.quickml.data.repo.FeedData
import com.parthdeuri.quickml.data.repo.ConceptRepository
import com.parthdeuri.quickml.domain.srs.Rating
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.stateIn
import java.time.DayOfWeek
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.TextStyle as JavaTextStyle
import java.time.temporal.TemporalAdjusters
import java.util.Locale
import kotlin.math.roundToInt

const val HEATMAP_WEEKS = 16
const val DAILY_DAYS = 14
private const val LOG_WINDOW_DAYS = 120L

data class ProgressRow(val label: String, val solved: Int, val total: Int) {
    val fraction: Float
        get() = if (total == 0) 0f else solved.toFloat() / total
}

data class HeatmapCell(val date: LocalDate, val count: Int, val durationMs: Long)

data class MonthLabel(val week: Int, val text: String)

/** weeks[w][d]: column w (oldest first), row d (Monday = 0). Null = a day in the future. */
data class HeatmapData(
    val weeks: List<List<HeatmapCell?>> = emptyList(),
    val months: List<MonthLabel> = emptyList()
)

data class DayStat(
    val date: LocalDate,
    val initial: String,
    val shortName: String,
    val millis: Long
) {
    val minutes: Float
        get() = millis / 60_000f
}

/** Reviews of one day; [counts] is indexed by Rating.ordinal (Forgot, Hard, Good, Easy). */
data class ReviewDayStat(
    val date: LocalDate,
    val initial: String,
    val shortName: String,
    val counts: List<Int>
) {
    val total: Int
        get() = counts.sum()
}

data class DashboardState(
    val loading: Boolean = true,
    val hasActivity: Boolean = false,
    /** Name of the selected sheet (solved / difficulty / sections / focus use it). */
    val sheetName: String = "",
    val currentStreak: Int = 0,
    val bestStreak: Int = 0,
    val solved: Int = 0,
    val total: Int = 0,
    val fullCount: Int = 0,
    val weekMillis: Long = 0L,
    val cardsToday: Int = 0,
    val dueToday: Int = 0,
    /** Share of the last 30 days' reviews rated Good or Easy (0..100); null without reviews. */
    val recallPercent: Int? = null,
    val reviewCount30: Int = 0,
    /** Pattern-quiz answers of the last 30 days (all sheets). */
    val quizCorrect30: Int = 0,
    val quizTotal30: Int = 0,
    val reviewDaily: List<ReviewDayStat> = emptyList(),
    val byDifficulty: List<ProgressRow> = emptyList(),
    val sections: List<ProgressRow> = emptyList(),
    val focus: List<ProgressRow> = emptyList(),
    val heatmap: HeatmapData = HeatmapData(),
    val daily: List<DayStat> = emptyList()
)

@OptIn(ExperimentalCoroutinesApi::class)
class DashboardViewModel(container: AppContainer) : ViewModel() {
    private val repository = container.repository
    private val settings = container.studySettings
    private val refreshTick = MutableStateFlow(0)

    val state: StateFlow<DashboardState> = refreshTick
        .flatMapLatest {
            val zone = ZoneId.systemDefault()
            val since = LocalDate.now(zone)
                .minusDays(LOG_WINDOW_DAYS)
                .atStartOfDay(zone)
                .toInstant()
                .toEpochMilli()
            combine(
                repository.observeFeed(settings.selectedSheetId),
                repository.observeActivitySince(since)
            ) { feed, logs ->
                compute(feed, logs, System.currentTimeMillis())
            }
        }
        .flowOn(Dispatchers.Default)
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), DashboardState())

    /** Re-evaluates "today" (e.g. after midnight) when the dashboard is opened. */
    fun refresh() {
        refreshTick.value += 1
    }

    private fun compute(data: FeedData, logs: List<ActivityLogEntity>, now: Long): DashboardState {
        val zone = ZoneId.systemDefault()
        val locale = Locale.getDefault()
        val today = Instant.ofEpochMilli(now).atZone(zone).toLocalDate()

        // Per-day VIEW totals.
        val countByDay = HashMap<LocalDate, Int>()
        val millisByDay = HashMap<LocalDate, Long>()
        for (log in logs) {
            if (log.type != ConceptRepository.LOG_VIEW) continue
            val day = Instant.ofEpochMilli(log.timestamp).atZone(zone).toLocalDate()
            countByDay[day] = (countByDay[day] ?: 0) + 1
            millisByDay[day] = (millisByDay[day] ?: 0L) + log.durationMs
        }

        // REVIEW totals: per day and rating (14-day chart) and the 30-day recall rate.
        val ratings = Rating.values()
        val reviewsByDay = HashMap<LocalDate, IntArray>()
        val recallSince = today.minusDays(29)
        var recallTotal = 0
        var recallGood = 0
        for (log in logs) {
            if (log.type != ConceptRepository.LOG_REVIEW) continue
            val rating = ratings.firstOrNull { it.name == log.rating } ?: continue
            val day = Instant.ofEpochMilli(log.timestamp).atZone(zone).toLocalDate()
            reviewsByDay.getOrPut(day) { IntArray(ratings.size) }[rating.ordinal]++
            if (!day.isBefore(recallSince)) {
                recallTotal++
                if (rating == Rating.GOOD || rating == Rating.EASY) recallGood++
            }
        }

        // Pattern quiz: last 30 days, all sheets.
        var quizTotal = 0
        var quizCorrect = 0
        for (log in logs) {
            if (log.type != ConceptRepository.LOG_QUIZ) continue
            val day = Instant.ofEpochMilli(log.timestamp).atZone(zone).toLocalDate()
            if (day.isBefore(recallSince)) continue
            quizTotal++
            if (log.rating == ConceptRepository.QUIZ_CORRECT) quizCorrect++
        }

        // Streaks: current ends today (or yesterday if nothing logged yet today).
        var cursor = if (countByDay.containsKey(today)) today else today.minusDays(1)
        var currentStreak = 0
        while (countByDay.containsKey(cursor)) {
            currentStreak++
            cursor = cursor.minusDays(1)
        }
        var bestStreak = 0
        var run = 0
        var previous: LocalDate? = null
        for (day in countByDay.keys.sorted()) {
            val prev = previous
            run = if (prev != null && prev.plusDays(1) == day) run + 1 else 1
            if (run > bestStreak) bestStreak = run
            previous = day
        }

        val monday = today.with(TemporalAdjusters.previousOrSame(DayOfWeek.MONDAY))
        var weekMillis = 0L
        for ((day, ms) in millisByDay) {
            if (!day.isBefore(monday)) weekMillis += ms
        }

        // Heatmap: last HEATMAP_WEEKS weeks, Monday-first columns.
        val firstMonday = monday.minusWeeks(HEATMAP_WEEKS - 1L)
        val weeks = (0 until HEATMAP_WEEKS).map { w ->
            (0 until 7).map { d ->
                val date = firstMonday.plusDays(w * 7L + d)
                if (date.isAfter(today)) {
                    null
                } else {
                    HeatmapCell(date, countByDay[date] ?: 0, millisByDay[date] ?: 0L)
                }
            }
        }
        val candidates = ArrayList<MonthLabel>()
        var lastMonth = -1
        for (w in 0 until HEATMAP_WEEKS) {
            val columnMonday = firstMonday.plusWeeks(w.toLong())
            if (columnMonday.monthValue != lastMonth) {
                lastMonth = columnMonday.monthValue
                candidates.add(
                    MonthLabel(w, columnMonday.month.getDisplayName(JavaTextStyle.SHORT, locale))
                )
            }
        }
        // Drop a label that would sit too close to the next one.
        val months = candidates.filterIndexed { i, label ->
            i == candidates.lastIndex || candidates[i + 1].week - label.week >= 3
        }

        val daily = (DAILY_DAYS - 1 downTo 0).map { back ->
            val date = today.minusDays(back.toLong())
            DayStat(
                date = date,
                initial = date.dayOfWeek.getDisplayName(JavaTextStyle.NARROW, locale),
                shortName = date.dayOfWeek.getDisplayName(JavaTextStyle.SHORT, locale),
                millis = millisByDay[date] ?: 0L
            )
        }

        val reviewDaily = (DAILY_DAYS - 1 downTo 0).map { back ->
            val date = today.minusDays(back.toLong())
            ReviewDayStat(
                date = date,
                initial = date.dayOfWeek.getDisplayName(JavaTextStyle.NARROW, locale),
                shortName = date.dayOfWeek.getDisplayName(JavaTextStyle.SHORT, locale),
                counts = reviewsByDay[date]?.toList() ?: List(ratings.size) { 0 }
            )
        }

        // Progress rows: the selected sheet. Due counts are global (Revise is cross-sheet).
        val items = data.items
        val todayEpoch = today.toEpochDay()
        val byDifficulty = listOf("Easy", "Medium", "Hard").map { difficulty ->
            val group = items.filter { it.problem.difficulty == difficulty }
            ProgressRow(difficulty, group.count { it.solved }, group.size)
        }
        val sections = items.groupBy { it.section }.map { (name, group) ->
            ProgressRow(name, group.count { it.solved }, group.size)
        }
        val focus = items
            .flatMap { item -> item.problem.topics.map { topic -> topic to item } }
            .groupBy({ it.first }, { it.second })
            .filter { it.value.size >= 3 }
            .map { (topic, group) -> ProgressRow(topic, group.count { it.solved }, group.size) }
            .sortedWith(compareBy<ProgressRow>({ it.fraction }, { -it.total }, { it.label }))
            .take(6)

        return DashboardState(
            loading = false,
            hasActivity = countByDay.isNotEmpty(),
            sheetName = data.sheetName,
            currentStreak = currentStreak,
            bestStreak = bestStreak,
            solved = items.count { it.solved },
            total = items.size,
            fullCount = items.count { it.isFull },
            weekMillis = weekMillis,
            cardsToday = countByDay[today] ?: 0,
            dueToday = data.allItems.count {
                it.solved && (it.dueDay ?: Long.MAX_VALUE) <= todayEpoch
            },
            recallPercent = if (recallTotal == 0) null else (recallGood * 100f / recallTotal).roundToInt(),
            reviewCount30 = recallTotal,
            quizCorrect30 = quizCorrect,
            quizTotal30 = quizTotal,
            reviewDaily = reviewDaily,
            byDifficulty = byDifficulty,
            sections = sections,
            focus = focus,
            heatmap = HeatmapData(weeks, months),
            daily = daily
        )
    }
}

class DashboardViewModelFactory(private val container: AppContainer) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T =
        DashboardViewModel(container) as T
}
