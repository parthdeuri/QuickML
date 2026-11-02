package com.parthdeuri.quickml.widget

import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.view.View
import android.widget.RemoteViews
import com.parthdeuri.quickml.MainActivity
import com.parthdeuri.quickml.R
import com.parthdeuri.quickml.QuickMLApp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.launch
import java.time.LocalDate
import kotlin.random.Random

/** What the widget shows; [slug] is null when there's no problem to suggest. */
private class WidgetData(
    val due: Int,
    val slug: String?,
    val title: String?,
    val difficulty: String?
)

/**
 * Home-screen widget (classic RemoteViews): cards due today and a problem of the day. Updated
 * every 3 h by the system and after solve / rate / import / restore / the reminder worker.
 */
class QuickMLWidgetProvider : AppWidgetProvider() {

    override fun onUpdate(
        context: Context,
        appWidgetManager: AppWidgetManager,
        appWidgetIds: IntArray
    ) {
        if (appWidgetIds.isEmpty()) return
        val app = context.applicationContext
        // Room must not be queried on the main thread: finish asynchronously.
        val pending = goAsync()
        scope.launch {
            try {
                val data = try {
                    loadData(app)
                } catch (e: Exception) {
                    null
                }
                for (id in appWidgetIds) {
                    appWidgetManager.updateAppWidget(id, buildViews(app, data))
                }
            } catch (e: Exception) {
                // The widget host went away meanwhile; the next update tries again.
            } finally {
                pending.finish()
            }
        }
    }

    companion object {
        const val ACTION_OPEN_PROBLEM = "com.parthdeuri.quickml.action.OPEN_PROBLEM"
        const val EXTRA_OPEN_SLUG = "com.parthdeuri.quickml.extra.OPEN_SLUG"
        private const val REQUEST_OPEN = 2001

        private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

        /** Asks every placed widget to update (no-op when none is placed). */
        fun refresh(context: Context) {
            try {
                val app = context.applicationContext
                val manager = AppWidgetManager.getInstance(app) ?: return
                val ids = manager.getAppWidgetIds(
                    ComponentName(app, QuickMLWidgetProvider::class.java)
                )
                if (ids == null || ids.isEmpty()) return
                val intent = Intent(app, QuickMLWidgetProvider::class.java)
                    .setAction(AppWidgetManager.ACTION_APPWIDGET_UPDATE)
                    .putExtra(AppWidgetManager.EXTRA_APPWIDGET_IDS, ids)
                app.sendBroadcast(intent)
            } catch (e: Exception) {
                // No widget support on this device: nothing to refresh.
            }
        }

        /**
         * Problem of the day, deterministic by epoch day: among the problems due today; else the
         * unsolved full problems of the selected sheet; else any full problem.
         */
        private suspend fun loadData(context: Context): WidgetData {
            val container = (context.applicationContext as QuickMLApp).container
            val sheetId = container.studySettings.selectedSheetId.value
            val feed = container.repository.observeFeed(flowOf(sheetId)).first()
            val today = LocalDate.now().toEpochDay()
            val due = feed.allItems.filter { it.solved && (it.dueDay ?: Long.MAX_VALUE) <= today }
            val pool = due
                .ifEmpty { feed.items.filter { it.isFull && !it.solved } }
                .ifEmpty { feed.allItems.filter { it.isFull } }
                .sortedBy { it.problem.slug }
            val pick = if (pool.isEmpty()) null else pool[Random(today).nextInt(pool.size)]
            return WidgetData(
                due = due.size,
                slug = pick?.problem?.slug,
                title = pick?.problem?.title,
                difficulty = pick?.problem?.difficulty
            )
        }

        private fun buildViews(context: Context, data: WidgetData?): RemoteViews {
            val views = RemoteViews(context.packageName, R.layout.widget_quickml)
            val due = data?.due ?: 0
            views.setTextViewText(R.id.widget_due, "$due due today")
            val title = data?.title
            val difficulty = data?.difficulty
            for (id in DIFFICULTY_VIEWS) views.setViewVisibility(id, View.GONE)
            if (title != null) {
                views.setTextViewText(R.id.widget_label, "Card of the day")
                views.setTextViewText(R.id.widget_title, title)
                if (difficulty != null) {
                    // Colour comes from the view's resource, so it follows light / dark mode.
                    val id = difficultyView(difficulty)
                    views.setTextViewText(id, difficulty)
                    views.setViewVisibility(id, View.VISIBLE)
                }
            } else {
                views.setTextViewText(R.id.widget_label, "Card of the day")
                views.setTextViewText(R.id.widget_title, "Open QuickML to get started")
            }
            views.setOnClickPendingIntent(R.id.widget_root, openIntent(context, data?.slug))
            return views
        }

        private fun openIntent(context: Context, slug: String?): PendingIntent {
            val intent = Intent(context, MainActivity::class.java).apply {
                action = ACTION_OPEN_PROBLEM
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or
                    Intent.FLAG_ACTIVITY_CLEAR_TOP or
                    Intent.FLAG_ACTIVITY_SINGLE_TOP
                if (slug != null) putExtra(EXTRA_OPEN_SLUG, slug)
            }
            return PendingIntent.getActivity(
                context,
                REQUEST_OPEN,
                intent,
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
            )
        }

        private val DIFFICULTY_VIEWS = intArrayOf(
            R.id.widget_difficulty,
            R.id.widget_difficulty_easy,
            R.id.widget_difficulty_medium,
            R.id.widget_difficulty_hard
        )

        private fun difficultyView(difficulty: String): Int = when (difficulty) {
            "Easy" -> R.id.widget_difficulty_easy
            "Medium" -> R.id.widget_difficulty_medium
            "Hard" -> R.id.widget_difficulty_hard
            else -> R.id.widget_difficulty
        }
    }
}
