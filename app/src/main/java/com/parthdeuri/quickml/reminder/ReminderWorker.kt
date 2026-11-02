package com.parthdeuri.quickml.reminder

import android.Manifest
import android.annotation.SuppressLint
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.parthdeuri.quickml.MainActivity
import com.parthdeuri.quickml.R
import com.parthdeuri.quickml.QuickMLApp
import com.parthdeuri.quickml.data.repo.ProblemRepository
import com.parthdeuri.quickml.domain.currentStreak
import com.parthdeuri.quickml.widget.QuickMLWidgetProvider
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId

class ReminderWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        val container = (applicationContext as QuickMLApp).container
        // Daily run: also re-pick the widget's problem of the day / due count.
        QuickMLWidgetProvider.refresh(applicationContext)
        if (!container.reminderSettings.enabled) return Result.success()

        val zone = ZoneId.systemDefault()
        val today = LocalDate.now(zone)
        val repository = container.repository

        val due = repository.dueCount(today.toEpochDay())
        val since = today.minusDays(STREAK_WINDOW_DAYS).atStartOfDay(zone).toInstant().toEpochMilli()
        val viewDays = repository.activitySince(since)
            .filter { it.type == ProblemRepository.LOG_VIEW }
            .map { Instant.ofEpochMilli(it.timestamp).atZone(zone).toLocalDate() }
            .toSet()
        val viewedToday = viewDays.contains(today)
        val streak = currentStreak(viewDays, today)

        val message: String? = when {
            due == 1 -> "1 card due for revision"
            due > 1 -> due.toString() + " cards due for revision"
            !viewedToday && streak > 0 -> "Keep your " + streak + "-day streak going"
            !viewedToday -> "A few minutes of revision today?"
            else -> null
        }
        if (message != null) showNotification(message)
        return Result.success()
    }

    @SuppressLint("MissingPermission")
    private fun showNotification(message: String) {
        val context = applicationContext
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            return
        }
        val manager = NotificationManagerCompat.from(context)
        if (!manager.areNotificationsEnabled()) return

        val intent = Intent(context, MainActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
        val pending = PendingIntent.getActivity(
            context,
            0,
            intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        val notification = NotificationCompat.Builder(context, ReminderNotifications.CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_stat_quickml)
            .setContentTitle("QuickML")
            .setContentText(message)
            .setContentIntent(pending)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .build()
        try {
            manager.notify(NOTIFICATION_ID, notification)
        } catch (e: SecurityException) {
            // Permission revoked between the check and the call; nothing to do.
        }
    }

    private companion object {
        const val NOTIFICATION_ID = 1001
        const val STREAK_WINDOW_DAYS = 120L
    }
}
