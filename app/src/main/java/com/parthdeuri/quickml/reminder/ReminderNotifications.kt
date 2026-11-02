package com.parthdeuri.quickml.reminder

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context

object ReminderNotifications {
    const val CHANNEL_ID = "reminders"

    /** Safe to call on every start; minSdk is 26 so channels always exist. */
    fun createChannel(context: Context) {
        val manager = context.getSystemService(NotificationManager::class.java) ?: return
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Daily reminder",
            NotificationManager.IMPORTANCE_DEFAULT
        )
        channel.description = "A nudge to revise your solved problems"
        manager.createNotificationChannel(channel)
    }
}
