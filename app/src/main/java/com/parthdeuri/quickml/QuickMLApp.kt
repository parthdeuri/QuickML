package com.parthdeuri.quickml

import android.app.Application
import com.parthdeuri.quickml.reminder.ReminderNotifications
import com.parthdeuri.quickml.reminder.ReminderScheduler

class QuickMLApp : Application() {
    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
        ReminderNotifications.createChannel(this)
        // Re-anchor the periodic work to the chosen time (it drifts a little between runs).
        val reminder = container.reminderSettings
        if (reminder.enabled) {
            ReminderScheduler.schedule(this, reminder.hour, reminder.minute)
        }
    }
}
