package com.parthdeuri.quickml

import android.content.Context
import android.content.SharedPreferences
import androidx.room.Room
import com.parthdeuri.quickml.ai.AiProvider
import com.parthdeuri.quickml.ai.AiSettings
import com.parthdeuri.quickml.ai.GeminiProvider
import com.parthdeuri.quickml.data.backup.BackupManager
import com.parthdeuri.quickml.data.content.ContentImporter
import com.parthdeuri.quickml.data.db.ChatDao
import com.parthdeuri.quickml.data.db.MIGRATION_1_2
import com.parthdeuri.quickml.data.db.MIGRATION_2_3
import com.parthdeuri.quickml.data.db.MIGRATION_3_4
import com.parthdeuri.quickml.data.db.MIGRATION_4_5
import com.parthdeuri.quickml.data.db.MIGRATION_5_6
import com.parthdeuri.quickml.data.db.MIGRATION_6_7
import com.parthdeuri.quickml.data.db.QuickMLDatabase
import com.parthdeuri.quickml.data.repo.ConceptRepository
import com.parthdeuri.quickml.data.settings.StudySettings
import com.parthdeuri.quickml.domain.srs.ReviewScheduler
import com.parthdeuri.quickml.domain.srs.Sm2Scheduler
import com.parthdeuri.quickml.reminder.ReminderSettings
import com.parthdeuri.quickml.widget.QuickMLWidgetProvider

/** Manual dependency container (no Hilt). */
class AppContainer(context: Context) {
    private val appContext: Context = context.applicationContext

    val prefs: SharedPreferences =
        appContext.getSharedPreferences("quickml_prefs", Context.MODE_PRIVATE)

    val database: QuickMLDatabase =
        Room.databaseBuilder(appContext, QuickMLDatabase::class.java, "quickml.db")
            .addMigrations(
                MIGRATION_1_2, MIGRATION_2_3, MIGRATION_3_4, MIGRATION_4_5, MIGRATION_5_6,
                MIGRATION_6_7
            )
            .build()

    /** "Ask" chat history (user table). */
    val chatDao: ChatDao
        get() = database.chatDao()

    val importer: ContentImporter = ContentImporter(appContext, database, prefs)

    val scheduler: ReviewScheduler = Sm2Scheduler

    /** Solving, rating, imports and own-problem edits refresh the home-screen widget. */
    val repository: ConceptRepository = ConceptRepository(database, scheduler) {
        QuickMLWidgetProvider.refresh(appContext)
    }

    val reminderSettings: ReminderSettings = ReminderSettings(prefs)

    /** Selected sheet + card preferences; observed by the Feed, Sheet and Dashboard. */
    val studySettings: StudySettings = StudySettings(prefs)

    /** AI toggle (default off), model and the encrypted Gemini key; not part of the backup. */
    val aiSettings: AiSettings = AiSettings(prefs)

    /** Creates the Gemini client; call only while [AiSettings.enabled] is on (it is cheap). */
    fun createAiProvider(): AiProvider = GeminiProvider(aiSettings)

    /** Settings -> back up / restore progress. */
    val backupManager: BackupManager = BackupManager(
        context = appContext,
        db = database,
        studySettings = studySettings,
        reminderSettings = reminderSettings,
        onRestored = { QuickMLWidgetProvider.refresh(appContext) }
    )
}
