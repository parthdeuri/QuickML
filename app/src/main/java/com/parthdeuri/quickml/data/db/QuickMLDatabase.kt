package com.parthdeuri.quickml.data.db

import androidx.room.Database
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(
    entities = [
        ConceptEntity::class,
        SheetEntity::class,
        SheetEntryEntity::class,
        UserProgressEntity::class,
        ActivityLogEntity::class,
        ReviewStateEntity::class,
        PatternEntity::class,
        ChatEntity::class,
        FlashcardEntity::class,
        FlashReviewEntity::class
    ],
    version = 8,
    exportSchema = false
)
@TypeConverters(Converters::class)
abstract class QuickMLDatabase : RoomDatabase() {
    abstract fun contentDao(): ContentDao
    abstract fun progressDao(): ProgressDao
    abstract fun activityDao(): ActivityDao
    abstract fun reviewDao(): ReviewDao
    abstract fun chatDao(): ChatDao
    abstract fun flashDao(): FlashDao
}

/**
 * v7 -> v8: flash cards. Two new tables, SQL identical to what Room generates for
 * FlashcardEntity and FlashReviewEntity. Existing tables and rows are never touched; the
 * flash cards are imported from flashcards.json afterwards.
 */
val MIGRATION_7_8: Migration = object : Migration(7, 8) {
    override fun migrate(db: SupportSQLiteDatabase) {
        db.execSQL(
            "CREATE TABLE IF NOT EXISTS `flashcards` (" +
                "`id` TEXT NOT NULL, " +
                "`slug` TEXT NOT NULL, " +
                "`kind` TEXT NOT NULL, " +
                "`front` TEXT NOT NULL, " +
                "`back` TEXT NOT NULL, " +
                "`decks` TEXT NOT NULL, " +
                "`sortOrder` INTEGER NOT NULL, " +
                "PRIMARY KEY(`id`))"
        )
        db.execSQL(
            "CREATE TABLE IF NOT EXISTS `flash_review` (" +
                "`cardId` TEXT NOT NULL, " +
                "`ease` REAL NOT NULL, " +
                "`intervalDays` INTEGER NOT NULL, " +
                "`reps` INTEGER NOT NULL, " +
                "`lapses` INTEGER NOT NULL, " +
                "`dueDay` INTEGER NOT NULL, " +
                "`lastRating` TEXT, " +
                "`lastReviewedAt` INTEGER, " +
                "`firstSeenDay` INTEGER NOT NULL, " +
                "PRIMARY KEY(`cardId`))"
        )
    }
}

/**
 * v6 -> v7: optional box-art diagram per card. One nullable TEXT column, exactly what Room
 * generates for ConceptEntity.diagramJson. Content is re-imported afterwards.
 */
val MIGRATION_6_7: Migration = object : Migration(6, 7) {
    override fun migrate(db: SupportSQLiteDatabase) {
        db.execSQL("ALTER TABLE problems ADD COLUMN diagramJson TEXT")
    }
}

/**
 * v5 -> v6: "Ask" chat history. New chat_messages table + slug index, SQL identical to what Room
 * generates for ChatEntity. Existing tables are never touched.
 */
val MIGRATION_5_6: Migration = object : Migration(5, 6) {
    override fun migrate(db: SupportSQLiteDatabase) {
        db.execSQL(
            "CREATE TABLE IF NOT EXISTS `chat_messages` (" +
                "`id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, " +
                "`slug` TEXT NOT NULL, " +
                "`role` TEXT NOT NULL, " +
                "`text` TEXT NOT NULL, " +
                "`timestamp` INTEGER NOT NULL)"
        )
        db.execSQL(
            "CREATE INDEX IF NOT EXISTS `index_chat_messages_slug` " +
                "ON `chat_messages` (`slug`)"
        )
    }
}

/**
 * v4 -> v5: notes / own solution / own problems / pattern library. Nullable columns without
 * defaults and a patterns table, exactly what Room generates for UserProgressEntity.mySolution,
 * ConceptEntity.userAdded and PatternEntity. Existing rows are never touched (content is
 * re-imported afterwards; user tables keep everything).
 */
val MIGRATION_4_5: Migration = object : Migration(4, 5) {
    override fun migrate(db: SupportSQLiteDatabase) {
        db.execSQL("ALTER TABLE user_progress ADD COLUMN mySolution TEXT")
        db.execSQL("ALTER TABLE problems ADD COLUMN userAdded INTEGER")
        db.execSQL(
            "CREATE TABLE IF NOT EXISTS `patterns` (" +
                "`id` TEXT NOT NULL, " +
                "`name` TEXT NOT NULL, " +
                "`summary` TEXT NOT NULL, " +
                "`whenToUseJson` TEXT NOT NULL, " +
                "`keyIdea` TEXT NOT NULL, " +
                "`template` TEXT NOT NULL, " +
                "`complexity` TEXT NOT NULL, " +
                "`pitfallsJson` TEXT NOT NULL, " +
                "`problemsJson` TEXT NOT NULL, " +
                "`aliasesJson` TEXT NOT NULL, " +
                "`sortOrder` INTEGER NOT NULL, " +
                "PRIMARY KEY(`id`))"
        )
    }
}

/**
 * v3 -> v4: key lines of the solution (Practice mode). Nullable TEXT without a default, exactly
 * what Room generates for ConceptEntity.keyLinesJson. Content is re-imported afterwards (the
 * bundled contentVersion is newer); user tables are never touched.
 */
val MIGRATION_3_4: Migration = object : Migration(3, 4) {
    override fun migrate(db: SupportSQLiteDatabase) {
        db.execSQL("ALTER TABLE problems ADD COLUMN keyLinesJson TEXT")
    }
}

/**
 * v2 -> v3: spaced repetition. New review_state table (SQL identical to what Room generates for
 * ReviewStateEntity) and the rating column on activity_log. Existing rows are never touched.
 */
val MIGRATION_2_3: Migration = object : Migration(2, 3) {
    override fun migrate(db: SupportSQLiteDatabase) {
        db.execSQL(
            "CREATE TABLE IF NOT EXISTS `review_state` (" +
                "`slug` TEXT NOT NULL, " +
                "`ease` REAL NOT NULL, " +
                "`intervalDays` INTEGER NOT NULL, " +
                "`reps` INTEGER NOT NULL, " +
                "`lapses` INTEGER NOT NULL, " +
                "`dueDay` INTEGER NOT NULL, " +
                "`lastRating` TEXT, " +
                "`lastReviewedAt` INTEGER, " +
                "PRIMARY KEY(`slug`))"
        )
        db.execSQL("ALTER TABLE activity_log ADD COLUMN rating TEXT")
        db.execSQL("ALTER TABLE problems ADD COLUMN bruteForce TEXT")
        db.execSQL("ALTER TABLE problems ADD COLUMN whyItWorks TEXT")
        db.execSQL("ALTER TABLE problems ADD COLUMN dryRunJson TEXT")
    }
}

/**
 * v1 -> v2: problem descriptions/examples/constraints, solved timestamps and the activity log.
 * Must match the entities exactly; never touches existing user_progress rows.
 */
val MIGRATION_1_2: Migration = object : Migration(1, 2) {
    override fun migrate(db: SupportSQLiteDatabase) {
        db.execSQL("ALTER TABLE problems ADD COLUMN description TEXT")
        db.execSQL("ALTER TABLE problems ADD COLUMN examplesJson TEXT")
        db.execSQL("ALTER TABLE problems ADD COLUMN constraintsJson TEXT")
        db.execSQL("ALTER TABLE user_progress ADD COLUMN solvedAt INTEGER")
        db.execSQL(
            "CREATE TABLE IF NOT EXISTS `activity_log` (" +
                "`id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, " +
                "`slug` TEXT NOT NULL, " +
                "`type` TEXT NOT NULL, " +
                "`timestamp` INTEGER NOT NULL, " +
                "`durationMs` INTEGER NOT NULL)"
        )
        db.execSQL(
            "CREATE INDEX IF NOT EXISTS `index_activity_log_timestamp` " +
                "ON `activity_log` (`timestamp`)"
        )
    }
}
