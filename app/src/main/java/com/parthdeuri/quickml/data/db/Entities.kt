package com.parthdeuri.quickml.data.db

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Content table: replaced when contentVersion increases, except the user's own problems
 * ([userAdded] = true), which content imports never delete.
 */
@Entity(tableName = "problems")
data class ConceptEntity(
    @PrimaryKey val slug: String,
    val title: String,
    val platform: String,
    val url: String,
    val difficulty: String,
    val topics: List<String>,
    val patterns: List<String>,
    val summary: String?,
    val intuition: String?,
    /** Empty list means "no approach" (metadata-only problem). */
    val approach: List<String>,
    val time: String?,
    val space: String?,
    val solution: String?,
    val edgeCases: List<String>,
    val related: List<String>,
    val description: String? = null,
    /** Raw JSON (List<ExampleJson>); decoded in the repository layer. */
    val examplesJson: String? = null,
    /** Raw JSON (List<String>); decoded in the repository layer. */
    val constraintsJson: String? = null,
    val bruteForce: String? = null,
    val whyItWorks: String? = null,
    /** Raw JSON (List<String>): trace of the solution on example 1. */
    val dryRunJson: String? = null,
    /** Raw JSON (List<Int>): 1-based key line numbers in [solution]; null when empty. */
    val keyLinesJson: String? = null,
    /** True for problems the user added ("My problems"); null / false for bundled content. */
    val userAdded: Boolean? = null,
    /** Raw JSON (List<String>): box-art diagram lines; null when the card has none. */
    val diagramJson: String? = null
)

/** Content table: the pattern library (replaced on import like the other content tables). */
@Entity(tableName = "patterns")
data class PatternEntity(
    @PrimaryKey val id: String,
    val name: String,
    val summary: String,
    /** Raw JSON (List<String>). */
    val whenToUseJson: String,
    val keyIdea: String,
    val template: String,
    val complexity: String,
    /** Raw JSON (List<String>). */
    val pitfallsJson: String,
    /** Raw JSON (List<String>): linked problem slugs, best first. */
    val problemsJson: String,
    /** Raw JSON (List<String>): pattern strings used in problems' `patterns`. */
    val aliasesJson: String,
    /** Index in the content's "patterns" array (teaching order). */
    val sortOrder: Int
)

/** Content table. */
@Entity(tableName = "sheets")
data class SheetEntity(
    @PrimaryKey val id: String,
    val name: String,
    val description: String
)

/** Content table. */
@Entity(
    tableName = "sheet_entries",
    primaryKeys = ["sheetId", "slug"],
    indices = [Index(value = ["slug"])]
)
data class SheetEntryEntity(
    val sheetId: String,
    val slug: String,
    val section: String,
    @ColumnInfo(name = "sortOrder") val order: Int
)

/** User table: never touched by content import. */
@Entity(tableName = "user_progress")
data class UserProgressEntity(
    @PrimaryKey val slug: String,
    val status: String = "UNSOLVED",
    val bookmarked: Boolean = false,
    val note: String? = null,
    val updatedAt: Long = 0L,
    val solvedAt: Long? = null,
    /** The user's own solution code (Code tab, "Mine"). */
    val mySolution: String? = null
)

/**
 * User table: one row per card visit ("VIEW"), solve event ("SOLVED"), rating ("REVIEW") or
 * pattern-quiz answer ("QUIZ").
 */
@Entity(
    tableName = "activity_log",
    indices = [Index(value = ["timestamp"])]
)
data class ActivityLogEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val slug: String,
    val type: String,
    val timestamp: Long,
    val durationMs: Long,
    /** Rating name for "REVIEW" rows, "CORRECT"/"WRONG" for "QUIZ" rows, null otherwise. */
    val rating: String? = null
)

/** User table: SM-2 scheduling state, one row per solved problem. */
@Entity(tableName = "review_state")
data class ReviewStateEntity(
    @PrimaryKey val slug: String,
    val ease: Double,
    val intervalDays: Int,
    val reps: Int,
    val lapses: Int,
    /** Local epoch day (LocalDate.toEpochDay()). */
    val dueDay: Long,
    val lastRating: String? = null,
    val lastReviewedAt: Long? = null
)

/** User table: one chat message of the "Ask" tab ([role] is "user" or "model"). */
@Entity(
    tableName = "chat_messages",
    indices = [Index(value = ["slug"])]
)
data class ChatEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val slug: String,
    val role: String,
    val text: String,
    val timestamp: Long
)
