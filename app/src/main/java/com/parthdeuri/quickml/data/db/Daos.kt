package com.parthdeuri.quickml.data.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface ContentDao {
    @Query("SELECT * FROM problems")
    fun observeProblems(): Flow<List<ProblemEntity>>

    @Query("SELECT * FROM problems")
    suspend fun getProblems(): List<ProblemEntity>

    @Query("SELECT * FROM problems WHERE slug = :slug")
    suspend fun getProblem(slug: String): ProblemEntity?

    @Query("SELECT * FROM problems WHERE userAdded = 1")
    suspend fun getUserProblems(): List<ProblemEntity>

    @Query("SELECT * FROM sheets ORDER BY name")
    fun observeSheets(): Flow<List<SheetEntity>>

    @Query("SELECT * FROM sheet_entries ORDER BY sortOrder")
    fun observeEntries(): Flow<List<SheetEntryEntity>>

    @Query("SELECT sheetId FROM sheet_entries WHERE slug = :slug")
    suspend fun sheetIdsFor(slug: String): List<String>

    @Query("SELECT * FROM patterns ORDER BY sortOrder")
    fun observePatterns(): Flow<List<PatternEntity>>

    /** Bundled problems only (the user's own problems don't count as imported content). */
    @Query("SELECT COUNT(*) FROM problems WHERE userAdded IS NULL OR userAdded = 0")
    suspend fun countContentProblems(): Int

    @Query("SELECT COUNT(*) FROM patterns")
    suspend fun countPatterns(): Int

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertProblems(items: List<ProblemEntity>)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertProblem(item: ProblemEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertSheets(items: List<SheetEntity>)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertEntries(items: List<SheetEntryEntity>)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPatterns(items: List<PatternEntity>)

    /** Deletes bundled problems only: the user's own problems survive content updates. */
    @Query("DELETE FROM problems WHERE userAdded IS NULL OR userAdded = 0")
    suspend fun clearContentProblems()

    @Query("DELETE FROM problems WHERE slug = :slug AND userAdded = 1")
    suspend fun deleteUserProblem(slug: String)

    @Query("DELETE FROM sheets")
    suspend fun clearSheets()

    @Query("DELETE FROM sheet_entries")
    suspend fun clearEntries()

    @Query("DELETE FROM patterns")
    suspend fun clearPatterns()
}

@Dao
interface ProgressDao {
    @Query("SELECT * FROM user_progress")
    fun observeAll(): Flow<List<UserProgressEntity>>

    @Query("SELECT * FROM user_progress")
    suspend fun getAll(): List<UserProgressEntity>

    @Query("SELECT * FROM user_progress WHERE slug = :slug")
    suspend fun get(slug: String): UserProgressEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(item: UserProgressEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(items: List<UserProgressEntity>)

    @Query("DELETE FROM user_progress WHERE slug = :slug")
    suspend fun delete(slug: String)

    @Query("DELETE FROM user_progress")
    suspend fun clearAll()
}

@Dao
interface ActivityDao {
    @Insert
    suspend fun insert(item: ActivityLogEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(items: List<ActivityLogEntity>)

    @Query("SELECT * FROM activity_log WHERE timestamp >= :since ORDER BY timestamp")
    fun observeSince(since: Long): Flow<List<ActivityLogEntity>>

    @Query("SELECT * FROM activity_log WHERE timestamp >= :since ORDER BY timestamp")
    suspend fun getSince(since: Long): List<ActivityLogEntity>

    @Query("SELECT * FROM activity_log ORDER BY timestamp")
    suspend fun getAll(): List<ActivityLogEntity>

    @Query("DELETE FROM activity_log")
    suspend fun clearAll()
}

@Dao
interface ReviewDao {
    @Query("SELECT * FROM review_state")
    fun observeAll(): Flow<List<ReviewStateEntity>>

    @Query("SELECT * FROM review_state")
    suspend fun getAll(): List<ReviewStateEntity>

    @Query("SELECT * FROM review_state WHERE slug = :slug")
    suspend fun get(slug: String): ReviewStateEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(item: ReviewStateEntity)

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertAllIgnore(items: List<ReviewStateEntity>)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(items: List<ReviewStateEntity>)

    @Query("DELETE FROM review_state WHERE slug = :slug")
    suspend fun delete(slug: String)

    @Query("DELETE FROM review_state")
    suspend fun clearAll()

    @Query(
        "SELECT slug FROM user_progress WHERE status = 'SOLVED' " +
            "AND slug NOT IN (SELECT slug FROM review_state)"
    )
    suspend fun solvedSlugsWithoutReview(): List<String>

    @Query("SELECT COUNT(*) FROM review_state WHERE dueDay <= :day")
    suspend fun countDue(day: Long): Int
}
