package com.parthdeuri.quickml.data.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface ChatDao {
    @Query("SELECT * FROM chat_messages WHERE slug = :slug ORDER BY id")
    fun observeMessages(slug: String): Flow<List<ChatEntity>>

    @Insert
    suspend fun insert(message: ChatEntity): Long

    @Query("DELETE FROM chat_messages WHERE slug = :slug")
    suspend fun deleteForSlug(slug: String)

    @Query("DELETE FROM chat_messages")
    suspend fun deleteAll()
}
