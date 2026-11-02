package com.parthdeuri.quickml.data.db

import androidx.room.TypeConverter
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json

private val stringListSerializer = ListSerializer(String.serializer())

class Converters {
    @TypeConverter
    fun fromStringList(value: List<String>): String =
        Json.encodeToString(stringListSerializer, value)

    @TypeConverter
    fun toStringList(value: String): List<String> =
        Json.decodeFromString(stringListSerializer, value)
}
