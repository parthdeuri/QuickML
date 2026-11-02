package com.parthdeuri.quickml.data.content

import android.content.Context
import android.content.SharedPreferences
import androidx.room.withTransaction
import com.parthdeuri.quickml.data.db.PatternEntity
import com.parthdeuri.quickml.data.db.ConceptEntity
import com.parthdeuri.quickml.data.db.QuickMLDatabase
import com.parthdeuri.quickml.data.db.SheetEntity
import com.parthdeuri.quickml.data.db.SheetEntryEntity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray

class ContentImporter(
    private val context: Context,
    private val db: QuickMLDatabase,
    private val prefs: SharedPreferences
) {
    private val json = Json {
        ignoreUnknownKeys = true
        coerceInputValues = true
    }

    /**
     * Replaces all content tables (never the user tables, never the user's own problems) when
     * the bundled contentVersion is newer than the stored one.
     */
    suspend fun importIfNeeded() {
        withContext(Dispatchers.IO) {
            val text = context.assets.open(ASSET_NAME).bufferedReader().use { it.readText() }
            val file = json.decodeFromString(ContentFile.serializer(), text)
            val patterns = decodePatterns(file)
            val stored = prefs.getInt(KEY_VERSION, 0)
            val contentDao = db.contentDao()
            if (stored >= file.contentVersion &&
                contentDao.countContentConcepts() > 0 &&
                (patterns.isEmpty() || contentDao.countPatterns() > 0)
            ) {
                return@withContext
            }

            val problems = file.problems.map { p ->
                ConceptEntity(
                    slug = p.slug,
                    title = p.title,
                    platform = p.platform,
                    url = p.url,
                    difficulty = p.difficulty,
                    topics = p.topics,
                    patterns = p.patterns,
                    summary = p.summary,
                    intuition = p.intuition,
                    approach = p.approach ?: emptyList(),
                    time = p.time,
                    space = p.space,
                    solution = p.solution,
                    edgeCases = p.edgeCases,
                    related = p.related,
                    description = p.description,
                    examplesJson = ContentJson.encodeExamples(p.examples),
                    constraintsJson = ContentJson.encodeConstraints(p.constraints),
                    bruteForce = p.bruteForce,
                    whyItWorks = p.whyItWorks,
                    dryRunJson = ContentJson.encodeStrings(p.dryRun),
                    keyLinesJson = ContentJson.encodeInts(p.keyLines),
                    diagramJson = ContentJson.encodeStrings(p.diagram)
                )
            }
            val sheets = file.sheets.map { SheetEntity(it.id, it.name, it.description) }
            val entries = file.sheets.flatMap { sheet ->
                sheet.entries.map { e -> SheetEntryEntity(sheet.id, e.slug, e.section, e.order) }
            }

            db.withTransaction {
                contentDao.clearEntries()
                contentDao.clearSheets()
                // The user's own problems (userAdded = 1) survive content updates.
                contentDao.clearContentConcepts()
                contentDao.clearPatterns()
                contentDao.insertConcepts(problems)
                contentDao.insertSheets(sheets)
                contentDao.insertEntries(entries)
                contentDao.insertPatterns(patterns)
            }
            prefs.edit().putInt(KEY_VERSION, file.contentVersion).apply()
        }
    }

    /** The pattern library; a missing list gives none, a malformed item is skipped. */
    private fun decodePatterns(file: ContentFile): List<PatternEntity> {
        val array = file.patterns as? JsonArray ?: return emptyList()
        return array.mapIndexedNotNull { index, element ->
            val p = try {
                json.decodeFromJsonElement(PatternJson.serializer(), element)
            } catch (e: Exception) {
                null
            }
            if (p == null || p.id.isBlank()) {
                null
            } else {
                PatternEntity(
                    id = p.id,
                    name = p.name.ifBlank { p.id },
                    summary = p.summary,
                    whenToUseJson = ContentJson.encodeStringList(p.whenToUse),
                    keyIdea = p.keyIdea,
                    template = p.templateText,
                    complexity = p.complexity,
                    pitfallsJson = ContentJson.encodeStringList(p.pitfalls),
                    problemsJson = ContentJson.encodeStringList(p.problems),
                    aliasesJson = ContentJson.encodeStringList(p.aliases),
                    sortOrder = index
                )
            }
        }.distinctBy { it.id }
    }

    companion object {
        private const val ASSET_NAME = "problems.json"
        private const val KEY_VERSION = "content_version"
    }
}
