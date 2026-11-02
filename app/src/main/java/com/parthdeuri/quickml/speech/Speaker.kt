package com.parthdeuri.quickml.speech

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.speech.tts.Voice
import android.widget.Toast
import androidx.compose.runtime.staticCompositionLocalOf
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.Locale

/** The activity's [Speaker]; null where none is provided (the Listen buttons then hide). */
val LocalSpeaker = staticCompositionLocalOf<Speaker?> { null }

/** A voice the engine offers: [name] is stored in settings, [label] is shown in the picker. */
data class VoiceOption(val name: String, val label: String, val requiresNetwork: Boolean)

/**
 * Read-aloud helper owned by MainActivity: one [TextToSpeech] instance, created on the first
 * [speak] and released in [shutdown]. Must be called on the main thread.
 */
class Speaker(context: Context) {
    private val appContext: Context = context.applicationContext
    private val mainHandler = Handler(Looper.getMainLooper())

    private var tts: TextToSpeech? = null
    private var engineState = STATE_NONE

    /** Request waiting for the engine to finish initialising. */
    private var pending: Pair<String, String>? = null

    /** Utterance id of the last chunk of the current request; it finishing ends the request. */
    private var lastChunkId: String? = null
    private var generation = 0

    /** Chosen voice (null = language default) and speed, from Settings via [configure]. */
    private var voiceName: String? = null
    private var speechRate = 1.0f

    /** Language the engine was set to on init; used when no (valid) voice is chosen. */
    private var defaultLocale: Locale = Locale.US

    private val voiceList = MutableStateFlow<List<VoiceOption>>(emptyList())

    /** English voices of the engine, sorted by name; empty until the engine is ready. */
    val voices: StateFlow<List<VoiceOption>> = voiceList.asStateFlow()

    private val speaking = MutableStateFlow<String?>(null)

    /** Id of the request being read (or about to be), null when silent. */
    val speakingId: StateFlow<String?> = speaking.asStateFlow()

    private val progressListener = object : UtteranceProgressListener() {
        override fun onStart(utteranceId: String?) = Unit

        override fun onDone(utteranceId: String?) {
            finished(utteranceId, lastOnly = true)
        }

        @Deprecated("Deprecated in Java")
        override fun onError(utteranceId: String?) {
            finished(utteranceId, lastOnly = false)
        }

        override fun onError(utteranceId: String?, errorCode: Int) {
            finished(utteranceId, lastOnly = false)
        }
    }

    /** Sets the voice (null = default) and speed used from the next [speak] on. */
    fun configure(voiceName: String?, speechRate: Float) {
        this.voiceName = voiceName
        this.speechRate = speechRate
        if (engineState == STATE_READY && speaking.value == null) applyVoiceSettings()
    }

    /** Starts the engine without speaking, so [voices] fills (no-op once started). */
    fun prepare() {
        if (engineState == STATE_NONE) initialise(null, null)
    }

    /** Reads a sample with the current voice and speed; a second call while it plays stops it. */
    fun preview(text: String) {
        if (speaking.value == PREVIEW_ID) stop() else speak(PREVIEW_ID, text)
    }

    /** Reads [text] aloud; [id] identifies the request (see [speakingId]). */
    fun speak(id: String, text: String) {
        when (engineState) {
            STATE_READY -> start(id, text)
            STATE_INITIALIZING -> {
                pending = id to text
                speaking.value = id
            }
            else -> initialise(id, text)
        }
    }

    /** Stops reading (no-op when silent). */
    fun stop() {
        pending = null
        lastChunkId = null
        if (engineState == STATE_READY) {
            try {
                tts?.stop()
            } catch (e: Exception) {
                // Engine died; nothing to stop.
            }
        }
        speaking.value = null
    }

    /** Stops reading if the current request's id starts with [prefix]. */
    fun stopIfSpeaking(prefix: String) {
        if (speaking.value?.startsWith(prefix) == true) stop()
    }

    /** Releases the engine; a later [speak] creates a new one. */
    fun shutdown() {
        stop()
        try {
            tts?.shutdown()
        } catch (e: Exception) {
            // Already gone.
        }
        tts = null
        engineState = STATE_NONE
    }

    private fun initialise(id: String?, text: String?) {
        engineState = STATE_INITIALIZING
        pending = if (id != null && text != null) id to text else null
        speaking.value = id
        val engine = try {
            // The callback may run synchronously (no engine) or on another thread: hop to main.
            TextToSpeech(appContext) { status -> mainHandler.post { onInit(status) } }
        } catch (e: Exception) {
            null
        }
        if (engine == null) {
            fail(silent = id == null)
        } else {
            tts = engine
        }
    }

    private fun onInit(status: Int) {
        val engine = tts
        if (engineState != STATE_INITIALIZING || engine == null) return
        val silent = pending == null
        if (status != TextToSpeech.SUCCESS) {
            fail(silent)
            return
        }
        val languageOk = try {
            val default = Locale.getDefault()
            val preferred = if (default.language == "en") default else Locale.US
            var result = engine.setLanguage(preferred)
            defaultLocale = preferred
            if (isBadLanguage(result) && preferred != Locale.US) {
                result = engine.setLanguage(Locale.US)
                defaultLocale = Locale.US
            }
            !isBadLanguage(result)
        } catch (e: Exception) {
            false
        }
        if (!languageOk) {
            fail(silent)
            return
        }
        engine.setOnUtteranceProgressListener(progressListener)
        engineState = STATE_READY
        loadVoices(engine)
        applyVoiceSettings()
        val request = pending
        pending = null
        if (request != null) {
            start(request.first, request.second)
        } else {
            speaking.value = null
        }
    }

    /** Publishes the engine's English voices, sorted by name. */
    private fun loadVoices(engine: TextToSpeech) {
        voiceList.value = try {
            (engine.voices ?: emptySet<Voice>())
                .filter { it.locale.language == "en" }
                .sortedBy { it.name }
                .map { voice ->
                    val place = voice.locale.getDisplayName(Locale.getDefault())
                    val online = voice.isNetworkConnectionRequired
                    VoiceOption(
                        name = voice.name,
                        label = place + " · " + voice.name + if (online) " (needs internet)" else "",
                        requiresNetwork = online
                    )
                }
        } catch (e: Exception) {
            emptyList()
        }
    }

    /** Applies speed and voice; a missing or unknown voice falls back to the language default. */
    private fun applyVoiceSettings() {
        val engine = tts ?: return
        try {
            engine.setSpeechRate(speechRate)
            val chosen = voiceName?.let { name -> engine.voices?.firstOrNull { it.name == name } }
            if (chosen != null) {
                engine.setVoice(chosen)
            } else {
                engine.setLanguage(defaultLocale)
            }
        } catch (e: Exception) {
            // Keep the engine's current voice.
        }
    }

    private fun isBadLanguage(result: Int): Boolean =
        result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED

    private fun start(id: String, text: String) {
        val engine = tts ?: return
        applyVoiceSettings()
        val maxLength = try {
            TextToSpeech.getMaxSpeechInputLength()
        } catch (e: Exception) {
            DEFAULT_MAX_INPUT
        }
        val chunks = splitForSpeech(text, (maxLength - 100).coerceAtLeast(500))
        if (chunks.isEmpty()) {
            stop()
            return
        }
        generation++
        val prefix = id + "#" + generation + "#"
        lastChunkId = prefix + chunks.lastIndex
        speaking.value = id
        var ok = true
        chunks.forEachIndexed { index, chunk ->
            val mode = if (index == 0) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
            val result = try {
                engine.speak(chunk, mode, null, prefix + index)
            } catch (e: Exception) {
                TextToSpeech.ERROR
            }
            if (result == TextToSpeech.ERROR) ok = false
        }
        if (!ok) {
            stop()
            toast("Couldn't read this aloud")
        }
    }

    /** Called on a binder thread when an utterance ends. */
    private fun finished(utteranceId: String?, lastOnly: Boolean) {
        if (utteranceId == null) return
        mainHandler.post {
            val last = lastChunkId ?: return@post
            val sameRequest = utteranceId.substringBeforeLast('#') == last.substringBeforeLast('#')
            if ((lastOnly && utteranceId == last) || (!lastOnly && sameRequest)) {
                lastChunkId = null
                speaking.value = null
            }
        }
    }

    private fun fail(silent: Boolean = false) {
        engineState = STATE_FAILED
        pending = null
        lastChunkId = null
        speaking.value = null
        try {
            tts?.shutdown()
        } catch (e: Exception) {
            // Ignore.
        }
        tts = null
        if (!silent) toast("Text-to-speech isn't available on this device")
    }

    private fun toast(message: String) {
        Toast.makeText(appContext, message, Toast.LENGTH_SHORT).show()
    }

    private companion object {
        const val STATE_NONE = 0
        const val STATE_INITIALIZING = 1
        const val STATE_READY = 2
        const val STATE_FAILED = 3
        const val DEFAULT_MAX_INPUT = 4000
        const val PREVIEW_ID = "voice-preview"
    }
}
