package com.parthdeuri.quickml.ui.feed

import android.os.SystemClock
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.parthdeuri.quickml.AppContainer
import com.parthdeuri.quickml.ai.AiError
import com.parthdeuri.quickml.ai.ChatTurn
import com.parthdeuri.quickml.data.db.ChatEntity
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

/** Stored messages of the chat of [slug] (null: no chat opened yet). */
data class AskChat(val slug: String?, val messages: List<ChatEntity>)

/** Transient state of the request of [slug]: streaming answer, or the error to retry. */
data class AskUiState(
    val slug: String? = null,
    val sending: Boolean = false,
    /** Answer received so far (shown live, saved when complete). */
    val streaming: String = "",
    val error: String? = null
)

/**
 * Chats of the "Ask" tab. Activity-scoped; one request at a time across all problems. The AI
 * provider is only created while the AI toggle is on.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class AskViewModel(private val container: AppContainer) : ViewModel() {
    private val slugFlow = MutableStateFlow<String?>(null)

    val chat: StateFlow<AskChat> = slugFlow
        .flatMapLatest { slug ->
            if (slug == null) {
                flowOf(AskChat(null, emptyList()))
            } else {
                container.chatDao.observeMessages(slug).map { AskChat(slug, it) }
            }
        }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), AskChat(null, emptyList()))

    private val uiFlow = MutableStateFlow(AskUiState())
    val ui: StateFlow<AskUiState> = uiFlow.asStateFlow()

    private var job: Job? = null
    private var lastStartMs = 0L

    /** Shows the chat of [slug]; another problem's running request is cancelled. */
    fun open(slug: String) {
        if (slugFlow.value == slug) return
        job?.cancel()
        job = null
        uiFlow.value = AskUiState()
        slugFlow.value = slug
    }

    /** Sends [text] as the next question of the chat of [slug]. */
    fun send(slug: String, system: String, text: String) {
        val clean = text.trim()
        if (clean.isEmpty()) return
        start(slug, system, clean)
    }

    /** Asks again for the last stored question (no duplicate message) after an error. */
    fun retry(slug: String, system: String) {
        start(slug, system, null)
    }

    /** Stops the running request; the part of the answer received so far is kept. */
    fun stop() {
        val state = uiFlow.value
        val slug = state.slug
        val partial = state.streaming.trim()
        job?.cancel()
        job = null
        uiFlow.value = AskUiState(slug = slug)
        if (state.sending && slug != null && partial.isNotEmpty()) {
            viewModelScope.launch { insert(slug, ChatTurn.ROLE_MODEL, partial) }
        }
    }

    fun clear(slug: String) {
        if (uiFlow.value.slug == slug) {
            job?.cancel()
            job = null
            uiFlow.value = AskUiState()
        }
        viewModelScope.launch { container.chatDao.deleteForSlug(slug) }
    }

    private suspend fun insert(slug: String, role: String, text: String) {
        container.chatDao.insert(
            ChatEntity(
                slug = slug,
                role = role,
                text = text,
                timestamp = System.currentTimeMillis()
            )
        )
    }

    /** [newQuestion] null = retry the last stored question. */
    private fun start(slug: String, system: String, newQuestion: String?) {
        // The toggle gates everything: no provider (and so no network) while it is off.
        if (!container.aiSettings.enabled.value) return
        if (uiFlow.value.sending) return
        uiFlow.value = AskUiState(slug = slug, sending = true)
        job = viewModelScope.launch {
            try {
                if (newQuestion != null) insert(slug, ChatTurn.ROLE_USER, newQuestion)
                val history = container.chatDao.observeMessages(slug).first()
                    .map { ChatTurn(it.role, it.text) }
                if (history.lastOrNull()?.role != ChatTurn.ROLE_USER) {
                    uiFlow.value = AskUiState(slug = slug)
                    return@launch
                }
                val wait = MIN_GAP_MS - (SystemClock.elapsedRealtime() - lastStartMs)
                if (wait > 0) delay(wait)
                lastStartMs = SystemClock.elapsedRealtime()

                val partial = StringBuilder()
                container.createAiProvider().ask(system, history).collect { delta ->
                    partial.append(delta)
                    uiFlow.value = uiFlow.value.copy(streaming = partial.toString())
                }
                val answer = partial.toString().trim()
                if (answer.isEmpty()) throw AiError.Blocked
                insert(slug, ChatTurn.ROLE_MODEL, answer)
                uiFlow.value = AskUiState(slug = slug)
            } catch (e: CancellationException) {
                throw e
            } catch (e: AiError) {
                uiFlow.value = AskUiState(slug = slug, error = e.message ?: AiError.Unknown.message)
            } catch (e: Exception) {
                uiFlow.value = AskUiState(slug = slug, error = AiError.Unknown.message)
            }
        }
    }

    private companion object {
        const val MIN_GAP_MS = 1_500L
    }
}

class AskViewModelFactory(private val container: AppContainer) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T = AskViewModel(container) as T
}
