package com.parthdeuri.quickml.ai

import kotlinx.coroutines.flow.Flow

/** One message of a conversation; [role] is [ROLE_USER] or [ROLE_MODEL]. */
data class ChatTurn(val role: String, val text: String) {
    companion object {
        const val ROLE_USER = "user"
        const val ROLE_MODEL = "model"
    }
}

/** A chat backend (Gemini today; another provider or an on-device model can implement this). */
interface AiProvider {
    /**
     * Streams the answer to the last turn of [history] as text deltas. The flow fails with an
     * [AiError] (never another exception type) and stops the request when the collector is
     * cancelled.
     */
    fun ask(system: String, history: List<ChatTurn>): Flow<String>
}
