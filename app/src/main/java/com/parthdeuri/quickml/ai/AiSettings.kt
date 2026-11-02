package com.parthdeuri.quickml.ai

import android.content.SharedPreferences
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * AI settings: master toggle (default off), model name and the Gemini API key. The key is
 * encrypted with an Android Keystore AES-GCM key (alias [KEY_ALIAS]); only ciphertext + IV
 * (Base64) are stored in [prefs]. The key is never logged and is not part of the backup.
 */
class AiSettings(private val prefs: SharedPreferences) {
    private val enabledFlow = MutableStateFlow(prefs.getBoolean(KEY_ENABLED, false))

    /** Master switch: while false the app shows no AI UI and makes no network request. */
    val enabled: StateFlow<Boolean> = enabledFlow.asStateFlow()

    private val modelFlow = MutableStateFlow(
        prefs.getString(KEY_MODEL, null)?.trim()?.takeIf { it.isNotEmpty() } ?: DEFAULT_MODEL
    )

    /** Model name used for requests: one of [MODEL_PRESETS] or free text. */
    val model: StateFlow<String> = modelFlow.asStateFlow()

    private val hasKeyFlow = MutableStateFlow(
        prefs.getString(KEY_CIPHER, null) != null && prefs.getString(KEY_IV, null) != null
    )

    /** True when a key is saved (it may still be rejected or undecryptable later). */
    val hasKey: StateFlow<Boolean> = hasKeyFlow.asStateFlow()

    fun setEnabled(on: Boolean) {
        if (on == enabledFlow.value) return
        prefs.edit().putBoolean(KEY_ENABLED, on).apply()
        enabledFlow.value = on
    }

    /** Blank resets to [DEFAULT_MODEL]. */
    fun setModel(name: String) {
        val clean = name.trim().ifEmpty { DEFAULT_MODEL }
        if (clean == modelFlow.value) return
        prefs.edit().putString(KEY_MODEL, clean).apply()
        modelFlow.value = clean
    }

    /** Encrypts and stores [key] (trimmed); blank clears. False if encryption fails. */
    fun setKey(key: String): Boolean {
        val clean = key.trim()
        if (clean.isEmpty()) {
            clearKey()
            return true
        }
        return try {
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(Cipher.ENCRYPT_MODE, secretKey())
            val encrypted = cipher.doFinal(clean.toByteArray(Charsets.UTF_8))
            prefs.edit()
                .putString(KEY_CIPHER, Base64.encodeToString(encrypted, Base64.NO_WRAP))
                .putString(KEY_IV, Base64.encodeToString(cipher.iv, Base64.NO_WRAP))
                .apply()
            hasKeyFlow.value = true
            true
        } catch (e: Exception) {
            false
        }
    }

    fun clearKey() {
        prefs.edit().remove(KEY_CIPHER).remove(KEY_IV).apply()
        hasKeyFlow.value = false
    }

    /** The decrypted key, or null when none is saved or it can no longer be decrypted. */
    fun getKey(): String? {
        val cipherText = prefs.getString(KEY_CIPHER, null) ?: return null
        val iv = prefs.getString(KEY_IV, null) ?: return null
        return try {
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(
                Cipher.DECRYPT_MODE,
                secretKey(),
                GCMParameterSpec(GCM_TAG_BITS, Base64.decode(iv, Base64.NO_WRAP))
            )
            val plain = cipher.doFinal(Base64.decode(cipherText, Base64.NO_WRAP))
            String(plain, Charsets.UTF_8).takeIf { it.isNotEmpty() }
        } catch (e: Exception) {
            // Keystore entry lost (restore on a new device, wiped lock screen): the key is gone.
            clearKey()
            null
        }
    }

    private fun secretKey(): SecretKey {
        val store = KeyStore.getInstance(ANDROID_KEYSTORE)
        store.load(null)
        (store.getKey(KEY_ALIAS, null) as? SecretKey)?.let { return it }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, ANDROID_KEYSTORE)
        generator.init(
            KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build()
        )
        return generator.generateKey()
    }

    companion object {
        const val DEFAULT_MODEL = "gemini-3.5-flash-lite"

        /** Models offered in Settings; any other name is a custom one. */
        val MODEL_PRESETS = listOf(
            DEFAULT_MODEL,
            "gemini-3.1-flash-lite",
            "gemini-3.8-flash",
            "gemini-3.6-flash"
        )

        private const val KEY_ALIAS = "quickml_ai_key"
        private const val ANDROID_KEYSTORE = "AndroidKeyStore"
        private const val TRANSFORMATION = "AES/GCM/NoPadding"
        private const val GCM_TAG_BITS = 128

        private const val KEY_ENABLED = "ai_enabled"
        private const val KEY_MODEL = "ai_model"
        private const val KEY_CIPHER = "ai_key_cipher"
        private const val KEY_IV = "ai_key_iv"
    }
}
