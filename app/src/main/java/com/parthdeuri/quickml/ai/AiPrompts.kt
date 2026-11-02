package com.parthdeuri.quickml.ai

import com.parthdeuri.quickml.data.repo.FeedItem

/** System instructions that ground the model in the card the user is looking at. */
object AiPrompts {

    private const val ROLE =
        "You are a patient machine-learning and deep-learning tutor inside a revision app. " +
            "The student is revising the concept below. Answer their questions concisely " +
            "(short paragraphs, at most about 250 words unless they ask for more), use Python (NumPy, scikit-learn or PyTorch) " +
            "for any code, and refer to line numbers of the shown code when useful. " +
            "Stay grounded in the concept and the reference notes; if you are unsure or the " +
            "notes disagree with you, say so instead of guessing."

    /** Builds the system instruction from a card; the user's notes are included on request. */
    fun systemFor(item: FeedItem, includeUserNotes: Boolean): String {
        val p = item.problem
        val examples = item.examples.mapIndexed { i, e ->
            val explanation =
                if (e.explanation.isBlank()) "" else "\n  Explanation: ${e.explanation}"
            "Q${i + 1}: ${e.input}\n  A: ${e.output}$explanation"
        }
        return systemFor(
            title = p.title,
            difficulty = p.difficulty,
            description = p.description,
            whenNotToUse = p.bruteForce,
            math = item.dryRun,
            examples = examples,
            constraints = item.constraints,
            approach = p.approach,
            intuition = p.intuition,
            whyItWorks = p.whyItWorks,
            solution = p.solution,
            time = p.time,
            space = p.space,
            mySolution = if (includeUserNotes) item.mySolution else null,
            note = if (includeUserNotes) item.note else null
        )
    }

    /** Plain-values variant (no UI or database types); blank / null parts are left out. */
    fun systemFor(
        title: String,
        difficulty: String,
        description: String?,
        examples: List<String> = emptyList(),
        constraints: List<String> = emptyList(),
        approach: List<String> = emptyList(),
        intuition: String? = null,
        whyItWorks: String? = null,
        solution: String? = null,
        time: String? = null,
        space: String? = null,
        mySolution: String? = null,
        note: String? = null,
        whenNotToUse: String? = null,
        math: List<String> = emptyList()
    ): String {
        val sb = StringBuilder(ROLE)
        sb.append("\n\n# Concept: ").append(title).append(" (").append(difficulty).append(")")
        section(sb, "When to use", description)
        if (examples.isNotEmpty()) {
            sb.append("\n\n## Interview questions")
            examples.forEach { sb.append('\n').append(it) }
        }
        if (constraints.isNotEmpty()) {
            sb.append("\n\n## Hyperparameters")
            constraints.forEach { sb.append("\n- ").append(it) }
        }
        section(sb, "When not to use", whenNotToUse)
        section(sb, "Intuition", intuition)
        if (math.isNotEmpty()) {
            sb.append("\n\n## Math")
            math.forEach { sb.append('\n').append(it) }
        }
        if (approach.isNotEmpty()) {
            sb.append("\n\n## How it works")
            approach.forEachIndexed { i, step ->
                sb.append('\n').append(i + 1).append(". ").append(step)
            }
        }
        section(sb, "Why it works", whyItWorks)
        if (!solution.isNullOrBlank()) {
            sb.append("\n\n## Reference code (Python, with line numbers)\n")
            sb.append(numbered(solution))
        }
        if (!time.isNullOrBlank() || !space.isNullOrBlank()) {
            sb.append("\n\n## Complexity")
            if (!time.isNullOrBlank()) sb.append("\nCompute: ").append(time)
            if (!space.isNullOrBlank()) sb.append("\nMemory: ").append(space)
        }
        if (!mySolution.isNullOrBlank()) {
            sb.append("\n\n## The student's own implementation (with line numbers)\n")
            sb.append(numbered(mySolution))
        }
        section(sb, "The student's own notes", note)
        return sb.toString()
    }

    private fun section(sb: StringBuilder, heading: String, text: String?) {
        if (text.isNullOrBlank()) return
        sb.append("\n\n## ").append(heading).append('\n').append(text.trim())
    }

    /** "1: first line" ... (1-based, same numbering as the Code tab). */
    private fun numbered(code: String): String =
        code.trimEnd().lines().mapIndexed { i, line -> "${i + 1}: $line" }.joinToString("\n")
}
