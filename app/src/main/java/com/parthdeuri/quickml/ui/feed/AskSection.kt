package com.parthdeuri.quickml.ui.feed

import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.scrollBy
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.parthdeuri.quickml.QuickMLApp
import com.parthdeuri.quickml.ai.AiPrompts
import com.parthdeuri.quickml.ai.ChatTurn
import com.parthdeuri.quickml.data.db.ChatEntity
import com.parthdeuri.quickml.data.repo.FeedItem
import com.parthdeuri.quickml.ui.theme.QuickMLTheme

private val SUGGESTIONS = listOf(
    "Why this approach?",
    "What if I used a different data structure?",
    "Explain the key lines",
    "What's the worst case?",
    "Common mistakes?"
)

/** The "Ask" tab: chat with Gemini about [item]; only shown while the AI toggle is on. */
@Composable
fun AskSection(item: FeedItem, onOpenSettings: () -> Unit) {
    val slug = item.problem.slug
    val container = (LocalContext.current.applicationContext as QuickMLApp).container
    val hasKey by container.aiSettings.hasKey.collectAsState()
    val vm: AskViewModel = viewModel(factory = AskViewModelFactory(container))
    val chat by vm.chat.collectAsState()
    val ui by vm.ui.collectAsState()
    val system = remember(item) { AiPrompts.systemFor(item, includeUserNotes = true) }

    LaunchedEffect(slug) { vm.open(slug) }

    if (!hasKey) {
        SetupCard(onOpenSettings)
        return
    }

    // The view model may still show the previous problem's chat for a frame.
    val ready = chat.slug == slug
    val messages = if (ready) chat.messages else emptyList()
    val sending = ready && ui.slug == slug && ui.sending
    val streaming = if (sending) ui.streaming else ""
    val error = if (ready && ui.slug == slug) ui.error else null

    var input by remember(slug) { mutableStateOf("") }
    var menuOpen by remember { mutableStateOf(false) }
    val listState = rememberLazyListState()

    // Newest message (or the growing answer) stays in view.
    val itemCount = messages.size + (if (sending) 1 else 0) + (if (error != null) 1 else 0)
    LaunchedEffect(itemCount, streaming.length / 60) {
        if (itemCount > 0) {
            listState.scrollToItem(itemCount - 1)
            // Past the end is clamped: lands on the bottom of a tall last bubble.
            listState.scrollBy(100_000f)
        }
    }

    fun submit(text: String) {
        if (text.isBlank() || sending) return
        vm.send(slug, system, text)
        input = ""
    }

    Column(modifier = Modifier.fillMaxSize().imePadding()) {
        Row(
            modifier = Modifier.padding(start = 16.dp, end = 4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "AI can be wrong — check it against the notes",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.weight(1f)
            )
            Box {
                IconButton(onClick = { menuOpen = true }, enabled = messages.isNotEmpty()) {
                    Icon(Icons.Filled.MoreVert, contentDescription = "Chat menu")
                }
                DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                    DropdownMenuItem(
                        text = { Text("Clear chat") },
                        onClick = {
                            menuOpen = false
                            vm.clear(slug)
                        }
                    )
                }
            }
        }

        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(10.dp),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp)
        ) {
            if (ready && messages.isEmpty() && !sending && error == null) {
                item(key = "suggestions") { Suggestions(onPick = { submit(it) }) }
            }
            items(messages, key = { it.id }) { message -> MessageBubble(message) }
            if (sending) {
                item(key = "streaming") { StreamingBubble(streaming) }
            }
            if (error != null) {
                item(key = "error") {
                    ErrorRow(
                        message = error,
                        canRetry = messages.lastOrNull()?.role == ChatTurn.ROLE_USER,
                        onRetry = { vm.retry(slug, system) }
                    )
                }
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 12.dp, end = 4.dp, bottom = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedTextField(
                value = input,
                onValueChange = { input = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Ask about this problem") },
                maxLines = 4,
                shape = RoundedCornerShape(20.dp)
            )
            if (sending) {
                IconButton(onClick = { vm.stop() }) {
                    Icon(
                        Icons.Filled.Stop,
                        contentDescription = "Stop",
                        tint = QuickMLTheme.colors.accent
                    )
                }
            } else {
                IconButton(onClick = { submit(input) }, enabled = input.isNotBlank()) {
                    Icon(
                        Icons.AutoMirrored.Filled.Send,
                        contentDescription = "Send",
                        tint = if (input.isNotBlank()) {
                            QuickMLTheme.colors.accent
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        }
                    )
                }
            }
        }
    }
}

@Composable
private fun SetupCard(onOpenSettings: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Add your free Gemini key in Settings",
            style = MaterialTheme.typography.titleMedium
        )
        Text(
            text = "Then ask questions about any problem right here.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 6.dp, bottom = 16.dp)
        )
        Button(onClick = onOpenSettings) { Text("Open Settings") }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun Suggestions(onPick: (String) -> Unit) {
    Column {
        Text(
            text = "Ask anything about this problem",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(bottom = 8.dp)
        )
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            SUGGESTIONS.forEach { suggestion ->
                AssistChip(onClick = { onPick(suggestion) }, label = { Text(suggestion) })
            }
        }
    }
}

@Composable
private fun MessageBubble(message: ChatEntity) {
    val isUser = message.role == ChatTurn.ROLE_USER
    Bubble(isUser = isUser) {
        if (isUser) {
            Text(text = message.text, style = MaterialTheme.typography.bodyMedium)
        } else {
            MarkdownText(message.text)
        }
    }
}

@Composable
private fun StreamingBubble(text: String) {
    Bubble(isUser = false) {
        if (text.isNotEmpty()) MarkdownText(text)
        Text(
            text = "…",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun Bubble(isUser: Boolean, content: @Composable () -> Unit) {
    val colors = QuickMLTheme.colors
    val width = if (isUser) Modifier.widthIn(max = 300.dp) else Modifier.fillMaxWidth(0.95f)
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        Column(
            modifier = width
                .background(
                    color = if (isUser) {
                        colors.accent.copy(alpha = 0.18f)
                    } else {
                        MaterialTheme.colorScheme.surfaceVariant
                    },
                    shape = RoundedCornerShape(
                        topStart = 16.dp,
                        topEnd = 16.dp,
                        bottomStart = if (isUser) 16.dp else 4.dp,
                        bottomEnd = if (isUser) 4.dp else 16.dp
                    )
                )
                .padding(horizontal = 12.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            content()
        }
    }
}

@Composable
private fun ErrorRow(message: String, canRetry: Boolean, onRetry: () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = message,
            style = MaterialTheme.typography.bodySmall,
            color = QuickMLTheme.colors.danger,
            modifier = Modifier.weight(1f)
        )
        if (canRetry) TextButton(onClick = onRetry) { Text("Retry") }
    }
}

// ---- Markdown ----------------------------------------------------------------------------

private sealed interface MdBlock {
    class Para(val text: String) : MdBlock
    class Code(val code: String) : MdBlock
}

/** Paragraphs and fenced code blocks (an unclosed fence, while streaming, ends the text). */
private fun parseBlocks(src: String): List<MdBlock> {
    val blocks = mutableListOf<MdBlock>()
    val para = mutableListOf<String>()
    val code = mutableListOf<String>()
    var inCode = false
    fun flushPara() {
        if (para.isNotEmpty()) blocks.add(MdBlock.Para(para.joinToString("\n")))
        para.clear()
    }
    for (raw in src.lines()) {
        if (raw.trimStart().startsWith("```")) {
            if (inCode) {
                blocks.add(MdBlock.Code(code.joinToString("\n")))
                code.clear()
            } else {
                flushPara()
            }
            inCode = !inCode
        } else if (inCode) {
            code.add(raw)
        } else if (raw.isBlank()) {
            flushPara()
        } else {
            val trimmed = raw.trimStart()
            val line = when {
                trimmed.startsWith("- ") || trimmed.startsWith("* ") ->
                    "• " + trimmed.substring(2)
                trimmed.startsWith("#") -> "**" + trimmed.trimStart('#').trim() + "**"
                else -> raw
            }
            para.add(line)
        }
    }
    if (inCode && code.isNotEmpty()) blocks.add(MdBlock.Code(code.joinToString("\n")))
    flushPara()
    return blocks
}

/** `**bold**` and backtick-quoted inline code spans. */
private fun inlineMarkdown(src: String, codeBackground: Color): AnnotatedString =
    buildAnnotatedString {
        var i = 0
        while (i < src.length) {
            if (src.startsWith("**", i)) {
                val end = src.indexOf("**", i + 2)
                if (end > i + 2) {
                    withStyle(SpanStyle(fontWeight = FontWeight.Bold)) {
                        append(src.substring(i + 2, end))
                    }
                    i = end + 2
                    continue
                }
            }
            if (src[i] == '`') {
                val end = src.indexOf('`', i + 1)
                if (end > i + 1) {
                    withStyle(
                        SpanStyle(fontFamily = FontFamily.Monospace, background = codeBackground)
                    ) {
                        append(src.substring(i + 1, end))
                    }
                    i = end + 1
                    continue
                }
            }
            append(src[i])
            i++
        }
    }

@Composable
private fun MarkdownText(text: String) {
    val palette = QuickMLTheme.colors.code
    val blocks = remember(text) { parseBlocks(text) }
    val codeStyle = remember {
        TextStyle(fontFamily = FontFamily.Monospace, fontSize = 12.sp, lineHeight = 18.sp)
    }
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        blocks.forEach { block ->
            when (block) {
                is MdBlock.Para -> {
                    val styled = remember(block, palette) {
                        inlineMarkdown(block.text, palette.background)
                    }
                    Text(text = styled, style = MaterialTheme.typography.bodyMedium)
                }
                is MdBlock.Code -> {
                    val highlighted = remember(block, palette) {
                        PythonHighlighter.highlight(block.code, palette)
                    }
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(palette.background, RoundedCornerShape(10.dp))
                            .horizontalScroll(rememberScrollState())
                            .padding(10.dp)
                    ) {
                        Text(text = highlighted, style = codeStyle, color = palette.text)
                    }
                }
            }
        }
    }
}
