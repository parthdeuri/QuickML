package com.parthdeuri.quickml

import android.content.Intent
import android.content.res.Configuration
import android.graphics.Color
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.filled.BarChart
import androidx.compose.material.icons.filled.Style
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import com.parthdeuri.quickml.data.settings.ThemeMode
import com.parthdeuri.quickml.speech.LocalSpeaker
import com.parthdeuri.quickml.speech.Speaker
import com.parthdeuri.quickml.ui.dashboard.DashboardScreen
import com.parthdeuri.quickml.ui.dashboard.DashboardViewModel
import com.parthdeuri.quickml.ui.dashboard.DashboardViewModelFactory
import com.parthdeuri.quickml.ui.feed.FeedScreen
import com.parthdeuri.quickml.ui.feed.FeedViewModel
import com.parthdeuri.quickml.ui.feed.FeedViewModelFactory
import com.parthdeuri.quickml.ui.settings.SettingsScreen
import com.parthdeuri.quickml.ui.sheet.SheetScreen
import com.parthdeuri.quickml.ui.sheet.SheetViewModel
import com.parthdeuri.quickml.ui.sheet.SheetViewModelFactory
import com.parthdeuri.quickml.ui.theme.QuickMLTheme
import com.parthdeuri.quickml.widget.QuickMLWidgetProvider

private const val TAB_FEED = 0
private const val TAB_SHEET = 1
private const val TAB_DASHBOARD = 2

/** Whether [mode] resolves to the dark theme, given the system's current setting. */
private fun isDarkTheme(mode: ThemeMode, systemDark: Boolean): Boolean = when (mode) {
    ThemeMode.SYSTEM -> systemDark
    ThemeMode.LIGHT -> false
    ThemeMode.DARK -> true
}

class MainActivity : ComponentActivity() {
    /** Read-aloud helper; the TTS engine inside is created on first use. */
    private lateinit var speaker: Speaker

    /** Concept to open from a widget tap (consumed by the composition). */
    private val pendingOpenSlug = mutableStateOf<String?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        speaker = Speaker(this)
        // Not on recreation (rotation): the original intent was handled already.
        if (savedInstanceState == null) handleOpenIntent(intent)
        val container = (application as QuickMLApp).container
        val study = container.studySettings
        val systemDark = (resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK) ==
            Configuration.UI_MODE_NIGHT_YES
        applySystemBars(isDarkTheme(study.themeMode.value, systemDark))
        speaker.configure(study.voiceName.value, study.speechRate.value)
        setContent {
            val themeMode by study.themeMode.collectAsState()
            val accent by study.accent.collectAsState()
            val voiceName by study.voiceName.collectAsState()
            val speechRate by study.speechRate.collectAsState()
            // Voice and speed changes reach the engine before the next Listen.
            LaunchedEffect(voiceName, speechRate) { speaker.configure(voiceName, speechRate) }
            val darkTheme = isDarkTheme(themeMode, isSystemInDarkTheme())
            // Status / navigation bar icons follow the theme (dark icons on light surfaces).
            DisposableEffect(darkTheme) {
                applySystemBars(darkTheme)
                onDispose { }
            }
            CompositionLocalProvider(LocalSpeaker provides speaker) {
                QuickMLTheme(darkTheme = darkTheme, accent = accent) {
                    Surface(
                        modifier = Modifier.fillMaxSize(),
                        color = MaterialTheme.colorScheme.background
                    ) {
                        // Activity-scoped: stays alive while other tabs are shown.
                        val feedViewModel: FeedViewModel =
                            viewModel(factory = FeedViewModelFactory(container))
                        var tab by rememberSaveable { mutableIntStateOf(TAB_FEED) }
                        // Settings is a state-driven screen on top of the Dashboard tab.
                        var showSettings by rememberSaveable { mutableStateOf(false) }

                        BackHandler(enabled = showSettings) { showSettings = false }

                        fun selectTab(target: Int) {
                            if (target == TAB_FEED && tab != TAB_FEED) feedViewModel.onFeedOpened()
                            showSettings = false
                            tab = target
                        }

                        // Widget tap: the Feed at that problem (switching sheet if needed).
                        val openSlug = pendingOpenSlug.value
                        LaunchedEffect(openSlug) {
                            if (openSlug != null) {
                                pendingOpenSlug.value = null
                                feedViewModel.openConceptAnywhere(openSlug)
                                showSettings = false
                                tab = TAB_FEED
                            }
                        }

                        Scaffold(
                            containerColor = MaterialTheme.colorScheme.background,
                            contentWindowInsets = WindowInsets(0, 0, 0, 0),
                            bottomBar = {
                                NavigationBar(containerColor = MaterialTheme.colorScheme.surfaceContainer) {
                                    val accent = QuickMLTheme.colors.accent
                                    val itemColors = NavigationBarItemDefaults.colors(
                                        selectedIconColor = accent,
                                        selectedTextColor = accent,
                                        indicatorColor = MaterialTheme.colorScheme.primaryContainer,
                                        unselectedIconColor = MaterialTheme.colorScheme.onSurfaceVariant,
                                        unselectedTextColor = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                    NavigationBarItem(
                                        selected = tab == TAB_FEED,
                                        onClick = { selectTab(TAB_FEED) },
                                        icon = { Icon(Icons.Filled.Style, contentDescription = null) },
                                        label = { Text("Feed") },
                                        colors = itemColors
                                    )
                                    NavigationBarItem(
                                        selected = tab == TAB_SHEET,
                                        onClick = { selectTab(TAB_SHEET) },
                                        icon = {
                                            Icon(Icons.AutoMirrored.Filled.List, contentDescription = null)
                                        },
                                        label = { Text("Sheet") },
                                        colors = itemColors
                                    )
                                    NavigationBarItem(
                                        selected = tab == TAB_DASHBOARD,
                                        onClick = { selectTab(TAB_DASHBOARD) },
                                        icon = { Icon(Icons.Filled.BarChart, contentDescription = null) },
                                        label = { Text("Dashboard") },
                                        colors = itemColors
                                    )
                                }
                            }
                        ) { innerPadding ->
                            Box(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .padding(innerPadding)
                                    .consumeWindowInsets(innerPadding)
                            ) {
                                when (tab) {
                                    TAB_FEED -> FeedScreen(
                                        viewModel = feedViewModel,
                                        onOpenSettings = {
                                            showSettings = true
                                            tab = TAB_DASHBOARD
                                        }
                                    )

                                    TAB_SHEET -> {
                                        val sheetViewModel: SheetViewModel =
                                            viewModel(factory = SheetViewModelFactory(container))
                                        SheetScreen(
                                            viewModel = sheetViewModel,
                                            onOpenConcept = { slug ->
                                                // Learn mode, filters cleared, scrolled to the problem.
                                                feedViewModel.openConcept(slug)
                                                tab = TAB_FEED
                                            }
                                        )
                                    }

                                    else -> {
                                        if (showSettings) {
                                            SettingsScreen(
                                                container = container,
                                                onBack = { showSettings = false }
                                            )
                                        } else {
                                            val dashboardViewModel: DashboardViewModel =
                                                viewModel(factory = DashboardViewModelFactory(container))
                                            DashboardScreen(
                                                viewModel = dashboardViewModel,
                                                onOpenSettings = { showSettings = true }
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /** Edge-to-edge with transparent bars; light styles give dark icons on the light theme. */
    private fun applySystemBars(darkTheme: Boolean) {
        if (darkTheme) {
            enableEdgeToEdge(
                statusBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
                navigationBarStyle = SystemBarStyle.dark(Color.TRANSPARENT)
            )
        } else {
            enableEdgeToEdge(
                statusBarStyle = SystemBarStyle.light(Color.TRANSPARENT, Color.TRANSPARENT),
                navigationBarStyle = SystemBarStyle.light(Color.TRANSPARENT, Color.TRANSPARENT)
            )
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleOpenIntent(intent)
    }

    private fun handleOpenIntent(intent: Intent?) {
        if (intent == null || intent.action != QuickMLWidgetProvider.ACTION_OPEN_PROBLEM) return
        val slug = intent.getStringExtra(QuickMLWidgetProvider.EXTRA_OPEN_SLUG)
        if (!slug.isNullOrBlank()) pendingOpenSlug.value = slug
    }

    override fun onStop() {
        // App in the background: stop reading aloud.
        if (::speaker.isInitialized) speaker.stop()
        super.onStop()
    }

    override fun onDestroy() {
        if (::speaker.isInitialized) speaker.shutdown()
        super.onDestroy()
    }
}
