package me.stedra.stegocam

import android.Manifest
import android.app.Activity
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.SurfaceTexture
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.Gravity
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.ViewGroup
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.CheckBox
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import android.content.SharedPreferences
import android.widget.Button

class MainActivity : Activity(), CameraController.Listener {

    companion object {
        const val TAG = "StegoCam"
        const val ACTION_DUMP = "me.stedra.stegocam.DUMP"
        const val ACTION_LIVE = "me.stedra.stegocam.LIVE"
        const val ACTION_DECODE_FILE = "me.stedra.stegocam.DECODE_FILE"
        const val ACTION_GATHER = "me.stedra.stegocam.GATHER"
        const val ACTION_POSE_TEST = "me.stedra.stegocam.POSE_TEST"
        const val ACTION_GATHER_TEST = "me.stedra.stegocam.GATHER_TEST"
        const val ACTION_SCANNER = "me.stedra.stegocam.SCANNER"
        private const val REQ_CAMERA = 1
    }

    private lateinit var textureView: AutoFitTextureView
    private lateinit var guide: GuideView
    private lateinit var liveOverlay: LiveOverlayView
    private lateinit var liveBox: CheckBox
    private lateinit var info: TextView
    private lateinit var legend: TextView
    private lateinit var summaryView: TextView
    private lateinit var procButton: Button
    private lateinit var camera: CameraController
    private lateinit var gather: GatherController
    private lateinit var scanner: ScannerController
    private lateinit var panel: LinearLayout
    private lateinit var prefs: SharedPreferences
    private var sensorOrientation = 90
    private var receiverRegistered = false
    private lateinit var statusBox: LinearLayout
    private fun statusBoxShow(v: Boolean) { statusBox.visibility = if (v) View.VISIBLE else View.GONE }
    private fun enterScanner() { panel.visibility = View.GONE; statusBoxShow(false); guide.visibility = View.GONE; liveBox.isChecked = false; liveOverlay.showBoxes(emptyList(), emptyList()) }
    private fun enterGather() { panel.visibility = View.GONE; statusBoxShow(false); guide.visibility = View.GONE }

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(c: Context, i: Intent) {
            if (i.action == ACTION_LIVE) {
                val on = i.getBooleanExtra("on", true)
                if (i.hasExtra("pin")) LiveDecoder.pin = i.getBooleanExtra("pin", true)
                if (i.hasExtra("fast")) LiveDecoder.fastBig = i.getBooleanExtra("fast", true)
                if (i.hasExtra("multi")) LiveDecoder.multi = i.getBooleanExtra("multi", true)
                if (i.hasExtra("decode")) camera.decodeOn = i.getBooleanExtra("decode", true)
                Log.i(TAG, "broadcast LIVE on=$on")
                runOnUiThread { liveBox.isChecked = on }
                return
            }
            if (i.action == ACTION_POSE_TEST) { PoseEstimator.testFile(applicationContext, i.getStringExtra("path"), i.getIntExtra("page", 1), i.getStringExtra("k"), i.getStringExtra("hint")); return }
            if (i.action == ACTION_GATHER_TEST) { i.getStringExtra("file")?.let { gather.testFile(it) }; return }
            if (i.action == ACTION_GATHER) {
                // --ez on true|false, --ez resume true (continue latest session), --ei idx N (start at scenario N), --es set demo|sheet4
                val on = i.getBooleanExtra("on", true)
                runOnUiThread {
                    if (!on) { if (gather.active) gather.stop() }
                    else if (!gather.active) {
                        val res = if (i.getBooleanExtra("resume", false)) Session.latest(applicationContext) else null
                        enterGather(); gather.start(res, i.getIntExtra("idx", -1), i.getStringExtra("set") ?: "sheet4")
                    }
                }
                return
            }
            if (i.action == ACTION_SCANNER) {
                // --ez on true|false, --es src both|qr|stego, --ez rescan true
                val on = i.getBooleanExtra("on", true)
                runOnUiThread {
                    if (!on) { if (scanner.active) scanner.stop() }
                    else if (!scanner.active) { enterScanner(); scanner.start(i.getStringExtra("src") ?: "both", i.getStringExtra("profile") ?: "demo") }
                    else if (i.getBooleanExtra("rescan", false)) scanner.rescan()
                    if (on && i.hasExtra("file")) scanner.replay(i.getStringExtra("file")!!, i.getStringExtra("file2"))
                    if (on && i.getBooleanExtra("reveal", false)) scanner.toggleReveal()
                }
                return
            }
            if (i.action == ACTION_DECODE_FILE) { DecodeCli.run(applicationContext, i.getStringExtra("path"), i.getStringExtra("boxes"), i.getStringExtra("opts")); return }
            val n = i.getIntExtra("n", 30)
            val tag = i.getStringExtra("tag")
            val proc = i.getStringExtra("proc")
            Log.i(TAG, "broadcast DUMP n=$n tag=$tag proc=$proc")
            camera.requestDump(n, tag, proc)
            if (proc == "off" || proc == "default") runOnUiThread { showProc(proc) }
        }
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun showProc(p: String) { procButton.text = "proc: $p" }

    /** Intent extras: cam=<id|auto>, proc=off|default. Persisted. */
    private fun applyIntent(i: Intent?) {
        val cam = i?.getStringExtra("cam")
        val proc = i?.getStringExtra("proc")
        if (cam != null) prefs.edit().putString("cam", cam).apply()
        if (proc == "off" || proc == "default") prefs.edit().putString("proc", proc).apply()
        val newCam = prefs.getString("cam", "auto") ?: "auto"
        val newProc = prefs.getString("proc", "off") ?: "off"
        val camChanged = newCam != camera.requestedCamera
        camera.requestedCamera = newCam
        camera.procMode = newProc
        showProc(newProc)
        if (camChanged && cam != null && camera.hasStarted) camera.restart()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (Build.VERSION.SDK_INT >= 28) {
            window.attributes.layoutInDisplayCutoutMode =
                WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
        }
        prefs = getSharedPreferences("stegocam", MODE_PRIVATE)
        camera = CameraController(this, this)

        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        textureView = AutoFitTextureView(this)
        root.addView(textureView, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT, Gravity.CENTER))
        guide = GuideView(this, textureView)
        root.addView(guide, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))

        liveOverlay = LiveOverlayView(this, textureView)
        root.addView(liveOverlay, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))

        info = TextView(this).apply {
            setTextColor(Color.WHITE); textSize = 12f
            setPadding(dp(8), dp(4), dp(8), dp(0))
            text = "StegoCam"
        }
        legend = TextView(this).apply {
            setTextColor(Color.rgb(200, 200, 200)); textSize = 9f
            setPadding(dp(8), dp(0), dp(8), dp(2))
            text = "cyan: pending \u00b7 green: accepted \u00b7 yellow: CRC ok, weak z \u00b7 red: rejected"
            visibility = View.GONE
        }
        summaryView = TextView(this).apply {
            setTextColor(Color.rgb(120, 255, 150)); textSize = 11f; typeface = android.graphics.Typeface.MONOSPACE
            setPadding(dp(8), dp(0), dp(8), dp(4))
            visibility = View.GONE
        }
        statusBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.argb(140, 0, 0, 0))
            addView(info); addView(legend); addView(summaryView)
        }
        root.addView(statusBox, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.TOP or Gravity.START).apply { setMargins(dp(8), dp(36), dp(8), 0) })

        panel = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(dp(4), dp(4), dp(4), dp(24))
            setBackgroundColor(Color.argb(110, 0, 0, 0))
        }
        val lock = CheckBox(this).apply {
            text = "Lock"; setTextColor(Color.WHITE)
            setOnCheckedChangeListener { _, on -> camera.setLock(on) }
        }
        liveBox = CheckBox(this).apply {
            text = "Live"; setTextColor(Color.WHITE)
            setOnCheckedChangeListener { _, on ->
                camera.liveOn = on
                prefs.edit().putBoolean("live", on).apply()
                guide.visibility = if (on) View.GONE else View.VISIBLE
                legend.visibility = if (on) View.VISIBLE else View.GONE
                summaryView.visibility = if (on) View.VISIBLE else View.GONE
                if (!on) liveOverlay.showBoxes(emptyList(), emptyList())
            }
            isChecked = prefs.getBoolean("live", false)
        }
        val orient = Button(this).apply {
            text = "guide H/V"; isAllCaps = false
            setOnClickListener { guide.vertical = !guide.vertical }
        }
        procButton = Button(this).apply {
            text = "proc: off"; isAllCaps = false
            setOnClickListener {
                val next = if (camera.procMode == "off") "default" else "off"
                prefs.edit().putString("proc", next).apply()
                camera.setProc(next); showProc(next)
            }
        }
        val dumpBtn = Button(this).apply {
            text = "Dump 30"; isAllCaps = false
            setOnClickListener { camera.requestDump(30, "manual", null) }
        }
        val gatherBtn = Button(this).apply {
            text = "Gather"; isAllCaps = false
            setOnClickListener { if (!gather.active) { enterGather(); gather.startDialog() } }
        }
        val scanBtn = Button(this).apply {
            text = "Scanner"; isAllCaps = false
            setOnClickListener { if (!scanner.active) { enterScanner(); scanner.start() } }
        }
        for (v in listOf(lock, liveBox, orient, procButton, dumpBtn, gatherBtn, scanBtn))
            panel.addView(v, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(panel, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM))

        gather = GatherController(this, root, camera) { panel.visibility = View.VISIBLE; statusBoxShow(true); if (!camera.liveOn) guide.visibility = View.VISIBLE }
        scanner = ScannerController(this, root, camera, textureView) { panel.visibility = View.VISIBLE; statusBoxShow(true); if (!camera.liveOn) guide.visibility = View.VISIBLE }
        textureView.setOnTouchListener { v, ev ->
            if (scanner.active && scanner.hasResult) { if (ev.action == MotionEvent.ACTION_UP) scanner.rescan(); return@setOnTouchListener true }
            if (ev.action == MotionEvent.ACTION_UP && v.width > 0 && v.height > 0) {
                // the TextureView is sized to the upright (portrait) aspect, so view coords are upright coords
                camera.focusAt((ev.x / v.width).coerceIn(0f, 1f), (ev.y / v.height).coerceIn(0f, 1f))
            }
            true
        }
        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) = openIfReady()
            override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) {}
            override fun onSurfaceTextureDestroyed(st: SurfaceTexture) = true
            override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}
        }
        setContentView(root)
        applyIntent(intent)
        if (intent?.getStringExtra("mode") == "scanner") root.post { if (!scanner.active) { enterScanner(); scanner.start(intent?.getStringExtra("src") ?: "both", intent?.getStringExtra("profile") ?: "demo") } }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        applyIntent(intent)
    }

    @Suppress("DEPRECATION")
    private fun hideSystemUi() {
        if (Build.VERSION.SDK_INT >= 30) {
            window.setDecorFitsSystemWindows(false)
            window.insetsController?.let {
                it.hide(WindowInsets.Type.systemBars())
                it.systemBarsBehavior = android.view.WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            }
        } else {
            window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_FULLSCREEN or
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or
                View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
        }
    }

    override fun onResume() {
        super.onResume()
        hideSystemUi()
        camera.start()
        applyIntent(null)
        val f = IntentFilter(ACTION_DUMP).apply { addAction(ACTION_LIVE); addAction(ACTION_DECODE_FILE); addAction(ACTION_GATHER); addAction(ACTION_POSE_TEST); addAction(ACTION_GATHER_TEST); addAction(ACTION_SCANNER) }
        if (Build.VERSION.SDK_INT >= 33) registerReceiver(receiver, f, Context.RECEIVER_EXPORTED)
        else registerReceiver(receiver, f)
        receiverRegistered = true
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
        } else openIfReady()
    }

    override fun onPause() {
        if (receiverRegistered) { unregisterReceiver(receiver); receiverRegistered = false }
        camera.stop()
        super.onPause()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) hideSystemUi()
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, res: IntArray) {
        if (code == REQ_CAMERA) {
            if (res.isNotEmpty() && res[0] == PackageManager.PERMISSION_GRANTED) openIfReady()
            else onMessage("CAMERA permission denied")
        }
    }

    private fun openIfReady() {
        val st = textureView.surfaceTexture ?: return
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        camera.open(st)
    }

    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        if (::scanner.isInitialized) scanner.onTouch(ev)
        return super.dispatchTouchEvent(ev)
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        if (::scanner.isInitialized && scanner.active) {
            if (keyCode == KeyEvent.KEYCODE_BACK) { scanner.stop(); return true }
            if (keyCode == KeyEvent.KEYCODE_VOLUME_UP || keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) { if (event.repeatCount == 0) scanner.rescan(); return true }
        }
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
            if (event.repeatCount == 0) camera.requestDump(30, "manual", null)
            return true
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent): Boolean =
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) true else super.onKeyUp(keyCode, event)

    // CameraController.Listener (called from camera thread)
    override fun onMessage(msg: String) {
        Log.i(TAG, msg)
        runOnUiThread { Toast.makeText(this, msg, Toast.LENGTH_SHORT).show() }
    }

    override fun onStatusLine(line: String) {
        runOnUiThread { info.text = line }
    }

    override fun onLive(boxes: List<FloatArray>, labels: List<LiveAnalyzer.Label?>, ms: Float, fps: Float, summary: String) {
        if (camera.liveOn) { liveOverlay.showBoxes(boxes, labels); runOnUiThread { summaryView.text = summary } }
    }

    override fun onPreviewSize(size: Size, sensorOrientation: Int) {
        runOnUiThread {
            this.sensorOrientation = sensorOrientation
            // portrait lock, display rotation 0: TextureView shows the sensor image upright when the
            // view has the swapped aspect for 90/270 sensors, so no extra matrix is needed.
            if (sensorOrientation % 180 == 90) textureView.setAspect(size.height, size.width)
            else textureView.setAspect(size.width, size.height)
        }
    }
}
