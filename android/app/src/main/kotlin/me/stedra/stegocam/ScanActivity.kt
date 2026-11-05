package me.stedra.stegocam

import android.Manifest
import android.app.Activity
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.SurfaceTexture
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.util.TypedValue
import android.view.Gravity
import android.view.TextureView
import android.view.View
import android.view.ViewGroup
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.TextView
import android.widget.Toast
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * The public scanner: full-screen camera, ONE tappable "Open link: ..." pill on the object found (QR URL or demo design; the UI never shows which), following the
 * object while it is in view. The dev UI (MainActivity: gather, dumps, reveal) opens on a double shake. Dev replay (debuggable builds only):
 * broadcast me.stedra.stegocam.REPLAY --es file <pgm> feeds a stored frame instead of the camera; --ez off true goes back to the camera.
 */
class ScanActivity : Activity(), CameraController.Listener, SensorEventListener {
    companion object {
        private const val TAG = "StegoCam"
        const val ACTION_REPLAY = "me.stedra.stegocam.REPLAY"
        private const val HOLD_MS = 500L
        private const val FADE_MS = 250L
        private const val QR_PRIORITY_MS = 200L
        private const val SHAKE_G = 2.5f
        private const val SHAKE_PEAKS = 4
        private const val SHAKE_WINDOW_MS = 1200L
        private const val SHAKE_GAP_MIN_MS = 80L
        private const val SHAKE_GAP_MAX_MS = 400L
        private const val SHAKE_LOBE_JOIN_MS = 60L
        private const val SHAKE_COOLDOWN_MS = 2500L
    }

    /** Last detection of one source (qr / logo); the UI never shows which. */
    private class Src(var url: String, var cx: Float, var cy: Float, var seen: Long)

    private lateinit var camera: CameraController
    private lateinit var texture: AutoFitTextureView
    private lateinit var layer: FrameLayout
    private var scanner: PublicScanner? = null
    private val srcs = HashMap<String, Src>()
    private val logged = HashSet<String>()
    private var pill: TextView? = null              // the one and only pill
    private var pillUrl = ""
    private var alphaCur = 0f
    private var pillSrc: Src? = null               // source the pill currently follows
    private var px = 0f; private var py = 0f        // smoothed centre (normalised upright)
    private val ui = Handler(Looper.getMainLooper())
    private var startMs = 0L
    private var replayThread: Thread? = null
    @Volatile private var replaying = false
    private var receiverRegistered = false

    // shake
    private var lastLaunch = 0L

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (Build.VERSION.SDK_INT >= 28) window.attributes.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
        camera = CameraController(this, this)
        camera.procMode = "off"
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        texture = AutoFitTextureView(this)
        root.addView(texture, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT, Gravity.CENTER))
        layer = FrameLayout(this)
        root.addView(layer, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        texture.setOnTouchListener { v, ev ->
            if (ev.action == android.view.MotionEvent.ACTION_UP && v.width > 0 && v.height > 0)
                camera.focusAt((ev.x / v.width).coerceIn(0f, 1f), (ev.y / v.height).coerceIn(0f, 1f))
            true
        }
        texture.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) = openIfReady()
            override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) {}
            override fun onSurfaceTextureDestroyed(st: SurfaceTexture) = true
            override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}
        }
        setContentView(root)
    }

    @Suppress("DEPRECATION")
    private fun hideSystemUi() {
        if (Build.VERSION.SDK_INT >= 30) {
            window.setDecorFitsSystemWindows(false)
            window.insetsController?.let { it.hide(WindowInsets.Type.systemBars()); it.systemBarsBehavior = android.view.WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE }
        } else window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_FULLSCREEN or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
            View.SYSTEM_UI_FLAG_LAYOUT_STABLE or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
    }

    override fun onResume() {
        super.onResume()
        hideSystemUi()
        startMs = SystemClock.elapsedRealtime()
        camera.start()
        val ps = PublicScanner(applicationContext) { d -> ui.post { onDetection(d) } }
        scanner = ps
        camera.setScanner(ps)
        for (l in lobes) l.clear()
        (getSystemService(Context.SENSOR_SERVICE) as SensorManager).let { sm ->
            (sm.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION) ?: sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER))?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        }
        if (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) {
            val f = IntentFilter(ACTION_REPLAY)
            if (Build.VERSION.SDK_INT >= 33) registerReceiver(receiver, f, Context.RECEIVER_EXPORTED) else registerReceiver(receiver, f)
            receiverRegistered = true
        }
        ui.post(ticker)
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) requestPermissions(arrayOf(Manifest.permission.CAMERA), 1) else openIfReady()
    }

    override fun onPause() {
        ui.removeCallbacks(ticker)
        replaying = false
        (getSystemService(Context.SENSOR_SERVICE) as SensorManager).unregisterListener(this)
        if (receiverRegistered) { unregisterReceiver(receiver); receiverRegistered = false }
        camera.setScanner(null)
        camera.stop()
        scanner?.shutdown(); scanner = null
        pill?.let { layer.removeView(it) }; pill = null; pillUrl = ""; srcs.clear(); logged.clear()
        super.onPause()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) { super.onWindowFocusChanged(hasFocus); if (hasFocus) hideSystemUi() }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, res: IntArray) {
        if (res.isNotEmpty() && res[0] == PackageManager.PERMISSION_GRANTED) openIfReady() else Toast.makeText(this, "Camera permission is needed to scan", Toast.LENGTH_LONG).show()
    }

    private fun openIfReady() {
        val st = texture.surfaceTexture ?: return
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        camera.open(st)
    }

    // ---- CameraController.Listener
    override fun onMessage(msg: String) { Log.i(TAG, msg) }
    override fun onStatusLine(line: String) {}
    override fun onPreviewSize(size: Size, sensorOrientation: Int) {
        runOnUiThread { if (sensorOrientation % 180 == 90) texture.setAspect(size.height, size.width) else texture.setAspect(size.width, size.height) }
    }

    // ---- detections -> pills
    private fun display(url: String): String {
        var s = url.replace(Regex("^[a-zA-Z]+://"), "")
        if (s.endsWith("/")) s = s.dropLast(1)
        return if (s.length > 34) s.take(31) + "..." else s
    }

    private fun makePill(url: String): TextView = TextView(this).apply {
        text = "Open link: " + display(url)
        setTextColor(Color.WHITE); setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f); typeface = Typeface.DEFAULT_BOLD
        setPadding(dp(16), dp(11), dp(16), dp(11)); maxLines = 1; elevation = dp(8).toFloat()
        background = GradientDrawable().apply { setColor(Color.argb(238, 20, 24, 34)); cornerRadius = dp(26).toFloat(); setStroke(dp(2), Color.rgb(79, 195, 247)) }
        setOnClickListener { open(url) }
        measure(View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED), View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED))
    }

    private fun open(url: String) {
        try { startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) }
        catch (e: Exception) { Toast.makeText(this, "Cannot open: $url", Toast.LENGTH_SHORT).show() }
    }

    private fun onDetection(d: PublicScanner.Det) {
        val now = SystemClock.elapsedRealtime()
        if (d.kind != "qr" && now - (srcs["qr"]?.seen ?: -1_000_000L) < QR_PRIORITY_MS) return   // QR has priority: a stego result needs >= 200 ms without a QR
        val e = srcs[d.kind]
        if (e == null) srcs[d.kind] = Src(d.url, d.cx, d.cy, now) else { e.url = d.url; e.cx = d.cx; e.cy = d.cy; e.seen = now }
        if (logged.add(d.kind)) {          // source only in logcat
            val lat = (System.nanoTime() - d.t0Ns) / 1e6
            Log.i(TAG, "SCAN_BUTTON kind=${d.kind} frame_to_button_ms=%.0f since_start_ms=${now - startMs} z=%.1f url=${d.url}".format(lat, d.z))
        }
        layoutPill(now)
    }

    private val ticker = object : Runnable {
        override fun run() { layoutPill(SystemClock.elapsedRealtime()); ui.postDelayed(this, 33) }
    }

    /** One pill for all sources (the UI never shows which): the QR while it is seen (it replaces a shown stego pill at once), else the stego result; held 0.5 s, then faded. */
    private fun layoutPill(now: Long) {
        if (texture.width == 0) return
        val act = srcs.values.filter { now - it.seen <= HOLD_MS }
        val newest = srcs.values.maxByOrNull { it.seen }
        var v = pill
        if (act.isEmpty()) {
            if (v != null) {
                alphaCur += 0.18f * (0f - alphaCur)                                   // exponential fade-out
                if (alphaCur < 0.03f) { layer.removeView(v); pill = null; pillUrl = ""; alphaCur = 0f } else { v.alpha = alphaCur; v.scaleX = 0.94f + 0.06f * alphaCur; v.scaleY = v.scaleX }
            }
            return
        }
        // QR position/url when a QR is seen, else the logo's (controlled demo inputs: no conflict handling)
        val pref = act.firstOrNull { it === srcs["qr"] } ?: act.first()
        val tx = pref.cx; val ty = pref.cy; val url = pref.url
        if (v == null) {
            v = makePill(url); pill = v; pillUrl = url; px = tx; py = ty; alphaCur = 0f
            layer.addView(v, FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.TOP or Gravity.START))
        } else if (url != pillUrl) {
            v.text = "Open link: " + display(url); pillUrl = url
            v.setOnClickListener { open(url) }
            v.measure(View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED), View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED))
        }
        pillSrc = pref
        px += 0.35f * (tx - px); py += 0.35f * (ty - py)               // exponential glide, also across a source switch
        alphaCur += 0.3f * (1f - alphaCur)                                           // exponential ease-out appear
        v.alpha = alphaCur; v.scaleX = 0.94f + 0.06f * alphaCur; v.scaleY = v.scaleX
        val pl = texture.left.toFloat(); val pt = texture.top.toFloat(); val pw = texture.width.toFloat(); val ph = texture.height.toFloat()
        val w = v.measuredWidth.toFloat(); val h = v.measuredHeight.toFloat()
        val sw = layer.width.toFloat(); val sh = layer.height.toFloat()
        v.translationX = (pl + px * pw - w / 2).coerceIn(dp(4).toFloat(), maxOf(dp(4).toFloat(), sw - w - dp(4)))
        v.translationY = (pt + py * ph - h / 2).coerceIn(dp(4).toFloat(), maxOf(dp(4).toFloat(), sh - h - dp(4)))
    }

    // ---- shake -> dev UI
    /**
     * Deliberate shake: >= 4 alternating-sign lobes on ONE axis of linear acceleration (gravity removed), each |a| > 2.5 g, consecutive lobes 80-400 ms apart,
     * all within 1.2 s. Ignored for 1 s after resume and while a pill is visible. Every candidate (>= 2 lobes) is logged.
     */
    private class Lobe(val t: Long, val sign: Int, var peak: Float, var lastT: Long)
    private val lobes = Array(3) { ArrayList<Lobe>() }
    private var linearOk = true

    private fun logCand(axis: Int, n: Int, gmax: Float, verdict: String) { if (n >= 2) Log.i(TAG, "SHAKE candidate axis=${"xyz"[axis]} peaks=$n max_g=%.1f $verdict".format(gmax)) }

    override fun onSensorChanged(e: SensorEvent) {
        val now = SystemClock.elapsedRealtime()
        for (ax in 0..2) {
            val a = e.values[ax]; val g = abs(a) / SensorManager.GRAVITY_EARTH
            val l = lobes[ax]
            // expire a stale sequence
            if (l.isNotEmpty() && now - l.last().lastT > SHAKE_GAP_MAX_MS) { logCand(ax, l.size, l.maxOf { abs(it.peak) } / SensorManager.GRAVITY_EARTH, "(timeout)"); l.clear() }
            if (g < SHAKE_G) continue
            val sg = if (a > 0) 1 else -1
            val last = l.lastOrNull()
            if (last != null && last.sign == sg && now - last.lastT <= SHAKE_LOBE_JOIN_MS) { last.lastT = now; if (abs(a) > abs(last.peak)) last.peak = a; continue }
            if (last != null && (last.sign == sg || now - last.t < SHAKE_GAP_MIN_MS || now - last.t > SHAKE_GAP_MAX_MS)) {
                logCand(ax, l.size, l.maxOf { abs(it.peak) } / SensorManager.GRAVITY_EARTH, "(not alternating / spacing)"); l.clear()
            }
            l.add(Lobe(now, sg, a, now))
            if (l.size >= SHAKE_PEAKS && now - l.first().t <= SHAKE_WINDOW_MS) {
                val gm = l.maxOf { abs(it.peak) } / SensorManager.GRAVITY_EARTH
                l.clear()
                if (now - startMs < 1000 || pill != null || now - lastLaunch < SHAKE_COOLDOWN_MS) { logCand(ax, SHAKE_PEAKS, gm, "(ignored: ${if (pill != null) "pill visible" else "just resumed / cooldown"})"); return }
                lastLaunch = now
                Log.i(TAG, "SHAKE -> dev UI axis=${"xyz"[ax]} peaks=$SHAKE_PEAKS max_g=%.1f".format(gm))
                startActivity(Intent(this, MainActivity::class.java))
                return
            }
        }
    }
    override fun onAccuracyChanged(s: Sensor?, a: Int) {}

    // ---- dev replay
    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(c: Context, i: Intent) {
            replaying = false
            if (i.getBooleanExtra("off", false)) { scanner?.let { it.enabled = true; camera.setScanner(it) }; return }
            val path = i.getStringExtra("file") ?: return
            val fps = i.getIntExtra("fps", 10)
            val ps = scanner ?: return
            camera.setScanner(null)
            replaying = true
            Thread {
                try {
                    val m = DecodeCli.readPgm(java.io.File(path))
                    Thread.sleep(200)
                    while (replaying && scanner === ps) { ps.onFrame(m, 90); Thread.sleep(1000L / fps) }
                    m.release()
                } catch (t: Throwable) { Log.e(TAG, "replay failed", t) }
            }.also { replayThread = it }.start()
        }
    }
}
