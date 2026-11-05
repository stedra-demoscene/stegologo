package me.stedra.stegocam

import android.app.Activity
import android.content.Intent
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.util.TypedValue
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast

/** Corner reticle + (reveal only) the region where the result was found. Normalised upright coordinates over the preview. */
class ScannerOverlay(ctx: android.content.Context, private val preview: View) : View(ctx) {
    @Volatile var region: FloatArray? = null
    @Volatile var regionColour = Color.WHITE
    @Volatile var regionLabel = ""
    @Volatile var showReticle = true
    private val d = resources.displayMetrics.density
    private val p = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 3f * d; strokeCap = Paint.Cap.ROUND }
    private val t = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 12f * resources.displayMetrics.scaledDensity; typeface = Typeface.MONOSPACE; isFakeBoldText = true }

    override fun onDraw(c: Canvas) {
        val w = width.toFloat(); val h = height.toFloat()
        if (showReticle) {
            p.color = Color.argb(200, 255, 255, 255)
            val s = minOf(w, h) * 0.62f; val l = w / 2 - s / 2; val tp = h * 0.42f - s / 2; val r = l + s; val b = tp + s; val a = 28f * d
            for ((x, y, dx, dy) in listOf(listOf(l, tp, 1f, 1f), listOf(r, tp, -1f, 1f), listOf(l, b, 1f, -1f), listOf(r, b, -1f, -1f))) {
                c.drawLine(x, y, x + dx * a, y, p); c.drawLine(x, y, x, y + dy * a, p)
            }
        }
        val rg = region ?: return
        if (preview.width == 0) return
        val pl = preview.left.toFloat(); val pt = preview.top.toFloat(); val pw = preview.width.toFloat(); val ph = preview.height.toFloat()
        p.color = regionColour
        val x0 = pl + rg[0] * pw; val y0 = pt + rg[1] * ph; val x1 = pl + rg[2] * pw; val y1 = pt + rg[3] * ph
        c.drawRoundRect(x0, y0, x1, y1, 8f * d, 8f * d, p)
        t.color = regionColour
        c.drawText(regionLabel, x0, maxOf(y0 - 6f * d, 14f * d), t)
    }
}

/**
 * Scanner mode UI. Preview + reticle; the result card (identical for QR and stego results): the link, "Tap to open". Hidden reveal toggle
 * (long-press the card, or a three-finger tap anywhere): source, z, time to result, both sources' timings, the region highlighted on the preview.
 */
class ScannerController(private val act: Activity, private val root: FrameLayout, private val camera: CameraController,
                        private val preview: View, private val onExit: () -> Unit) {
    var active = false
        private set
    private var engine: ScannerEngine? = null
    private val ui = Handler(Looper.getMainLooper())
    private val d = act.resources.displayMetrics.density
    private fun dp(v: Int) = (v * d).toInt()
    private var reveal = false
    private var lastThree = 0L
    private var shown: ScanResult? = null
    private var overlay: ScannerOverlay? = null
    private var card: LinearLayout? = null
    private var hint: TextView? = null
    private var debug: TextView? = null
    private var urlView: TextView? = null
    private var domainView: TextView? = null

    private fun build() {
        overlay = ScannerOverlay(act, preview).also { root.addView(it, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)) }
        hint = TextView(act).apply {
            text = "Point the camera at a code"; setTextColor(Color.WHITE); setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
            setShadowLayer(6f, 0f, 2f, Color.BLACK); gravity = Gravity.CENTER
        }.also { root.addView(it, FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.TOP or Gravity.CENTER_HORIZONTAL).apply { topMargin = dp(88) }) }
        debug = TextView(act).apply {
            setTextColor(Color.rgb(120, 255, 150)); setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f); typeface = Typeface.MONOSPACE
            setBackgroundColor(Color.argb(170, 0, 0, 0)); setPadding(dp(8), dp(4), dp(8), dp(4)); visibility = View.GONE
        }.also { root.addView(it, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.TOP).apply { topMargin = dp(36) }) }
        val title = TextView(act).apply {
            text = "LINK FOUND"; setTextColor(Color.rgb(140, 150, 165)); setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f); letterSpacing = 0.18f; typeface = Typeface.DEFAULT_BOLD
        }
        urlView = TextView(act).apply {
            setTextColor(Color.WHITE); setTextSize(TypedValue.COMPLEX_UNIT_SP, 22f); typeface = Typeface.DEFAULT_BOLD
            maxLines = 2; ellipsize = android.text.TextUtils.TruncateAt.MIDDLE
        }
        domainView = TextView(act).apply { setTextColor(Color.rgb(79, 195, 247)); setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f); setPadding(0, dp(6), 0, 0); text = "Tap to open" }
        card = LinearLayout(act).apply {
            orientation = LinearLayout.VERTICAL; setPadding(dp(22), dp(18), dp(22), dp(20))
            background = GradientDrawable().apply { setColor(Color.argb(244, 17, 20, 26)); cornerRadius = 22f * d; setStroke(dp(1), Color.argb(60, 255, 255, 255)) }
            addView(title); addView(urlView, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(6) }); addView(domainView)
            visibility = View.GONE; elevation = 12f * d
            setOnClickListener { open() }
            setOnLongClickListener { toggleReveal(); true }
        }.also { root.addView(it, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM).apply { setMargins(dp(14), 0, dp(14), dp(40)) }) }
    }

    private fun destroy() {
        for (v in listOf<View?>(overlay, hint, debug, card)) if (v != null) root.removeView(v)
        overlay = null; hint = null; debug = null; card = null
    }

    /** src: "both" (default), "qr" or "stego" (testing). */
    fun start(src: String = "both", profile: String = "demo") {
        if (active) return
        active = true; reveal = false
        build()
        val e = ScannerEngine(act.applicationContext, Sources.create(act.applicationContext, profile))
        e.useQr = src != "stego"; e.useStego = src != "qr"
        e.onChange = { ui.post { refresh() } }
        engine = e
        camera.setScanner(e)
        e.reset()
    }

    fun stop() {
        if (!active) return
        active = false
        camera.setScanner(null)
        engine?.shutdown(); engine = null
        destroy()
        onExit()
    }

    /** Test/demo helper: feed frames from a PGM (and optionally a second one, alternating) instead of the camera. */
    fun replay(path: String, path2: String? = null, fps: Int = 10) {
        val e = engine ?: return
        camera.setScanner(null)
        Thread {
            try {
                val frames = listOfNotNull(path, path2).map { DecodeCli.readPgm(java.io.File(it)) }
                var n = 0
                while (active && engine === e) {
                    e.onFrame(frames[n++ % frames.size], 90)
                    Thread.sleep(1000L / fps)
                }
                frames.forEach { it.release() }
            } catch (t: Throwable) { android.util.Log.e("StegoCam", "scanner replay failed", t) }
        }.start()
    }

    /** Scan again: dismiss the card. */
    fun rescan() {
        shown = null
        card?.visibility = View.GONE
        hint?.visibility = View.VISIBLE
        overlay?.region = null
        engine?.reset()
        refresh()
    }

    val hasResult get() = shown != null

    private fun open() {
        val r = shown ?: return
        try {
            val u = r.text
            if (u.startsWith("http://") || u.startsWith("https://")) act.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(u)).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            else Toast.makeText(act, u, Toast.LENGTH_LONG).show()
        } catch (e: Exception) { Toast.makeText(act, "Cannot open: ${r.text}", Toast.LENGTH_SHORT).show() }
    }

    fun toggleReveal() { reveal = !reveal; refresh() }

    /** Called from Activity.dispatchTouchEvent: a three-finger touch toggles the reveal. Never consumes. */
    fun onTouch(ev: MotionEvent) {
        if (!active) return
        if (ev.actionMasked == MotionEvent.ACTION_POINTER_DOWN && ev.pointerCount >= 3) {
            val now = System.currentTimeMillis()
            if (now - lastThree > 600) { lastThree = now; toggleReveal() }
        }
    }

    private fun refresh() {
        val e = engine ?: return
        val w = e.winner
        if (w != null && shown !== w) {
            shown = w
            urlView?.text = w.text
            hint?.visibility = View.GONE
            card?.apply { visibility = View.VISIBLE; translationY = 300f * d; alpha = 0f; animate().translationY(0f).alpha(1f).setDuration(220).start() }
        }
        val ov = overlay ?: return
        val dbg = debug ?: return
        if (!reveal) { dbg.visibility = View.GONE; ov.region = null; ov.showReticle = w == null; ov.postInvalidate(); return }
        dbg.visibility = View.VISIBLE
        val sb = StringBuilder()
        fun line(tag: String, r: ScanResult?) {
            sb.append(tag).append(": ")
            if (r == null) sb.append("--\n") else sb.append("%d ms  z %s  %s\n".format(r.tMs, if (r.source == ScanResult.Source.QR) "-" else "%.1f".format(r.z), r.detail))
        }
        sb.append("source: ").append(w?.source?.name ?: "none").append('\n')
        line("QR   ", e.qr); line("STEGO", e.stego)
        if (w != null) sb.append("text: ").append(w.raw.take(60))
        dbg.text = sb.toString()
        ov.region = w?.box
        ov.regionColour = if (w?.source == ScanResult.Source.QR) Color.rgb(255, 210, 0) else Color.rgb(0, 255, 120)
        ov.regionLabel = w?.source?.name ?: ""
        ov.showReticle = w == null
        ov.postInvalidate()
    }
}
