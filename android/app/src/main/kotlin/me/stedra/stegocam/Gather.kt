package me.stedra.stegocam

import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs
import kotlin.math.sqrt

/** One prefilled capture scenario (see assets/scenarios.json; override by pushing files/scenarios.json). */
class Scenario(val j: JSONObject) {
    val id: String = j.getString("id")
    val page: Int = j.optInt("page", 0)
    /** Demo-card target: "020", "020_15um", "020_ctrl" (the label printed on the card); "none" = no card (manual shot, no pose checks). */
    val card: String? = if (j.has("card")) j.getString("card") else null
    val isCard: Boolean get() = card != null
    val manual: Boolean = j.optBoolean("manual", false)
    val basePct: Int get() = card?.take(3)?.toIntOrNull() ?: 0
    val maxRadius: Double = j.optDouble("max_radius", 0.2)           // card scenarios: design centre within this fraction of the half-diagonal
    val minRadius: Double = j.optDouble("min_radius", 0.0)
    val baselineKey: String get() = if (isCard) "c%03d".format(basePct) else "$page"
    val kind: String = j.optString("kind", "fixture")                 // fixture | handheld | bend
    val tiltLr: Double = j.optDouble("tilt_lr_deg", 0.0)
    val tiltUd: Double = j.optDouble("tilt_ud_deg", 0.0)
    val tolDeg: Double = j.optDouble("tol_deg", 3.0)
    val distFactor: Double? = if (j.has("dist_factor")) j.getDouble("dist_factor") else null
    val distMm: Double? = if (j.has("dist_mm")) j.getDouble("dist_mm") else null
    val distTol: Double = j.optDouble("tol", 0.10)
    val nFrames: Int = j.optInt("n_frames", 5)
    val proc: String = j.optString("proc", "off")
    val instruction: String = j.optString("instruction", if (card != null) "Show card $card" else "Put page $page in view")
    val baseline: Boolean = j.optBoolean("baseline", false)           // the flat shot that sets the page's baseline distance
    val signed: Boolean = j.optBoolean("signed", false)               // tilt targets are signed (default: |tilt| must match)
    val steadyText: String get() = when (kind) {
        "handheld" -> "Hold the phone by hand. Keep as still as you can."
        "bend" -> "Hold the bend still."
        else -> "Phone fixed. Do not touch the table."
    }
    val label: String get() = if (isCard) "%s (card %s, %s)".format(id, card, kind) else "%s (page %d, %s)".format(id, page, kind)
}

/** A session folder under captures/: session.json lists every scenario with its status. */
class Session(val dir: File, val manifest: JSONObject) {
    val items: JSONArray = manifest.getJSONArray("items")
    val baselines: JSONObject = manifest.getJSONObject("baselines")
    val size get() = items.length()
    val name: String get() = dir.name
    fun scenario(i: Int) = Scenario(items.getJSONObject(i))
    fun item(i: Int): JSONObject = items.getJSONObject(i)
    fun status(i: Int): String = item(i).optString("status", "pending")
    fun firstPending(): Int { for (i in 0 until size) if (status(i) == "pending") return i; return size }
    fun doneCount(): Int { var n = 0; for (i in 0 until size) if (status(i).startsWith("done")) n++; return n }
    @Synchronized fun save() { try { File(dir, "session.json").writeText(manifest.toString(1)) } catch (e: Exception) { Log.e("StegoCam", "session save", e) } }
    fun baselineFor(key: String): Double? =
        if (baselines.has(key)) baselines.getDouble(key) else if (!key.startsWith("c") && baselines.has("last")) baselines.getDouble("last") else null

    companion object {
        fun fileFor(set: String) = if (set == "demo") "scenarios_demo.json" else "scenarios.json"
        fun loadScenarios(ctx: Context, set: String = "sheet4"): List<JSONObject> {
            val f = File(ctx.getExternalFilesDir(null) ?: ctx.filesDir, fileFor(set))
            val txt = if (f.exists()) f.readText() else ctx.assets.open(fileFor(set)).bufferedReader().use { it.readText() }
            val arr = JSONObject(txt).getJSONArray("scenarios")
            return (0 until arr.length()).map { arr.getJSONObject(it) }
        }
        fun create(ctx: Context, camera: CameraController, set: String = "sheet4"): Session {
            val base = ctx.getExternalFilesDir("captures") ?: ctx.filesDir
            val name = "session_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val dir = File(base, name); dir.mkdirs()
            val m = JSONObject().put("session", name).put("device", Build.MANUFACTURER + " " + Build.MODEL)
                .put("started", System.currentTimeMillis()).put("intrinsics", JSONArray(camera.intrinsics.toList()))
                .put("intrinsics_source", camera.intrinsicsSrc).put("baselines", JSONObject()).put("kind", set)
            val items = JSONArray()
            for ((i, s) in loadScenarios(ctx, set).withIndex()) items.put(JSONObject(s.toString()).put("idx", i).put("status", "pending").put("attempts", 0))
            m.put("items", items)
            return Session(dir, m).also { it.save() }
        }
        fun load(dir: File) = Session(dir, JSONObject(File(dir, "session.json").readText()))
        fun latest(ctx: Context): Session? {
            val base = ctx.getExternalFilesDir("captures") ?: ctx.filesDir
            val d = base.listFiles { f -> f.isDirectory && f.name.startsWith("session_") && File(f, "session.json").exists() }?.maxByOrNull { it.name } ?: return null
            return try { load(d) } catch (e: Exception) { null }
        }
    }
}

/** The guided capture UI + state machine (main thread). Poses arrive from the analysis thread via [onPose]. */
class GatherController(private val act: Activity, private val parent: FrameLayout, private val camera: CameraController,
                       private val onExit: () -> Unit) {
    companion object {
        private const val TAG = "StegoCam"
        private const val READY_MS = 1000L
        private const val SETTLE_MS = 300L
        private const val STALE_MS = 1500L
        private const val STEADY_LR = 0.6; private const val STEADY_UD = 0.6; private const val STEADY_D = 0.015
    }
    private enum class St { IDLE, ALIGN, COUNT, SHOOT, SAVED, FINISHED }

    private val h = Handler(Looper.getMainLooper())
    private var st = St.IDLE
    private var session: Session? = null
    private var idx = 0
    private var cur: Scenario? = null
    private var attempt = 0

    private class Sample(val t: Long, val lr: Double, val ud: Double, val d: Double)
    private val samples = ArrayList<Sample>()
    @Volatile private var lastPose: PoseEstimator.Pose? = null
    private var okSince = 0L
    private var countN = 0
    private var badInCount = 0
    private var lastLog = 0L
    private var tts: TextToSpeech? = null
    private var ttsOk = false
    private var tone: ToneGenerator? = null
    private val demo by lazy { DemoStego(act.applicationContext) }
    private val qrOnce by lazy { QrScanner() }
    private val resultExec = java.util.concurrent.Executors.newSingleThreadExecutor { Thread(it, "gather-result") }
    private val result = tv(15f, Color.rgb(160, 255, 160), mono = true)
    private lateinit var shootBtn: Button

    // ---- UI
    private fun dp(v: Int) = (v * act.resources.displayMetrics.density).toInt()
    private fun tv(sp: Float, color: Int = Color.WHITE, bold: Boolean = false, mono: Boolean = false) = TextView(act).apply {
        textSize = sp; setTextColor(color); setShadowLayer(4f, 0f, 0f, Color.BLACK)
        if (bold) setTypeface(typeface, Typeface.BOLD); if (mono) typeface = Typeface.create(Typeface.MONOSPACE, if (bold) Typeface.BOLD else Typeface.NORMAL)
    }
    private val progress = tv(15f, Color.LTGRAY)
    private val instr = tv(24f, Color.YELLOW, bold = true)
    private val lLr = tv(21f, mono = true); private val lUd = tv(21f, mono = true); private val lD = tv(21f, mono = true); private val lInfo = tv(15f, Color.LTGRAY, mono = true)
    private val hint = tv(22f, Color.CYAN, bold = true)
    private val banner = tv(80f, Color.WHITE, bold = true).apply { gravity = Gravity.CENTER }
    private val steady = tv(16f, Color.LTGRAY, mono = true)
    private val bar = ProgressBar(act, null, android.R.attr.progressBarStyleHorizontal).apply { max = 100 }
    private val ui = FrameLayout(act).apply { visibility = View.GONE }

    init {
        val top = LinearLayout(act).apply {
            orientation = LinearLayout.VERTICAL; setBackgroundColor(Color.argb(150, 0, 0, 0)); setPadding(dp(10), dp(36), dp(10), dp(6))
            for (v in listOf(progress, instr, lLr, lUd, lD, lInfo, hint, result)) addView(v)
        }
        val bottom = LinearLayout(act).apply {
            orientation = LinearLayout.VERTICAL; setBackgroundColor(Color.argb(150, 0, 0, 0)); setPadding(dp(8), dp(4), dp(8), dp(24))
            addView(steady); addView(bar, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(14)))
            val row = LinearLayout(act).apply { orientation = LinearLayout.HORIZONTAL }
            fun btn(t: String, f: () -> Unit) = Button(act).apply { text = t; isAllCaps = false; textSize = 16f; setOnClickListener { f() } }
            shootBtn = btn("Shoot") { if (st == St.ALIGN && cur?.manual == true) startCountdown() }.apply { visibility = View.GONE }
            for (b in listOf(shootBtn, btn("Back") { go(idx - 1) }, btn("Redo") { go(idx) }, btn("Skip") { skip() }, btn("Exit") { stop() }))
                row.addView(b, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(row)
        }
        ui.addView(top, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.TOP))
        ui.addView(banner, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.CENTER))
        ui.addView(bottom, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM))
        parent.addView(ui, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
    }

    val active get() = st != St.IDLE

    // ---- start / stop

    fun startDialog() {
        val last = Session.latest(act)
        val b = AlertDialog.Builder(act).setTitle("Gather data")
        val ns = Session.loadScenarios(act, "sheet4").size; val nd = Session.loadScenarios(act, "demo").size
        val items = ArrayList<String>(); val acts = ArrayList<() -> Unit>()
        if (last != null && last.firstPending() < last.size) { items.add("Resume ${last.name} (${last.doneCount()}/${last.size})"); acts.add { start(last) } }
        items.add("New session: demo cards ($nd)"); acts.add { start(null, -1, "demo") }
        items.add("New session: sheet 4 ($ns)"); acts.add { start(null, -1, "sheet4") }
        b.setItems(items.toTypedArray()) { _, w -> acts[w]() }
        b.setNegativeButton("Cancel") { _, _ -> onExit() }.setOnCancelListener { onExit() }.show()
    }

    fun start(resume: Session?, startIdx: Int = -1, set: String = "sheet4") {
        if (!PoseEstimator.init(act)) { Log.e(TAG, "GATHER: OpenCV/pose init failed"); return }
        if (tts == null) tts = TextToSpeech(act) { s -> ttsOk = s == TextToSpeech.SUCCESS; if (ttsOk) tts?.setLanguage(Locale.US) }
        if (tone == null) tone = try { ToneGenerator(AudioManager.STREAM_MUSIC, 100) } catch (e: Exception) { null }
        val s = resume ?: Session.create(act, camera, set)
        session = s
        ui.visibility = View.VISIBLE
        camera.liveOn = false
        idx = if (startIdx >= 0) startIdx else s.firstPending()
        Log.i(TAG, "GATHER start session=${s.name} idx=$idx/${s.size}")
        camera.setPoseSink({ p -> h.post { onPose(p) } }, poseTarget(s.scenario(minOf(idx, s.size - 1))))
        applyTarget(s.scenario(minOf(idx, s.size - 1)))
        h.post(ticker)
        go(idx)
    }

    fun stop() {
        h.removeCallbacksAndMessages(null)
        camera.setPoseSink(null, 1); camera.setPoseDemo(null)
        ui.visibility = View.GONE
        st = St.IDLE
        tts?.stop()
        Log.i(TAG, "GATHER stop")
        onExit()
    }

    fun shutdown() { h.removeCallbacksAndMessages(null); resultExec.shutdown(); demo.shutdown(); tts?.shutdown(); tone?.release(); tts = null; tone = null }

    private fun poseTarget(sc: Scenario) = if (sc.isCard) sc.basePct else sc.page
    private fun applyTarget(sc: Scenario) {
        camera.setPoseDemo(if (sc.isCard) demo else null)
        camera.setPoseExpected(poseTarget(sc))
    }

    private fun skip() {
        val s = session ?: return
        if (idx < s.size && s.status(idx) == "pending") { s.item(idx).put("status", "skipped"); s.save() }
        go(idx + 1)
    }

    private fun go(i: Int) {
        val s = session ?: return
        h.removeCallbacks(countStep); h.removeCallbacks(shootStep)
        idx = i.coerceIn(0, s.size)
        samples.clear(); okSince = 0; badInCount = 0
        if (idx >= s.size) {
            st = St.FINISHED; cur = null
            progress.text = "${s.doneCount()}/${s.size} done"; instr.text = "All scenarios done. Press Exit, then pull the session."
            for (v in listOf(lLr, lUd, lD, lInfo, hint, steady)) v.text = ""
            banner.text = "DONE"; banner.setTextColor(Color.GREEN); bar.progress = 0
            speak("All done."); return
        }
        val sc = s.scenario(idx); cur = sc
        applyTarget(sc)
        shootBtn.visibility = if (sc.manual) View.VISIBLE else View.GONE
        result.text = ""
        st = St.ALIGN
        attempt = s.item(idx).optInt("attempts", 0)
        progress.text = "${idx + 1}/${s.size}  " + (if (sc.isCard) "card ${sc.card}" else "page ${sc.page}") + "  [${sc.id}]" + if (attempt > 0) "  (take ${attempt + 1})" else ""
        instr.text = sc.instruction
        banner.text = "ALIGN"; banner.setTextColor(Color.WHITE)
        speak(sc.instruction)
        render(lastPose)
    }

    // ---- evaluation

    private class Check(val ok: Boolean, val lr: Boolean?, val ud: Boolean?, val dist: Boolean?, val hint: String, val targetDist: Double?)

    private fun targetDist(sc: Scenario, s: Session): Double? {
        if (sc.distMm != null) return sc.distMm
        if (sc.baseline) return null
        val b = s.baselineFor(sc.baselineKey) ?: return null
        return (sc.distFactor ?: 1.0) * b
    }

    private fun evaluate(p: PoseEstimator.Pose?, sc: Scenario, s: Session): Check {
        if (sc.manual) return Check(true, null, null, null, "No pose check. Press Shoot when ready.", null)
        if (p == null || !p.found) return Check(false, null, null, null, if (sc.isCard) "Card not found. Show card ${sc.card} fully, in focus." else "Page not found. Show page ${sc.page} fully, in focus.", null)
        if (!p.pageOk) return Check(false, null, null, null, if (sc.isCard) "WRONG CARD: this looks like the ${p.page}% card, need ${sc.basePct}%." else "WRONG PAGE: this looks like page ${p.page}, need page ${sc.page}.", null)
        if (sc.isCard && !p.radius.isNaN() && (p.radius > sc.maxRadius || p.radius < sc.minRadius)) {
            val where = if (p.radius > sc.maxRadius) "Move the card toward the CENTRE of the frame" else "Move the card toward the EDGE of the frame"
            return Check(false, null, null, null, "$where (radius %.2f, want %.2f..%.2f).".format(p.radius, sc.minRadius, sc.maxRadius), null)
        }
        if (sc.kind != "fixture") return Check(true, null, null, null, (if (sc.isCard) "Card found" else "Page found") + (if (sc.kind == "handheld") ". Hold still." else ". Hold the bend still."), null)
        val lr = if (sc.signed) p.tiltLr else abs(p.tiltLr); val ud = if (sc.signed) p.tiltUd else abs(p.tiltUd)
        val tl = if (sc.signed) sc.tiltLr else abs(sc.tiltLr); val tu = if (sc.signed) sc.tiltUd else abs(sc.tiltUd)
        val lrOk = abs(lr - tl) <= sc.tolDeg; val udOk = abs(ud - tu) <= sc.tolDeg
        val td = targetDist(sc, s)
        val dOk = td == null || abs(p.distMm / td - 1) <= sc.distTol
        val hs = ArrayList<String>()
        if (!lrOk) hs.add("tilt left-right %s %.0f deg".format(if (lr < tl) "MORE by" else "LESS by", abs(tl - lr)))
        if (!udOk) hs.add("tilt up-down %s %.0f deg".format(if (ud < tu) "MORE by" else "LESS by", abs(tu - ud)))
        if (!dOk && td != null) hs.add(if (p.distMm > td) "move CLOSER by %.0f%%".format((p.distMm / td - 1) * 100) else "move FARTHER by %.0f%%".format((1 - p.distMm / td) * 100))
        return Check(lrOk && udOk && dOk, lrOk, udOk, dOk, if (hs.isEmpty()) "In position" else hs.joinToString("\n"), td)
    }

    private fun stats(): Triple<Double, Double, Double>? {          // sigma lr, ud (deg), relative sigma dist over the last 1 s
        val now = System.currentTimeMillis()
        val w = samples.filter { now - it.t <= 1000 }
        if (w.size < 3) return null
        fun sd(f: (Sample) -> Double): Double { val m = w.map(f).average(); return sqrt(w.map { (f(it) - m) * (f(it) - m) }.average()) }
        return Triple(sd { it.lr }, sd { it.ud }, sd { it.d } / w.map { it.d }.average())
    }

    private fun col(ok: Boolean?) = when (ok) { true -> Color.GREEN; false -> Color.rgb(255, 90, 90); null -> Color.WHITE }

    private fun render(p: PoseEstimator.Pose?) {
        val sc = cur ?: return; val s = session ?: return
        val c = evaluate(p, sc, s)
        val fix = sc.kind == "fixture"
        if (p != null && p.found) {
            val tl = if (sc.signed) sc.tiltLr else abs(sc.tiltLr); val tu = if (sc.signed) sc.tiltUd else abs(sc.tiltUd)
            lLr.text = "tilt LR %+6.1f" .format(p.tiltLr) + if (fix) "  -> %.0f +-%.0f".format(tl, sc.tolDeg) else ""; lLr.setTextColor(col(c.lr))
            lUd.text = "tilt UD %+6.1f".format(p.tiltUd) + if (fix) "  -> %.0f +-%.0f".format(tu, sc.tolDeg) else ""; lUd.setTextColor(col(c.ud))
            lD.text = "dist %5.0f mm".format(p.distMm) + (c.targetDist?.let { "  -> %.0f +-%.0f%%".format(it, sc.distTol * 100) } ?: if (sc.baseline) "  (sets baseline)" else ""); lD.setTextColor(col(c.dist))
            lInfo.text = "%.1f px/mm  ".format(p.pxPerMm) + (if (sc.isCard) "card %d (%d inl)".format(p.page, p.inliers) else "page %d (%d inl)".format(p.page, p.inliers)) +
                (if (p.radius.isNaN()) "" else "  r %.2f".format(p.radius)) + "  %.0f ms".format(p.ms)
        } else { lLr.text = "tilt LR  --"; lUd.text = "tilt UD  --"; lD.text = "dist  --"; lInfo.text = ""; for (v in listOf(lLr, lUd, lD)) v.setTextColor(Color.GRAY) }
        hint.text = c.hint
        hint.setTextColor(if (c.ok) Color.GREEN else Color.CYAN)
        val sd = stats()
        steady.text = sc.steadyText + (if (fix && sd != null) "\nsteadiness: LR %.2f UD %.2f deg, dist %.1f%%".format(sd.first, sd.second, sd.third * 100) else "")
    }

    // ---- pose updates

    private fun onPose(p: PoseEstimator.Pose?) {
        val sc = cur ?: return; val s = session ?: return
        if (st == St.IDLE || st == St.FINISHED) return
        lastPose = p
        val now = System.currentTimeMillis()
        if (now - lastLog > 1000) { lastLog = now; Log.i(TAG, "GATHER st=$st " + (p?.toJson()?.apply { remove("H_mm_to_px") } ?: "null")) }
        if (p != null && p.found) { samples.add(Sample(now, p.tiltLr, p.tiltUd, p.distMm)); while (samples.size > 40) samples.removeAt(0) }
        val c = evaluate(p, sc, s)
        when (st) {
            St.ALIGN -> {
                render(p)
                val sd = stats()
                val steadyOk = sc.kind != "fixture" || (sd != null && sd.first < STEADY_LR && sd.second < STEADY_UD && sd.third < STEADY_D)
                if (c.ok) { if (okSince == 0L) okSince = now } else okSince = 0
                val held = if (okSince == 0L) 0L else now - okSince
                bar.progress = (100 * minOf(held, READY_MS) / READY_MS).toInt()
                if (c.ok && !steadyOk) { banner.text = "STEADY"; banner.setTextColor(Color.YELLOW); okSince = if (held > READY_MS) now - READY_MS else okSince }
                else if (c.ok) { banner.text = "READY"; banner.setTextColor(Color.GREEN) }
                else { banner.text = "ALIGN"; banner.setTextColor(Color.WHITE) }
                if (sc.manual) { banner.text = "PRESS SHOOT"; banner.setTextColor(Color.WHITE); bar.progress = 0; return }
                if (c.ok && steadyOk && held >= READY_MS) startCountdown()
            }
            St.COUNT -> { render(p); if (!c.ok) badInCount++ else badInCount = 0; if (badInCount >= 3) { Log.i(TAG, "GATHER countdown aborted: pose lost"); abortCount() } }
            else -> {}
        }
    }

    private val ticker = object : Runnable {
        override fun run() {
            val p = lastPose
            if ((st == St.ALIGN || st == St.COUNT) && cur?.manual != true && (p == null || System.currentTimeMillis() - p.t > STALE_MS)) {
                lastPose = null; okSince = 0; bar.progress = 0
                if (st == St.COUNT) abortCount() else { render(null); banner.text = "ALIGN"; banner.setTextColor(Color.WHITE) }
            }
            h.postDelayed(this, 300)
        }
    }

    // ---- countdown and shot

    private fun startCountdown() {
        st = St.COUNT; countN = 3; badInCount = 0
        instr.text = cur?.steadyText ?: ""
        speak(cur?.steadyText ?: "")
        showCount()
    }

    private fun showCount() { banner.text = "$countN"; banner.setTextColor(Color.YELLOW); beep(false); h.postDelayed(countStep, 1000) }

    private val countStep = Runnable {
        countN--
        if (st != St.COUNT) return@Runnable
        if (countN > 0) showCount() else {
            // no haptic (it shook the camera during the dump); a 300 ms settle after the last beep, then the shot (pose re-checked in shoot())
            banner.text = "HOLD"; banner.setTextColor(Color.YELLOW)
            h.postDelayed(shootStep, SETTLE_MS)
        }
    }
    private val shootStep = Runnable { if (st == St.COUNT) shoot() }

    private fun abortCount() {
        h.removeCallbacks(countStep); h.removeCallbacks(shootStep); st = St.ALIGN; okSince = 0; bar.progress = 0
        instr.text = cur?.instruction ?: ""
        banner.text = "ALIGN"; banner.setTextColor(Color.WHITE)
    }

    private fun shoot() {
        val sc = cur ?: return; val s = session ?: return
        val p = lastPose
        val c = evaluate(p, sc, s)
        if (!sc.manual && (p == null || System.currentTimeMillis() - p.t > 700 || !c.ok)) { Log.i(TAG, "GATHER shot refused: pose not ok"); abortCount(); return }
        st = St.SHOOT
        banner.text = "SHOT"; banner.setTextColor(Color.RED)    // silent: the confirmation beep comes after the dump
        val it = s.item(idx)
        val att = it.optInt("attempts", 0)
        val sd = stats()
        val extras = JSONObject(sc.j.toString()).put("idx", idx).put("attempt", att).put("session", s.name)
            .put("pose", p?.toJson() ?: JSONObject.NULL).put("target_dist_mm", c.targetDist ?: JSONObject.NULL)
            .put("baseline_mm", s.baselineFor(sc.baselineKey) ?: JSONObject.NULL)
            .put("steady_sigma", if (sd == null) JSONObject.NULL else JSONObject().put("lr_deg", sd.first).put("ud_deg", sd.second).put("dist_rel", sd.third))
            .put("pose_recent", JSONArray(samples.takeLast(8).map { JSONObject().put("t_ms_ago", System.currentTimeMillis() - it.t).put("lr", it.lr).put("ud", it.ud).put("d", it.d) }))
        val dist = samples.filter { System.currentTimeMillis() - it.t < 1500 }.map { it.d }.sorted()
        val name = "%02d_%s".format(idx, sc.id) + if (att > 0) "_r$att" else ""
        val myIdx = idx
        camera.requestDump(sc.nFrames, sc.id, sc.proc, extras, s.dir, name) { dir, err ->
            h.post { shotDone(myIdx, dir, err, p, if (dist.isEmpty()) null else dist[dist.size / 2]) }
        }
    }

    private fun shotDone(i: Int, dir: File, err: String?, p: PoseEstimator.Pose?, medDist: Double?) {
        val s = session ?: return
        val it = s.item(i)
        it.put("attempts", it.optInt("attempts", 0) + 1)
        if (err != null) {
            it.put("status", "pending").put("error", err); s.save()
            banner.text = "ERROR"; banner.setTextColor(Color.RED); speak("Capture failed. Retrying.")
            h.postDelayed({ if (idx == i) go(i) }, 1500); return
        }
        val sc = s.scenario(i)
        it.put("status", "done").put("dir", dir.name).put("pose", p?.toJson() ?: JSONObject.NULL)
        val after = lastPose
        if (after != null && cur != null) {
            val ca = evaluate(after, sc, s)
            it.put("pose_after_ok", ca.ok)
            if (!ca.ok) it.put("status", "done_drift")
        }
        if (sc.baseline && medDist != null) { s.baselines.put(sc.baselineKey, medDist); if (!sc.isCard) s.baselines.put("last", medDist) }
        if (sc.isCard || sc.manual) runOnPhone(i, dir)
        s.save()
        st = St.SAVED
        banner.text = if (it.optString("status") == "done_drift") "SAVED (drifted)" else "SAVED"; banner.setTextColor(Color.GREEN)
        beep(false)
        h.postDelayed({ if (st == St.SAVED && idx == i) go(i + 1) }, 1200)
    }

    /** Scanner-engine decode + ML Kit QR on the first saved frame; stored in the scenario's session entry and shown as one line. */
    private fun runOnPhone(i: Int, dir: File) {
        val s = session ?: return
        resultExec.execute {
            val j = try { onPhoneResult(File(dir, "frame_000.pgm")) } catch (e: Throwable) { Log.e(TAG, "on-phone result failed", e); JSONObject().put("error", e.toString()) }
            h.post {
                val it = s.item(i); it.put("onphone", j); s.save()
                if (idx == i || cur?.id == it.optString("id")) result.text = summarize(j)
                Log.i(TAG, "GATHER onphone ${it.optString("id")} $j")
            }
        }
    }

    private fun summarize(j: JSONObject): String {
        if (j.has("error")) return "on-phone: error ${j.getString("error").take(60)}"
        val c = if (j.has("card")) "card ${j.getInt("card")}% inl ${j.getInt("inliers")} " + (if (j.optBoolean("accept")) "ACCEPT ${j.optString("token")}" else "reject") + " z %.1f".format(j.optDouble("z", 0.0)) else "no card (inl ${j.optInt("inliers")})"
        return "on-phone: $c  QR " + (if (j.isNull("qr")) "-" else j.getString("qr").takeLast(20))
    }

    /** Shared by the shot callback and the GATHER_TEST hook. */
    fun onPhoneResult(f: File): JSONObject {
        val m = org.opencv.imgcodecs.Imgcodecs.imread(f.absolutePath, org.opencv.imgcodecs.Imgcodecs.IMREAD_GRAYSCALE)
        val j = JSONObject()
        if (m.empty()) return j.put("error", "cannot read ${f.name}")
        try {
            val id = demo.identify(m)
            if (id != null) {
                j.put("inliers", id.inliers).put("inliers_by_card", JSONObject(id.byCard.mapKeys { it.key.toString() })).put("match_ms", id.msMatch)
                val c = id.card
                if (c != null && id.inliers >= DemoStego.MIN_INLIERS) {
                    j.put("card", c.pct)
                    id.hCard?.let { j.put("radius", demo.radius(it, m.cols(), m.rows())) }
                    val o = demo.decodeIdent(m, id)
                    if (o != null) j.put("token", o.token).put("accept", o.accept).put("crc_ok", o.crcOk).put("tag_ok", o.tagOk).put("z", o.z)
                        .put("zs", JSONArray(o.zs.toList())).put("gains", JSONArray(o.gains.toList())).put("reg", o.regScore).put("wide", o.wide).put("core_ms", o.msTotal)
                } else j.put("best_card", c?.pct ?: JSONObject.NULL)
            }
            val t0 = System.nanoTime()
            val qr = try { qrOnce.scanBlocking(m) } catch (e: Throwable) { Log.w(TAG, "QR one-shot failed: $e"); null }
            j.put("qr", qr ?: JSONObject.NULL).put("qr_ms", (System.nanoTime() - t0) / 1e6)
        } finally { m.release() }
        return j
    }

    /** Test hook (broadcast GATHER_TEST --es file path): the same on-phone result for a stored frame; logs `GATHER_TEST {json}`. */
    fun testFile(path: String) {
        resultExec.execute {
            if (!LiveAnalyzer.ensureCv()) { Log.e(TAG, "GATHER_TEST cv init failed"); return@execute }
            val j = try { onPhoneResult(File(path)) } catch (e: Throwable) { JSONObject().put("error", e.toString()) }
            Log.i(TAG, "GATHER_TEST $j")
        }
    }

    private fun beep(long: Boolean) {
        try { tone?.startTone(if (long) ToneGenerator.TONE_PROP_ACK else ToneGenerator.TONE_PROP_BEEP, if (long) 300 else 120) } catch (_: Exception) {}
    }

    private fun speak(t: String) { if (ttsOk && t.isNotEmpty()) try { tts?.speak(t, TextToSpeech.QUEUE_FLUSH, null, "g") } catch (_: Exception) {} }
}
