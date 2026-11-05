package me.stedra.stegocam

import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.opencv.core.Mat
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

/**
 * One kind of mark the live path knows: geometry shared by one or more decode assets (one per key/region), plus detector hints.
 * Declared in assets/marks.json; a new mark candidate (e.g. the swirl) is added there, with its exported assets (tools/export_mark_assets.py):
 *   {"name": "swirl", "label": "swirl 40 mm", "assets": ["mark_swirl_a"], "sym": 3, "ppm": [8, 26]}
 * sym = n-fold rotational symmetry of the geometry: the detector returns sym rotation hypotheses, the decode (pilot z) picks the right one.
 */
class MarkSpec(val name: String, val label: String, val assetNames: List<String>, val sym: Int, val ppmLo: Double, val ppmHi: Double) {
    var det = -1
    var assets = IntArray(0)
    @Volatile var loaded = false
    var lastScore = 0.0                  // last detector score (priority of the next attempt)
    var lastTry = 0L
    var nextTry = 0L
    val exclusions = ArrayList<DoubleArray>()   // accepted marks of this spec in the current scan: (cx, cy, r) in frame px

    companion object {
        fun parse(ctx: Context): List<MarkSpec> = try {
            val txt = ctx.assets.open("marks.json").bufferedReader().use { it.readText() }
            val arr = JSONObject(txt).getJSONArray("marks")
            (0 until arr.length()).map {
                val o = arr.getJSONObject(it)
                val names = o.getJSONArray("assets").let { a -> (0 until a.length()).map { i -> a.getString(i) } }
                val ppm = o.optJSONArray("ppm")
                MarkSpec(o.getString("name"), o.optString("label", o.getString("name")), names, o.optInt("sym", 1), ppm?.getDouble(0) ?: 8.0, ppm?.getDouble(1) ?: 24.0)
            }
        } catch (e: Exception) { Log.w("StegoCam", "marks.json: $e"); emptyList() }
    }
}

/**
 * Live mark detection + decode on its own thread (C++ core: detector pose -> crop -> decode, see core/src/mark_decode.cpp).
 * The analysis thread only offers frames: when idle, a frame is copied (12 MB memcpy) and one spec is tried. Specs cycle (least recently
 * tried first, with a back-off when nothing is detected) but a spec that just detected a mark is retried at once. Accepted marks are
 * excluded from the next detection, so several marks of one kind (sheet-4 page 1) are found one after the other.
 */
class MarkLive(private val ctx: Context) {
    /** A detection: [accepted] = CRC ok and z over the threshold. box = crop box in sensor px. */
    class Hit(val spec: MarkSpec, val assetName: String, val hex: String, val z: Double, val score: Double, val accepted: Boolean,
              val box: IntArray, val ms: Double, val msDetect: Double, val tMs: Long, val hyp: Int, val ppm: Double, val rotDeg: Double, val t: Long)

    companion object {
        private const val TAG = "StegoCam"
        const val DETECT_OK = 0.7            // detector score above which something mark-like is there
        const val Z_ACCEPT = 5.0
        const val RETRY_MS = 350L            // spec back-off when nothing is detected
        const val TTL_MS = 1500L
    }

    @Volatile var enabled = false
    @Volatile var specs: List<MarkSpec> = emptyList()
    /** called on the mark thread for every newly accepted mark */
    @Volatile var onAccept: ((Hit) -> Unit)? = null
    private val lock = Any()
    private val accepted = ArrayList<Hit>()
    private val latest = HashMap<String, Hit>()
    private var busy = false
    private var gen = 0
    private var startMs = System.currentTimeMillis()
    private var affinityDone = false
    private val pool = Executors.newSingleThreadExecutor { r -> Thread({ android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND - 2); r.run() }, "mark-live") }

    fun loadSpecs() { if (specs.isEmpty()) specs = MarkSpec.parse(ctx) }

    fun reset() {
        synchronized(lock) {
            gen++; accepted.clear(); latest.clear(); startMs = System.currentTimeMillis()
            for (s in specs) { s.exclusions.clear(); s.lastScore = 0.0; s.nextTry = 0L }
        }
    }

    /** Hits to draw/report: all accepted marks + the latest non-accepted detections younger than [TTL_MS]. */
    fun current(): List<Hit> = synchronized(lock) {
        val now = System.currentTimeMillis()
        accepted.toList() + latest.values.filter { !it.accepted && now - it.t < TTL_MS }
    }

    fun shutdown() { pool.shutdown() }

    /** Analysis thread: called with every frame; cheap unless a worker slot is free. */
    fun offer(full: Mat) {
        if (!enabled || specs.isEmpty()) return
        val now = System.currentTimeMillis()
        val spec: MarkSpec
        val g: Int
        synchronized(lock) {
            if (busy) return
            // best spec: a recent detection first, otherwise the one waiting longest (after its back-off)
            var best: MarkSpec? = null; var bestKey = -1.0
            for (s in specs) {
                if (now < s.nextTry) continue
                val key = (if (s.lastScore > DETECT_OK) 1e7 else 0.0) + (now - s.lastTry)
                if (key > bestKey) { bestKey = key; best = s }
            }
            spec = best ?: return
            busy = true; g = gen
        }
        val frame = full.clone()
        try {
            pool.execute { work(spec, frame, g) }
        } catch (e: Exception) { synchronized(lock) { busy = false }; frame.release() }
    }

    private fun work(spec: MarkSpec, frame: Mat, g: Int) {
        var out: MarkOutcome? = null
        val t0 = System.currentTimeMillis()
        try {
            if (!affinityDone) {
                affinityDone = true
                val m = StegoCore.primeCpuMask()
                if (m != 0L) StegoCore.setAffinity(android.os.Process.myTid(), m)
            }
            if (!spec.loaded) {
                val hs = spec.assetNames.map { StegoCore.loadAsset(ctx, it) }.filter { it >= 0 }
                if (hs.isNotEmpty()) { spec.assets = hs.toIntArray(); spec.det = StegoCore.markInit(hs[0]); spec.loaded = spec.det >= 0 }
                Log.i(TAG, "MARK_SPEC ${spec.name} loaded=${spec.loaded} assets=${hs.size} in ${System.currentTimeMillis() - t0} ms")
            }
            if (spec.loaded) {
                val ex = synchronized(lock) { DoubleArray(spec.exclusions.size * 3) { spec.exclusions[it / 3][it % 3] } }
                out = StegoCore.decodeMark(spec.det, spec.assets, frame, spec.ppmLo, spec.ppmHi, spec.sym, 2 /* fast */, ex)
            }
        } catch (e: Throwable) { Log.e(TAG, "mark live failed", e) }
        finally { frame.release() }
        val now = System.currentTimeMillis()
        var hit: Hit? = null
        synchronized(lock) {
            busy = false
            if (g != gen) return
            spec.lastTry = now
            val o = out
            if (o == null || !o.found) { spec.lastScore = 0.0; spec.nextTry = now + RETRY_MS; return }
            spec.lastScore = o.score
            if (o.score < DETECT_OK || o.core == null) { spec.nextTry = now + RETRY_MS; return }
            val c = o.core
            val acc = c.accept && c.zCal >= Z_ACCEPT
            val name = if (o.asset in spec.assetNames.indices) spec.assetNames[o.asset] else "?"
            hit = Hit(spec, name, c.hex, c.zCal, o.score, acc, o.crop, o.msTotal, o.msDetect, now - startMs, o.hyp, o.ppm, o.rotDeg, now)
            if (acc) {
                accepted.add(hit!!)
                val r = 0.5 * max(o.crop[2] - o.crop[0], o.crop[3] - o.crop[1])
                spec.exclusions.add(doubleArrayOf(o.cx, o.cy, r * 0.8))
                latest.remove(spec.name)
            } else latest[spec.name] = hit!!
            spec.nextTry = if (acc) 0L else now
        }
        val h = hit ?: return
        Log.i(TAG, "MARK spec=${spec.name} asset=${h.assetName} hex=${h.hex} acc=${h.accepted} z=%.1f score=%.2f ppm=%.1f rot=%.0f hyp=${h.hyp} detect_ms=%.0f total_ms=%.0f".format(h.z, h.score, h.ppm, h.rotDeg, h.msDetect, h.ms))
        if (h.accepted) onAccept?.invoke(h)
    }
}
