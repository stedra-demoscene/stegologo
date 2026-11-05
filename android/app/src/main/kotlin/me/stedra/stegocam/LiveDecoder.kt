package me.stedra.stegocam

import android.content.Context
import android.util.Log
import org.opencv.core.Mat
import java.util.concurrent.Executors
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Tracks detected blocks across frames and decodes them on ONE background thread via the shared C++ core (StegoCore, JNI).
 *  - a Track per block (centroid association); crops of unresolved blocks are cloned into a short rolling buffer,
 *  - every try is a single-frame core decode of the freshest crop (asset chosen from block size / glyph size; wide windows on poor registration),
 *  - a rejected block is retried with exponential back-off (BACKOFF_MS * 2^n, capped at BACKOFF_MAX_MS) measured from the end of
 *    the previous decode, and at most one decode is in flight, so decode CPU is bounded (a control block costs ~10 decodes/min),
 *  - the decode thread is pinned to the prime cores (cpu6/7 on SM8750) when `pin` is on; falls back silently if denied.
 */
class LiveDecoder(ctx: Context) {
    class View(val hex: String?, val colour: Int, val z: Double, val latencyMs: Double, val ttaMs: Long, val stable: Boolean,
               val mode: String, val nFrames: Int, val id: Int, val pendingNote: String)

    private class Track(val id: Int, var x0: Int, var y0: Int, var x1: Int, var y1: Int, val born: Long) {
        var lastSeen = born
        val buf = ArrayList<Mat>()                    // newest last, unresolved tracks only
        var lastBuf = 0L
        var inFlight = false
        var attempts = 0
        var nextTry = 0L                              // back-off: not before this time
        var res: CoreResult? = null
        var mode = ""
        var latency = 0.0
        var nFrames = 1
        var hex: String? = null                       // accepted payload
        var acceptZ = 0.0
        var firstAccept = 0L
        var accY: DoubleArray? = null                 // accumulated chip statistics over attempts (inverse-variance weighted)
        var accW: DoubleArray? = null
        var accN = 0
        var asset = -1                                // StegoCore asset handle (chosen from the block's size relative to its glyph size)
        var g = 0.0                                   // glyph size estimate (px) and block long side / g: sheet-2 blocks ~5.6, the 128-bit block ~11
        var ratio = 0.0
        var wideUsed = 0
        var cand: String? = null                      // accepted by the accumulated stat once; final after a repeat or z > Z_FINAL
        val stable get() = hex != null                // one scan: an accepted block is final
        val cx get() = (x0 + x1) / 2f
        val cy get() = (y0 + y1) / 2f
        val size get() = max(x1 - x0, y1 - y0)
        fun releaseBuf() { buf.forEach { it.release() }; buf.clear() }
    }

    companion object {
        private const val TAG = "StegoCam"
        const val NCOMB = 1                    // crops kept per track: only the freshest (no multi-frame escalation: z5 ~ 1.04 z1 on real data)
        const val CROP_MARGIN = 0.04           // snug crop (eval_sheet4); 8% made the registration of the big block fail
        const val RATIO_LARGE = 8.0            // block long side / glyph size above this: the 128-bit block asset; below RATIO_SMALL: sheet 2; between: try both
        const val RATIO_SMALL = 6.8
        const val WIDE_BELOW_REG = 0.6         // a failed decode with a registration score below this is retried with the wide windows (every 4th attempt)
        const val Z_FINAL = 6.0                // a first accumulated accept with z above this is final at once; otherwise it must repeat
        const val BUF_SPACING_MS = 60L
        const val BACKOFF_MS = 150L            // after the 1st rejection; doubles per rejection
        const val BACKOFF_MAX_MS = 2000L
        const val LOST_MS = 1500L
        /** 0 = no affinity, 1 = prime cores (the two fastest cpus). Settable via the LIVE broadcast (--ez pin). */
        @Volatile var pin = true
        /** debug: skip the decode_multi stage (single-frame only) */
        @Volatile var multi = true
        /** coarser fit for the big block asset (LIVE --ez fast false to disable) */
        @Volatile var fastBig = true
    }

    private val lock = Any()
    private val tracks = ArrayList<Track>()
    private var nextId = 1
    @Volatile private var generation = 0
    @Volatile var liveOnMs = System.currentTimeMillis()
        private set
    @Volatile var enabled = true
    private var inFlight = false
    private var appliedPin: Boolean? = null
    private val primeMask: Long by lazy { primeCpuMask() }

    private val pool = Executors.newSingleThreadExecutor { r ->
        Thread({ android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND - 2); r.run() }, "live-decode")
    }

    init { StegoCore.init(ctx) }

    /** Events also go to files/live_events.log (logcat's chatty filter drops bursts right after start-up). Truncated by reset(). */
    private val eventFile = java.io.File(ctx.getExternalFilesDir(null), "live_events.log")
    private fun event(line: String) { try { eventFile.appendText(line + "\n") } catch (e: Throwable) { } }

    fun reset() {
        synchronized(lock) {
            generation++
            tracks.forEach { it.releaseBuf() }; tracks.clear(); nextId = 1
            liveOnMs = System.currentTimeMillis()
            try { eventFile.writeText("") } catch (e: Throwable) { }
            event("LIVE_ON ${liveOnMs}")
        }
    }

    fun shutdown() { pool.shutdown() }

    /** cpus whose cpuinfo_max_freq equals the maximum (prime cluster); fallback: the top two cpus. */
    private fun primeCpuMask(): Long {
        val n = Runtime.getRuntime().availableProcessors()
        try {
            val f = LongArray(n) { java.io.File("/sys/devices/system/cpu/cpu$it/cpufreq/cpuinfo_max_freq").readText().trim().toLong() }
            val mx = f.max()
            var m = 0L; for (i in 0 until n) if (f[i] == mx) m = m or (1L shl i)
            if (m != 0L && m != (1L shl n) - 1) return m
        } catch (e: Throwable) { }
        return if (n >= 8) 0xC0L else 0L
    }

    private fun applyAffinity() {
        val want = pin
        if (appliedPin == want) return
        appliedPin = want
        val tid = android.os.Process.myTid()
        val rc = StegoCore.setAffinity(tid, if (want) primeMask else 0L)
        Log.i(TAG, "AFFINITY tid=$tid pin=$want mask=0x%x rc=$rc now=0x%x".format(if (want) primeMask else 0L, StegoCore.affinity(tid)))
    }

    /**
     * Analysis thread: associate [blocks] (sensor px) with tracks, buffer crops, submit at most one job, return one View per block (null = pending).
     * [full] is the camera frame (Y plane); crops are cloned here, so the worker never touches camera memory.
     */
    fun update(full: Mat, blocks: List<BlockDetector.Block>): Pair<List<View?>, String> {
        val now = System.currentTimeMillis()
        val w = full.cols(); val h = full.rows()
        val views = ArrayList<View?>()
        synchronized(lock) {
            // --- association (greedy, nearest centroid first)
            val matched = arrayOfNulls<Track>(blocks.size)
            val used = HashSet<Track>()
            val pairs = ArrayList<Triple<Double, Int, Track>>()
            for ((i, b) in blocks.withIndex()) {
                val bx = (b.x0 + b.x1) / 2f; val by = (b.y0 + b.y1) / 2f; val bs = max(b.x1 - b.x0, b.y1 - b.y0)
                for (t in tracks) {
                    val d = hypot((bx - t.cx).toDouble(), (by - t.cy).toDouble())
                    if (d < 0.35 * min(bs, t.size)) pairs.add(Triple(d, i, t))
                }
            }
            pairs.sortBy { it.first }
            for ((_, i, t) in pairs) if (matched[i] == null && t !in used) { matched[i] = t; used.add(t) }
            for ((i, b) in blocks.withIndex()) {
                var t = matched[i]
                if (t == null) { t = Track(nextId++, b.x0, b.y0, b.x1, b.y1, now); tracks.add(t); matched[i] = t }
                else { t.x0 = b.x0; t.y0 = b.y0; t.x1 = b.x1; t.y1 = b.y1 }
                t.lastSeen = now; t.g = b.g; t.ratio = if (b.g > 0) max(b.w, b.h) / b.g else 0.0
            }
            val it = tracks.iterator()
            while (it.hasNext()) { val t = it.next(); if (now - t.lastSeen > LOST_MS) { t.releaseBuf(); it.remove() } }

            if (enabled) {
                // --- buffer crops of unresolved tracks, only in the window before their next try (back-off = no work at all)
                for ((i, b) in blocks.withIndex()) {
                    val t = matched[i]!!
                    if (t.stable || now - t.lastBuf < BUF_SPACING_MS) continue
                    if (now < t.nextTry - (NCOMB + 1) * BUF_SPACING_MS) continue
                    val cb = BlockDetector.cropBox(b, w, h, CROP_MARGIN)
                    t.buf.add(full.submat(cb[1], cb[3], cb[0], cb[2]).clone())
                    t.lastBuf = now
                    while (t.buf.size > NCOMB) t.buf.removeAt(0).release()
                }
                // --- at most one decode in flight: fresh blocks first, then the longest-waiting due block
                if (!inFlight) {
                    var best: Track? = null
                    for (i in blocks.indices) {
                        val t = matched[i]!!
                        if (t.stable || t.inFlight || now < t.nextTry || t.buf.isEmpty()) continue
                        if (best == null || t.attempts < best.attempts || (t.attempts == best.attempts && t.nextTry < best.nextTry)) best = t
                    }
                    if (best != null) submit(best, now)
                }
            }
            for (i in blocks.indices) views.add(viewOf(matched[i]!!))
        }
        return Pair(views, summary())
    }

    private fun summary(): String = synchronized(lock) {
        val ts = tracks.sortedWith(compareBy({ Math.round(it.cy / 450f) }, { it.x0 }))
        val acc = ts.count { it.hex != null }
        "decode: $acc/${ts.size} accepted  " + ts.filter { it.hex != null }.joinToString("  ") { "${it.hex} ${"%.1f".format((it.firstAccept - it.born) / 1000.0)}s" }
    }

    private fun viewOf(t: Track): View? {
        val r = t.res
        if (t.hex != null) return View(t.hex, 0, t.acceptZ, t.latency, t.firstAccept - t.born, true, t.mode, t.nFrames, t.id, "")
        if (r == null) return null
        return View(r.hex, if (r.crcOk) 1 else 2, r.zCal, t.latency, -1, false, t.mode, t.nFrames, t.id, "")
    }

    private fun assetsFor(t: Track): IntArray {
        if (StegoCore.nAssets < 2) return intArrayOf(0)
        if (t.asset >= 0) return intArrayOf(t.asset)
        return when {
            t.ratio <= 0 -> intArrayOf(0, 1)
            t.ratio >= RATIO_LARGE -> intArrayOf(1)
            t.ratio <= RATIO_SMALL -> intArrayOf(0)
            else -> intArrayOf(0, 1)                       // ambiguous: decode against both once, keep the one that registers
        }
    }

    // called under lock
    private fun submit(t: Track, now: Long) {
        val frame: Mat = t.buf.last().also { t.buf.removeAt(t.buf.size - 1); t.releaseBuf() }
        t.inFlight = true; inFlight = true
        val gen = generation
        val t0 = System.nanoTime()
        val cand = assetsFor(t)
        val tryWide = t.attempts % 4 == 0
        try {
            pool.execute {
                var r: CoreResult? = null
                var ai = cand[0]; var wide = false
                try {
                    applyAffinity()
                    var bestKey = -1e9
                    for (a in cand) {
                        val fast = a == 1 && fastBig
                        var x = StegoCore.decode(frame, a, false, fast) ?: continue
                        var w = false
                        if (!x.accept && x.regScore < WIDE_BELOW_REG && tryWide) {
                            val y = StegoCore.decode(frame, a, true, fast)
                            if (y != null && (y.accept || y.regScore > x.regScore + 0.02)) { x = y; w = true }
                        }
                        val key = (if (x.accept) 100.0 + x.zCal else 0.0) + x.regScore
                        if (key > bestKey) { bestKey = key; r = x; ai = a; wide = w }
                    }
                } catch (e: Throwable) { Log.e(TAG, "live decode failed", e) }
                finally { frame.release() }
                finish(t, gen, r, if (wide) 2 else 1, (System.nanoTime() - t0) / 1e6, ai)
            }
        } catch (e: Exception) { t.inFlight = false; inFlight = false; frame.release() }
    }

    private fun finish(t: Track, gen: Int, r0: CoreResult?, used: Int, lat: Double, ai: Int) {
        val now = System.currentTimeMillis()
        var event = ""
        var r = r0
        var accZ = 0.0; var accN = 0
        synchronized(lock) {
            inFlight = false
            if (gen != generation) return
            t.inFlight = false; t.attempts++
            if (r0 != null && t.asset < 0 && ai >= 0 && (r0.accept || r0.regScore > 0.5 || assetsFor(t).size == 1)) t.asset = ai
            if (r0 != null && t.asset >= 0 && t.asset == ai && r0.pilotS2 > 0 && r0.chipY.isNotEmpty() && t.hex == null) {
                if (t.accY != null && t.accY!!.size != r0.chipY.size) { t.accY = null; t.accW = null; t.accN = 0; t.cand = null }
                // a clearly different payload (strong on its own) = a different block under this track: restart the accumulation
                if (r0.accept && t.cand != null && r0.hex != t.cand) { t.accY = null; t.accW = null; t.accN = 0; t.cand = null }
                val a = 1.0 / r0.pilotS2
                if (t.accY == null) { t.accY = DoubleArray(r0.chipY.size); t.accW = DoubleArray(r0.chipY.size); t.accN = 0 }
                for (i in r0.chipY.indices) { t.accY!![i] += a * r0.chipY[i]; t.accW!![i] += a * r0.chipW[i] }
                t.accN++
                r = StegoCore.finalize(t.accY!!, t.accW!!, t.asset) ?: r0
                accZ = r!!.zCal; accN = t.accN
            }
            val rr = r
            if (rr != null) {
                t.res = rr; t.latency = lat; t.nFrames = used; t.mode = (if (used > 1) "wide" else "1f") + (if (t.accN > 1) "+${t.accN}" else "")
                if (rr.accept && t.hex == null) {
                    if (rr.zCal > Z_FINAL || t.cand == rr.hex) {
                        t.hex = rr.hex; t.acceptZ = rr.zCal; t.firstAccept = now
                        event = " FIRST_ACCEPT tta_ms=${now - t.born} since_live_ms=${now - liveOnMs}"
                    } else t.cand = rr.hex
                } else if (t.hex == null) t.cand = null
            }
            if (t.hex != null) { t.accY = null; t.accW = null }
            else t.nextTry = now + (if (t.cand != null) BACKOFF_MS else min(BACKOFF_MAX_MS, BACKOFF_MS shl min(t.attempts - 1, 6)))
        }
        val x = r0 ?: return
        event("BLOCK t=${now - liveOnMs} id=${t.id} asset=${t.asset} hex=${x.hex} acc=${x.accept} z=%.2f accHex=${r?.hex} accZ=%.2f accRaw=%.2f accN=$accN final=${t.hex != null} mode=${t.mode} ms=%.0f core=%.1f try=${t.attempts}$event".format(x.z, accZ, r?.z ?: 0.0, lat, x.totalMs))
        Log.i(TAG, "BLOCK id=${t.id} hex=%s acc=%b z=%.2f accZ=%.2f accN=%d final=%b mode=%s total_ms=%.0f core_ms=%.1f %s try=%d%s".format(
            x.hex, x.accept, x.z, accZ, accN, t.hex != null, t.mode, lat, x.totalMs, x.stages, t.attempts, event))
    }
}
