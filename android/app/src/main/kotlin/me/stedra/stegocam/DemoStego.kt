package me.stedra.stegocam

import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.util.concurrent.Executors

/** The one place that knows where a demo token points. */
object DemoConfig {
    const val URL_PREFIX = "https://go.example/"   // fallback only; demo_cards.json "url_prefix" is what the scanner uses
    @Volatile var prefix: String = URL_PREFIX
}

/** Token (6 base-62 chars) -> link for demo hits; everything else goes to the stand-in resolver. */
class DemoResolver : PayloadResolver {
    private val fallback = StandInResolver()
    override fun resolve(hit: StegoHit): String? =
        if (hit.kind.startsWith("demo")) DemoConfig.prefix + hit.payload else fallback.resolve(hit)
}

/**
 * Demo cards (QR progressively hidden under its own design): the WHOLE card is one fixed non-periodic reference, so no mark detector and no OCR:
 *  1. ORB on frame/4 against each card's reference render (QR modules included) -> homography by MAGSAC (OpenCV USAC); best inlier count = card identity.
 *  2. The seed H goes to the core: edge registration of the whole design, then the three regions (text / logo / text) are decoded under that one H and
 *     their soft values are summed before the Golay decode (core/src/demo_decode.cpp). accept = CRC-16 + tag + combined z > 8 (core default).
 * Runs on its own thread (one frame in flight; the analysis thread only offers a copy).
 */
class DemoStego(private val ctx: Context) : StegoSource {
    override val name = "demo"

    class Card(val pct: Int, val refPts: Array<Point>, val refDesc: Mat, val compName: String, val regionNames: List<String>,
                       val shifts: DoubleArray, val cardToComp: DoubleArray, val refPpm: Double) {
        var comp = -1; var regions = IntArray(0); var loaded = false
    }

    companion object {
        private const val TAG = "StegoCam"
        const val DS = 4
        const val MIN_INLIERS = 25
        const val RETRY_MS = 120L
    }

    private val lock = Any()
    private var cards: List<Card>? = null
    private var orb: ORB? = null
    private var matcher: BFMatcher? = null
    private var busy = false
    private var gen = 0
    private var hit: StegoHit? = null
    private var startMs = System.currentTimeMillis()
    private var nextTry = 0L
    private var affinityDone = false
    /** last attempt, for the reveal panel / logs */
    @Volatile var lastInfo = ""
    private val pool = Executors.newSingleThreadExecutor { r -> Thread({ android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND - 2); r.run() }, "demo-stego") }

    override fun onFrame(full: Mat): List<StegoHit> {
        val now = System.currentTimeMillis()
        val g: Int
        synchronized(lock) {
            hit?.let { return listOf(it) }
            if (busy || now < nextTry) return emptyList()
            busy = true; g = gen
        }
        val frame = full.clone()
        try { pool.execute { work(frame, g) } } catch (e: Exception) { synchronized(lock) { busy = false }; frame.release() }
        return emptyList()
    }

    override fun reset() { synchronized(lock) { gen++; hit = null; startMs = System.currentTimeMillis(); nextTry = 0L } }
    override fun shutdown() { pool.shutdown() }

    private fun mul(a: DoubleArray, b: DoubleArray) = DoubleArray(9) { i -> (0..2).sumOf { k -> a[(i / 3) * 3 + k] * b[k * 3 + i % 3] } }

    private fun loadCards() {
        if (cards != null) return
        if (!LiveAnalyzer.ensureCv()) return
        orb = ORB.create(1500, 1.2f, 8, 15, 0, 2, ORB.HARRIS_SCORE, 31, 15)
        val refOrb = ORB.create(3000, 1.2f, 8, 15, 0, 2, ORB.HARRIS_SCORE, 31, 15)
        matcher = BFMatcher.create(Core.NORM_HAMMING, false)
        val j = JSONObject(ctx.assets.open("demo_cards.json").bufferedReader().use { it.readText() })
        DemoConfig.prefix = j.optString("url_prefix", DemoConfig.URL_PREFIX).ifEmpty { DemoConfig.URL_PREFIX }
        val arr = j.getJSONArray("cards")
        val out = ArrayList<Card>()
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            val bytes = ctx.assets.open(o.getString("ref")).use { it.readBytes() }
            val img = Imgcodecs.imdecode(MatOfByte(*bytes), Imgcodecs.IMREAD_GRAYSCALE)
            val kp = MatOfKeyPoint(); val d = Mat()
            refOrb.detectAndCompute(img, Mat(), kp, d); img.release()
            val sh = o.getJSONArray("shifts"); val sd = DoubleArray(6) { sh.getJSONArray(it / 2).getDouble(it % 2) }
            val cc = o.getJSONArray("card_to_comp")
            val rn = o.getJSONArray("regions")
            out.add(Card(o.getInt("pct"), kp.toArray().map { it.pt }.toTypedArray(), d, o.getString("comp"), (0 until rn.length()).map { rn.getString(it) },
                sd, doubleArrayOf(cc.getDouble(0), cc.getDouble(1)), o.getDouble("ref_ppm")))
        }
        cards = out
        Log.i(TAG, "DEMO cards loaded: ${out.map { it.pct }}")
    }

    /** Result of the whole-card identification: best card (null if none matched), its inlier count, per-card inliers, card mm -> full px homography. */
    class Ident(val card: Card?, val inliers: Int, val byCard: Map<Int, Int>, val hCard: DoubleArray?, val seed: DoubleArray?, val msMatch: Double)

    /** Card centre in card mm (the design is centred on the plate, see make_demo_cards). */
    val cardCentreMm = doubleArrayOf(52.5, 70.0)

    /** Design centre in full-frame px under the card -> frame homography. */
    fun centrePx(h: DoubleArray): DoubleArray {
        val x = cardCentreMm[0]; val y = cardCentreMm[1]
        val q = h[6] * x + h[7] * y + h[8]
        return doubleArrayOf((h[0] * x + h[1] * y + h[2]) / q, (h[3] * x + h[4] * y + h[5]) / q)
    }
    fun ensureLoaded() { synchronized(this) { loadCards() } }

    /** ORB + MAGSAC of [frame] (full-res Y) against every card reference. Serialised: the ORB / matcher objects are shared. */
    @Synchronized fun identify(frame: Mat): Ident? {
        val t0 = System.nanoTime()
        loadCards()
        val cs = cards ?: return null
        val small = Mat()
        Imgproc.resize(frame, small, Size((frame.cols() / DS).toDouble(), (frame.rows() / DS).toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val kp = MatOfKeyPoint(); val desc = Mat()
        orb!!.detectAndCompute(small, Mat(), kp, desc); small.release()
        val pts = kp.toArray().map { it.pt }
        var best: Card? = null; var bestInl = 0; var bestH: DoubleArray? = null
        val by = LinkedHashMap<Int, Int>()
        if (desc.rows() >= 8) for (c in cs) {
            val knn = ArrayList<MatOfDMatch>()
            matcher!!.knnMatch(desc, c.refDesc, knn, 2)
            val src = ArrayList<Point>(); val dst = ArrayList<Point>()
            for (m in knn) { val a = m.toArray(); if (a.size == 2 && a[0].distance < 0.8f * a[1].distance) { src.add(c.refPts[a[0].trainIdx]); dst.add(pts[a[0].queryIdx]) } }
            by[c.pct] = 0
            if (src.size < 8) continue
            val mask = Mat()
            val h = Calib3d.findHomography(MatOfPoint2f(*src.toTypedArray()), MatOfPoint2f(*dst.toTypedArray()), Calib3d.USAC_MAGSAC, 3.0, mask)
            if (h.empty()) { mask.release(); continue }
            val inl = Core.countNonZero(mask); mask.release()
            by[c.pct] = inl
            if (inl > bestInl) { bestInl = inl; best = c; bestH = DoubleArray(9).also { h.get(0, 0, it) } }
            h.release()
        }
        desc.release()
        val tMatch = (System.nanoTime() - t0) / 1e6
        if (best == null || bestH == null) return Ident(null, bestInl, by, null, null, tMatch)
        // ref idx px -> small idx px (bestH); card mm -> ref idx: p*mm - 0.5; small idx -> frame px: (idx + 0.5) * DS
        val p = best.refPpm
        val mm2ref = doubleArrayOf(p, 0.0, -0.5, 0.0, p, -0.5, 0.0, 0.0, 1.0)
        val s2f = doubleArrayOf(DS.toDouble(), 0.0, 0.5 * DS, 0.0, DS.toDouble(), 0.5 * DS, 0.0, 0.0, 1.0)
        val hCard = mul(s2f, mul(bestH, mm2ref))
        val seed = mul(hCard, doubleArrayOf(1.0, 0.0, best.cardToComp[0], 0.0, 1.0, best.cardToComp[1], 0.0, 0.0, 1.0))
        return Ident(best, bestInl, by, hCard, seed, tMatch)
    }

    /** Core decode of the three regions under the identified pose; null if assets are missing. */
    fun decodeIdent(frame: Mat, id: Ident): DemoOutcome? {
        val best = id.card ?: return null
        val seed = id.seed ?: return null
        synchronized(this) {
            if (!best.loaded) {
                best.comp = StegoCore.loadAsset(ctx, best.compName)
                best.regions = best.regionNames.map { StegoCore.loadAsset(ctx, it) }.toIntArray()
                best.loaded = best.comp >= 0 && best.regions.all { it >= 0 }
            }
        }
        if (!best.loaded) { Log.w(TAG, "DEMO assets not loaded for card ${best.pct}: comp=${best.comp} regions=${best.regions.toList()}"); return null }
        return StegoCore.demoDecode(best.comp, best.regions, best.shifts, frame, seed, false)
    }

    /** Gather mode: pose of the card (page-like): tilt / distance / px per mm from the card homography, plus the design centre's radius in the frame. */
    fun pose(full: Mat, k: DoubleArray, expectedPct: Int): PoseEstimator.Pose? {
        val id = identify(full) ?: return null
        val c = id.card; val h = id.hCard
        if (c == null || h == null || id.inliers < MIN_INLIERS) return PoseEstimator.Pose(-1, expectedPct, 0, 0, IntArray(8), 0.0, 0.0, 0.0, 0.0, DoubleArray(9), id.msMatch.toFloat())
        val d = PoseEstimator.decompose(h, k)
        val rad = radius(h, full.cols(), full.rows())
        return PoseEstimator.Pose(c.pct, expectedPct, id.inliers, id.inliers, IntArray(8), d?.get(0) ?: 0.0, d?.get(1) ?: 0.0, d?.get(2) ?: 0.0, d?.get(3) ?: 0.0, h,
            id.msMatch.toFloat(), radius = rad)
    }

    /** Distance of the design centre from the image centre as a fraction of the half-diagonal. */
    fun radius(h: DoubleArray, w: Int, hh: Int): Double {
        val x = cardCentreMm[0]; val y = cardCentreMm[1]
        val q = h[6] * x + h[7] * y + h[8]
        val u = (h[0] * x + h[1] * y + h[2]) / q; val v = (h[3] * x + h[4] * y + h[5]) / q
        return Math.hypot(u - w / 2.0, v - hh / 2.0) / Math.hypot(w / 2.0, hh / 2.0)
    }

    private fun work(frame: Mat, g: Int) {
        var res: StegoHit? = null
        val t0 = System.nanoTime()
        var info = ""
        try {
            if (!affinityDone) { affinityDone = true; val m = StegoCore.primeCpuMask(); if (m != 0L) StegoCore.setAffinity(android.os.Process.myTid(), m) }
            val id = identify(frame)
            if (id != null) {
                val best = id.card
                if (best != null && id.inliers >= MIN_INLIERS) {
                    val o = decodeIdent(frame, id)
                    if (o != null) {
                        info = "card ${best.pct}%% inl=${id.inliers} match=%.0fms reg=%.2f z=%.1f zs=%s tok=${o.token} acc=${o.accept} core=%.0fms".format(id.msMatch, o.regScore, o.z, o.zs.joinToString("/") { "%.1f".format(it) }, o.msTotal)
                        if (o.accept) {
                            val c = o.crop
                            res = StegoHit(o.token, o.z, 0, intArrayOf(c[0], c[1], c[2], c[3]), "demo:${best.pct}", "demo card ${best.pct}%% z=%.1f  inliers ${id.inliers}  %.0f ms".format(o.z, (System.nanoTime() - t0) / 1e6))
                        }
                    }
                } else info = "no card (best inliers ${id.inliers}) %.0fms".format(id.msMatch)
            }
        } catch (e: Throwable) { Log.e(TAG, "demo stego failed", e) }
        finally { frame.release() }
        val now = System.currentTimeMillis()
        synchronized(lock) {
            busy = false
            if (g != gen) return
            lastInfo = info
            if (res != null) hit = res else nextTry = now + RETRY_MS
        }
        Log.i(TAG, "DEMO " + info + " total=" + "%.0f".format((System.nanoTime() - t0) / 1e6) + "ms")
    }
}
