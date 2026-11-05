package me.stedra.stegocam

import android.content.Context
import android.util.Log
import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.*

/**
 * Pose of the sheet relative to the camera from ORB matches against the bundled page previews (5 px/mm, A4 portrait, origin top-left).
 * H maps page mm -> full-resolution stream pixels; K = [fx, fy, cx, cy] at the stream resolution.
 * Tilt: camera z axis expressed in page coordinates a = (ax, ay, az): tiltLr = atan2(ax, az) (about the page's vertical axis),
 * tiltUd = atan2(ay, az) (about the horizontal axis); both 0 when looking straight down. Distance = depth along the optical axis to the page plane.
 */
object PoseEstimator {
    private const val TAG = "StegoCam"
    const val DS = 4
    const val REF_PXMM = 5.0
    const val PAGES = 7
    private const val RATIO = 0.8f
    private const val MIN_INLIERS = 12

    class Pose(
        val page: Int, val expected: Int, val inliers: Int, val matches: Int, val inliersByPage: IntArray,
        val tiltLr: Double, val tiltUd: Double, val distMm: Double, val pxPerMm: Double, val h: DoubleArray, val ms: Float,
        val t: Long = System.currentTimeMillis(), val radius: Double = Double.NaN
    ) {
        val found get() = inliers >= MIN_INLIERS
        val pageOk get() = page == expected
        fun toJson(): JSONObject = JSONObject().put("matched_page", page).put("expected_page", expected).put("inliers", inliers)
            .put("n_matches", matches).put("inliers_by_page", JSONArray(inliersByPage.toList()))
            .put("tilt_lr_deg", tiltLr).put("tilt_ud_deg", tiltUd).put("dist_mm", distMm).put("px_per_mm", pxPerMm)
            .put("H_mm_to_px", JSONArray(h.toList())).put("ms", ms.toDouble()).also { if (!radius.isNaN()) it.put("radius", radius) }
    }

    private class Ref(val desc: Mat, val pts: Array<Point>)

    private val refs = arrayOfNulls<Ref>(PAGES + 1)
    private var orb: ORB? = null
    private var matcher: BFMatcher? = null
    private var ready = false

    @Synchronized fun init(ctx: Context): Boolean {
        if (ready) return true
        if (!LiveAnalyzer.ensureCv()) return false
        orb = ORB.create(1500, 1.2f, 8, 15, 0, 2, ORB.HARRIS_SCORE, 31, 15)
        val refOrb = ORB.create(3000, 1.2f, 8, 15, 0, 2, ORB.HARRIS_SCORE, 31, 15)
        matcher = BFMatcher.create(Core.NORM_HAMMING, false)
        for (p in 1..PAGES) {
            val bytes = ctx.assets.open("pages/p$p.png").use { it.readBytes() }
            val img = Imgcodecs.imdecode(MatOfByte(*bytes), Imgcodecs.IMREAD_GRAYSCALE)
            val kp = MatOfKeyPoint(); val d = Mat()
            refOrb.detectAndCompute(img, Mat(), kp, d)
            refs[p] = Ref(d, kp.toArray().map { it.pt }.toTypedArray())
            Log.i(TAG, "POSE ref page $p: ${kp.rows()} keypoints")
            img.release()
        }
        ready = true
        return true
    }

    /** Intrinsics [fx, fy, cx, cy] for a stream of size fw x fh from the active array / calibration, as (values, source). */
    fun intrinsics(calib: FloatArray?, focalMm: Float?, physW: Float?, physH: Float?, arrW: Int, arrH: Int, fw: Int, fh: Int): Pair<DoubleArray, String> {
        var fx: Double; var fy: Double; var cx: Double; var cy: Double; val src: String
        if (calib != null && calib.size >= 5 && calib[0] > 0) {
            fx = calib[0].toDouble(); fy = calib[1].toDouble(); cx = calib[2].toDouble(); cy = calib[3].toDouble(); src = "LENS_INTRINSIC_CALIBRATION"
        } else {
            val f = focalMm?.toDouble() ?: 6.0
            fx = f / ((physW ?: 7.0f).toDouble() / arrW); fy = f / ((physH ?: 5.25f).toDouble() / arrH)
            cx = arrW / 2.0; cy = arrH / 2.0; src = "focal/pixel size"
        }
        val s = max(fw.toDouble() / arrW, fh.toDouble() / arrH)       // stream = centre crop of the array, scaled
        val ox = (arrW - fw / s) / 2; val oy = (arrH - fh / s) / 2
        return Pair(doubleArrayOf(fx * s, fy * s, (cx - ox) * s, (cy - oy) * s), src)
    }

    private class Match(val inl: Int, val n: Int, val h: DoubleArray?)

    private fun match(desc: Mat, pts: List<Point>, page: Int): Match {
        val ref = refs[page] ?: return Match(0, 0, null)
        val knn = ArrayList<MatOfDMatch>()
        matcher!!.knnMatch(desc, ref.desc, knn, 2)
        val src = ArrayList<Point>(); val dst = ArrayList<Point>()
        for (m in knn) {
            val a = m.toArray()
            if (a.size == 2 && a[0].distance < RATIO * a[1].distance) { src.add(ref.pts[a[0].trainIdx]); dst.add(pts[a[0].queryIdx]) }
        }
        if (src.size < 8) return Match(0, src.size, null)
        val mask = Mat()
        val h = Calib3d.findHomography(MatOfPoint2f(*src.toTypedArray()), MatOfPoint2f(*dst.toTypedArray()), Calib3d.RANSAC, 3.0, mask)
        val inl = if (h.empty()) 0 else Core.countNonZero(mask)
        mask.release()
        if (h.empty()) return Match(0, src.size, null)
        val hd = DoubleArray(9); h.get(0, 0, hd); h.release()
        return Match(inl, src.size, hd)
    }

    private fun mul(a: DoubleArray, b: DoubleArray) = DoubleArray(9) { i -> (0..2).sumOf { k -> a[(i / 3) * 3 + k] * b[k * 3 + i % 3] } }

    /**
     * [full] = Y plane (sensor orientation). [hint] = a recent H (page mm -> full px): the frame is first warped into the reference geometry with it
     * (ORB is not perspective invariant, this keeps strongly tilted marks matchable) and only the expected page is tried. Without a usable hint
     * (or if it fails) plain matching runs; [all] checks every page.
     */
    @Synchronized fun estimate(full: Mat, k: DoubleArray, expected: Int, all: Boolean, hint: DoubleArray? = null): Pose? {
        if (!ready) return null
        val t0 = System.nanoTime()
        val small = Mat()
        Imgproc.resize(full, small, Size((full.cols() / DS).toDouble(), (full.rows() / DS).toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val byPage = IntArray(PAGES + 1)
        var bestPage = -1; var best: Match? = null
        if (hint != null) {
            // Hd0: ref px -> small px
            val hd0 = mul(doubleArrayOf(1.0 / DS, 0.0, 0.0, 0.0, 1.0 / DS, 0.0, 0.0, 0.0, 1.0), mul(hint, doubleArrayOf(1 / REF_PXMM, 0.0, 0.0, 0.0, 1 / REF_PXMM, 0.0, 0.0, 0.0, 1.0)))
            val hm = Mat(3, 3, CvType.CV_64F); hm.put(0, 0, *hd0)
            val warped = Mat()
            Imgproc.warpPerspective(small, warped, hm, Size(1050.0, 1485.0), Imgproc.INTER_LINEAR or Imgproc.WARP_INVERSE_MAP, Core.BORDER_REPLICATE)
            hm.release()
            val kp = MatOfKeyPoint(); val d = Mat()
            trackOrb().detectAndCompute(warped, Mat(), kp, d); warped.release()
            if (d.rows() >= 8) {
                val m = match(d, kp.toArray().map { it.pt }, expected)
                if (m.h != null && m.inl >= 20) { best = Match(m.inl, m.n, mul(hd0, m.h)); bestPage = expected; byPage[expected] = m.inl }
            }
            d.release()
        }
        if (best == null) {
            val kp = MatOfKeyPoint(); val desc = Mat()
            orb!!.detectAndCompute(small, Mat(), kp, desc)
            val pts = kp.toArray().map { it.pt }
            if (desc.rows() >= 8) for (p in (if (all || hint != null) 1..PAGES else expected..expected)) {
                val m = match(desc, pts, p)
                byPage[p] = m.inl
                if (m.h != null && m.inl > (best?.inl ?: 0)) { best = m; bestPage = p }
            }
            desc.release()
        }
        small.release()
        val ms = (System.nanoTime() - t0) / 1e6f
        val b = best
        if (b == null || b.h == null) return Pose(-1, expected, 0, 0, byPage, 0.0, 0.0, 0.0, 0.0, DoubleArray(9), ms)
        val hf = toFull(b.h)
        val d = decompose(hf, k) ?: return Pose(bestPage, expected, b.inl, b.n, byPage, 0.0, 0.0, 0.0, 0.0, hf, ms)
        return Pose(bestPage, expected, b.inl, b.n, byPage, d[0], d[1], d[2], d[3], hf, ms)
    }

    private var trackOrbInst: ORB? = null
    private fun trackOrb() = trackOrbInst ?: ORB.create(2000, 1.2f, 8, 15, 0, 2, ORB.HARRIS_SCORE, 31, 15).also { trackOrbInst = it }

    /** H (page px at 5 px/mm -> downscaled px) to (page mm -> full-res px). */
    fun toFull(hd: DoubleArray): DoubleArray {
        val s = DS * REF_PXMM
        return doubleArrayOf(hd[0] * s, hd[1] * s, hd[2] * DS, hd[3] * s, hd[4] * s, hd[5] * DS, hd[6] * REF_PXMM, hd[7] * REF_PXMM, hd[8])
    }

    /** -> [tiltLr deg, tiltUd deg, distMm, pxPerMm] or null. Pure math (unit-testable). */
    fun decompose(h: DoubleArray, k: DoubleArray): DoubleArray? {
        val fx = k[0]; val fy = k[1]; val cx = k[2]; val cy = k[3]
        // G = K^-1 H
        fun col(j: Int) = doubleArrayOf((h[j] - cx * h[6 + j]) / fx, (h[3 + j] - cy * h[6 + j]) / fy, h[6 + j])
        val g1 = col(0); val g2 = col(1); val g3 = col(2)
        fun nrm(v: DoubleArray) = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        if (nrm(g1) < 1e-9 || nrm(g2) < 1e-9) return null
        var lam = 2.0 / (nrm(g1) + nrm(g2))
        if (g3[2] * lam < 0) lam = -lam
        val r1 = g1.map { it * lam }.toDoubleArray(); val r2 = g2.map { it * lam }.toDoubleArray(); val t = g3.map { it * lam }.toDoubleArray()
        val n1 = nrm(r1); val a = DoubleArray(3) { r1[it] / n1 }
        val dot = a[0] * r2[0] + a[1] * r2[1] + a[2] * r2[2]
        val b0 = DoubleArray(3) { r2[it] - dot * a[it] }; val nb = nrm(b0); val b = DoubleArray(3) { b0[it] / nb }
        val c = doubleArrayOf(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
        if (c[2] <= 1e-3) return null                                          // looking at the back / grazing
        // camera z axis in page coords = third row of R = (a_z, b_z, c_z)
        val ax = a[2]; val ay = b[2]; val az = c[2]
        val lr = Math.toDegrees(atan2(ax, az)); val ud = Math.toDegrees(atan2(ay, az))
        val dist = (c[0] * t[0] + c[1] * t[1] + c[2] * t[2]) / az
        // px/mm at the page point seen at the image centre
        val det = h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) + h[2] * (h[3] * h[7] - h[4] * h[6])
        if (abs(det) < 1e-12) return null
        // p = H^-1 (cx, cy, 1) via adjugate
        val inv = doubleArrayOf(
            h[4] * h[8] - h[5] * h[7], h[2] * h[7] - h[1] * h[8], h[1] * h[5] - h[2] * h[4],
            h[5] * h[6] - h[3] * h[8], h[0] * h[8] - h[2] * h[6], h[2] * h[3] - h[0] * h[5],
            h[3] * h[7] - h[4] * h[6], h[1] * h[6] - h[0] * h[7], h[0] * h[4] - h[1] * h[3])
        val q = doubleArrayOf(inv[0] * cx + inv[1] * cy + inv[2], inv[3] * cx + inv[4] * cy + inv[5], inv[6] * cx + inv[7] * cy + inv[8])
        val px = q[0] / q[2]; val py = q[1] / q[2]
        val w = h[6] * px + h[7] * py + h[8]
        val u = (h[0] * px + h[1] * py + h[2]) / w; val v = (h[3] * px + h[4] * py + h[5]) / w
        val j00 = (h[0] - u * h[6]) / w; val j01 = (h[1] - u * h[7]) / w
        val j10 = (h[3] - v * h[6]) / w; val j11 = (h[4] - v * h[7]) / w
        val ppm = sqrt(abs(j00 * j11 - j01 * j10))
        if (lr.isNaN() || dist.isNaN()) return null
        return doubleArrayOf(lr, ud, dist, ppm)
    }

    /** Offline test entry: broadcast POSE_TEST --es path <png/pgm> --ei page N --es k fx,fy,cx,cy. Logs `StegoCam POSE_TEST {json}`. */
    fun testFile(ctx: Context, path: String?, page: Int, k: String?, hint: String? = null) {
        Thread {
            try {
                if (path == null || k == null) { Log.e(TAG, "POSE_TEST {\"error\":\"missing path/k\"}"); return@Thread }
                init(ctx)
                val img = Imgcodecs.imread(path, Imgcodecs.IMREAD_GRAYSCALE)
                if (img.empty()) { Log.e(TAG, "POSE_TEST {\"error\":\"cannot read $path\"}"); return@Thread }
                val kk = k.split(",").map { it.toDouble() }.toDoubleArray()
                val hh = hint?.split(",")?.map { it.toDouble() }?.toDoubleArray()
                var p = estimate(img, kk, page, true, hh)
                val t0 = System.nanoTime(); repeat(3) { p = estimate(img, kk, page, false, hh) }
                val ms1 = (System.nanoTime() - t0) / 3e6
                val j = (p ?: Pose(-1, page, 0, 0, IntArray(8), 0.0, 0.0, 0.0, 0.0, DoubleArray(9), 0f)).toJson()
                j.put("file", File(path).name).put("ms_expected_only", ms1)
                Log.i(TAG, "POSE_TEST $j")
                img.release()
            } catch (e: Throwable) { Log.e(TAG, "POSE_TEST failed", e) }
        }.start()
    }
}
