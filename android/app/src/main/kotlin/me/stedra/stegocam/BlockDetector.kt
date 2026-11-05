package me.stedra.stegocam

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.*

/**
 * Compact text block detector, port of tools/decode_capture.py detect_blocks / _detect_blocks_one (single image).
 * Everything is derived from the image: downsample from a glyph-size estimate (~target_g px per glyph), background
 * window scaled with the glyph size, line-merging close tried at 1.2/1.5/1.0/1.8 x glyph size (most non-clipped
 * blocks wins), size-consistency window 0.5-1.8 x median. Constants are the Python ones.
 */
object BlockDetector {
    class Block(val x0: Int, val y0: Int, val x1: Int, val y1: Int, val clipped: Boolean, val g: Double = 0.0) {
        val w get() = x1 - x0
        val h get() = y1 - y0
    }

    private const val TARGET_G = 20.0

    private class Ink(val s: Mat, val ink: Mat, val sw: Int, val sh: Int) { fun release() { s.release(); ink.release() } }

    private fun inkMask(full: Mat, ds: Int, win: Int?): Ink {
        val s = Mat(); val small = Mat(); val bg = Mat(); val bgFull = Mat(); val thr = Mat(); val ink = Mat()
        Imgproc.resize(full, s, Size(), 1.0 / ds, 1.0 / ds, Imgproc.INTER_AREA)
        val sw = s.cols(); val sh = s.rows()
        Imgproc.resize(s, small, Size(), 0.25, 0.25, Imgproc.INTER_AREA)
        val w = win ?: max(9, min(small.cols(), small.rows()) / 12)
        Imgproc.medianBlur(small, bg, w or 1)
        Imgproc.resize(bg, bgFull, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        Core.multiply(bgFull, Scalar(0.6), thr)
        Core.compare(s, thr, ink, Core.CMP_LT)
        val cross = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, Size(3.0, 3.0))
        Imgproc.morphologyEx(ink, ink, Imgproc.MORPH_OPEN, cross)
        for (m in listOf(small, bg, bgFull, thr, cross)) m.release()
        return Ink(s, ink, sw, sh)
    }

    private class Glyphs(val lab: Mat, val n: Int, val st: IntArray, val keep: BooleanArray, val g: Double)

    private fun median(x: List<Int>): Double {
        val s = x.sorted(); val m = s.size
        return if (m % 2 == 1) s[m / 2].toDouble() else 0.5 * (s[m / 2 - 1] + s[m / 2])
    }

    private fun percentile(sorted: List<Int>, q: Double): Double {   // numpy linear
        val pos = q / 100 * (sorted.size - 1); val lo = floor(pos).toInt(); val hi = min(lo + 1, sorted.size - 1)
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)
    }

    private fun glyphs(ink: Mat, sw: Int, sh: Int): Glyphs {
        val lab = Mat(); val st = Mat(); val cen = Mat()
        val n = Imgproc.connectedComponentsWithStats(ink, lab, st, cen, 4, CvType.CV_32S)
        cen.release()
        val si = IntArray(n * 5); if (n > 0) st.get(0, 0, si); st.release()
        if (n <= 1) return Glyphs(lab, n, si, BooleanArray(n), 10.0)
        val areas = (1 until n).map { si[it * 5 + 4] }.sorted()
        val thr = 0.25 * percentile(areas, 90.0)
        val keep = BooleanArray(n) { it > 0 && si[it * 5 + 4] >= thr }
        val sizes = (1 until n).filter { keep[it] && max(si[it * 5 + 2], si[it * 5 + 3]) < 0.2 * min(sw, sh) }
            .map { max(si[it * 5 + 2], si[it * 5 + 3]) }
        return Glyphs(lab, n, si, keep, if (sizes.isEmpty()) 10.0 else median(sizes))
    }

    private fun labelsToMask(lab: Mat, lut: ByteArray, dst: Mat) {
        val rows = lab.rows(); val cols = lab.cols()
        val li = IntArray(cols); val out = ByteArray(cols)
        for (r in 0 until rows) {
            lab.get(r, 0, li)
            for (c in 0 until cols) out[c] = lut[li[c]]
            dst.put(r, 0, out)
        }
    }

    fun detect(full: Mat): List<Block> {
        val W = full.cols(); val H = full.rows()
        var ds = 4
        var ink = inkMask(full, ds, null)
        var gl = glyphs(ink.ink, ink.sw, ink.sh)
        val ds2 = Math.rint(ds * gl.g / TARGET_G).toInt().coerceIn(2, 12)
        if (ds2.toDouble() / ds !in 0.75..1.34) {
            gl.lab.release(); ink.release(); ds = ds2
            ink = inkMask(full, ds, null); gl = glyphs(ink.ink, ink.sw, ink.sh)
        }
        val g0 = gl.g
        gl.lab.release(); ink.release()
        ink = inkMask(full, ds, max(9, (3 * g0 / 4).toInt()))
        gl = glyphs(ink.ink, ink.sw, ink.sh)
        val sw = ink.sw; val sh = ink.sh; val g = gl.g
        var best: Triple<Pair<Int, Int>, List<Block>, Unit>? = null
        if (gl.n > 1) {
            val lut = ByteArray(gl.n) { if (gl.keep[it]) -1 else 0 }
            val ink2 = Mat(sh, sw, CvType.CV_8UC1); labelsToMask(gl.lab, lut, ink2)
            for (mult in doubleArrayOf(1.2, 1.5, 1.0, 1.8)) {
                val rows = mergeBlocks(ink2, ds, W, H, sw, sh, g, mult)
                val sc = Pair(rows.count { !it.clipped }, rows.size)
                if (best == null || sc.first > best.first.first || (sc.first == best.first.first && sc.second > best.first.second))
                    best = Triple(sc, rows, Unit)
            }
            ink2.release()
        }
        gl.lab.release(); ink.release()
        var rows = best?.second ?: emptyList()
        if (rows.isNotEmpty()) {
            val L = rows.map { max(it.x1 - it.x0, it.y1 - it.y0) }
            val mx = L.max()
            val ref = median(L.filter { it >= 0.5 * mx })
            rows = rows.filterIndexed { i, _ -> L[i] >= 0.5 * ref && L[i] <= 1.8 * ref }
        }
        return rows
    }

    private fun mergeBlocks(ink2: Mat, ds: Int, W: Int, H: Int, sw: Int, sh: Int, g: Double, mult: Double): List<Block> {
        val k = Math.rint(mult * g).toInt() or 1
        val padded = Mat()
        Core.copyMakeBorder(ink2, padded, k, k, k, k, Core.BORDER_CONSTANT, Scalar(0.0))
        val kc = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(k.toDouble(), k.toDouble()))
        Imgproc.morphologyEx(padded, padded, Imgproc.MORPH_CLOSE, kc)
        val closed = padded.submat(k, k + sh, k, k + sw).clone()
        val lab2 = Mat(); val st2 = Mat(); val cen2 = Mat()
        val n2 = Imgproc.connectedComponentsWithStats(closed, lab2, st2, cen2, 4, CvType.CV_32S)
        val s2 = IntArray(n2 * 5); st2.get(0, 0, s2)
        for (m in listOf(padded, kc, closed, lab2, st2, cen2)) m.release()
        val out = ArrayList<Block>()
        for (i in 1 until n2) {
            val x0 = s2[i * 5]; val y0 = s2[i * 5 + 1]; val bw = s2[i * 5 + 2]; val bh = s2[i * 5 + 3]; val a = s2[i * 5 + 4]
            if (a < 50) continue
            val long = max(bw, bh); val short = min(bw, bh)
            val fill = a.toDouble() / (bw.toDouble() * bh)
            if (long > 2.2 * short || long < 3 * g || long > 0.7 * min(sh, sw) || fill < 0.3) continue
            val x1 = x0 + bw; val y1 = y0 + bh; val m = 3
            val clipped = x0 <= m || y0 <= m || x1 >= sw - m || y1 >= sh - m
            out.add(Block(x0 * ds, y0 * ds, min(x1 * ds, W), min(y1 * ds, H), clipped, g * ds))
        }
        return out
    }

    /** crop box: [margin] (fraction of the short side; 4% is what eval_sheet4 uses for the snug crop, 8% was decode_capture's) around the block, clipped to the frame. */
    fun cropBox(b: Block, W: Int, H: Int, margin: Double = 0.08): IntArray {
        val mg = (margin * min(b.x1 - b.x0, b.y1 - b.y0)).toInt()
        return intArrayOf(max(0, b.x0 - mg), max(0, b.y0 - mg), min(W, b.x1 + mg), min(H, b.y1 + mg))
    }
}
