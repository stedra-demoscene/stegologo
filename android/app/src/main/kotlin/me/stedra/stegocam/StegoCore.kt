package me.stedra.stegocam

import android.content.Context
import android.util.Log
import org.opencv.core.Mat
import java.nio.ByteBuffer

/** Result of one native decode. [ms] = darkness, coarse, edge, measure, fit, tail, golay, total (milliseconds). */
class CoreResult(
    val hex: String, val crcOk: Boolean, val accept: Boolean, val z: Double, val ampUm: Double, val ampRawUm: Double,
    val nValid: Int, val H: DoubleArray, val regScore: Double, val ms: DoubleArray,
    val chipY: DoubleArray = DoubleArray(0), val chipW: DoubleArray = DoubleArray(0), val pilotS2: Double = 0.0,
    /** decoy-calibrated pilot z (the accept statistic) and the decoy spread; [z] is the raw pilot z */
    val zCal: Double = 0.0, val nullSd: Double = 1.0,
) {
    val totalMs get() = ms[7]
    val stages get() = "dark=%.1f coarse=%.1f edge=%.1f meas=%.1f fit=%.1f tail=%.1f".format(ms[0], ms[1], ms[2], ms[3], ms[4], ms[5])
}

/** Result of one native mark detect + decode (core/src/mark_decode.cpp). [crop] = x0,y0,x1,y1 in the frame; [detMs] = prep, stageA, polar, fine, ecc, total. */
class MarkOutcome(
    val found: Boolean, val score: Double, val iso: Double, val ppm: Double, val rotDeg: Double, val cx: Double, val cy: Double,
    val H: DoubleArray, val asset: Int, val hyp: Int, val refined: Boolean, val crop: IntArray,
    val msDetect: Double, val msDecode: Double, val msTotal: Double, val nDecodes: Int, val detMs: DoubleArray, val core: CoreResult?,
)

/** Combined result of the demo path (core/src/demo_decode.cpp): [H] = design (composite) page mm -> frame px, [crop] = x0,y0,x1,y1. */
class DemoOutcome(
    val accept: Boolean, val crcOk: Boolean, val tagOk: Boolean, val token: String, val z: Double, val zs: DoubleArray, val gains: DoubleArray,
    val regScore: Double, val wide: Boolean, val H: DoubleArray, val crop: IntArray, val msReg: Double, val msDec: Double, val msTotal: Double,
)

/** JNI wrapper over the shared C++ decoder (core/). Thread-safe (the asset is immutable). */
object StegoCore {
    private const val TAG = "StegoCam"
    private var loaded = false

    /** Assets in handle order; handle 0 = sheet 2 (6 text blocks). Handle 1 = sheet-4 128-bit block (per-region key), if present. */
    val ASSETS = listOf("sheet2_ref", "sheet4_block_ref")
    @Volatile var nAssets = 0
        private set

    @Synchronized fun init(ctx: Context) {
        if (loaded) return
        System.loadLibrary("stegojni")
        for (name in ASSETS) {
            try {
                val j = ctx.assets.open("$name.json").bufferedReader().use { it.readText() }
                val b = ctx.assets.open("$name.bin").use { it.readBytes() }
                val h = nInit(j, b)
                check(h == nAssets) { "asset load failed: $name" }
                nAssets++
            } catch (e: java.io.FileNotFoundException) { Log.w(TAG, "asset $name missing (skipped)") }
        }
        Log.i(TAG, "ASSETS loaded: $nAssets")
        loaded = true
    }

    /** Decode a uint8 crop in place (Mat.dataAddr + step; works for submats too). */
    fun decode(m: Mat, asset: Int = 0, wide: Boolean = false, fast: Boolean = false): CoreResult? =
        nDecode(asset, m.dataAddr(), m.cols(), m.rows(), m.step1().toInt(), (if (wide) 1 else 0) or (if (fast) 2 else 0))

    /** Decode a rectangle of a direct ByteBuffer plane (e.g. ImageReader Y plane), no copy. */
    fun decode(buf: ByteBuffer, rowStride: Int, x0: Int, y0: Int, x1: Int, y1: Int, asset: Int = 0, wide: Boolean = false): CoreResult? =
        nDecodeBuf(asset, buf, y0 * rowStride + x0, x1 - x0, y1 - y0, rowStride, if (wide) 1 else 0)

    /** K crops of one block (crop 0 is the reference frame). */
    fun decodeMulti(ms: List<Mat>, asset: Int = 0, wide: Boolean = false): CoreResult? = nDecodeMulti(
        asset, LongArray(ms.size) { ms[it].dataAddr() }, IntArray(ms.size) { ms[it].cols() }, IntArray(ms.size) { ms[it].rows() },
        IntArray(ms.size) { ms[it].step1().toInt() }, if (wide) 1 else 0)

    /** Result from accumulated chip statistics (sum of a_k*chipY, a_k*chipW with a_k = 1/pilotS2). */
    fun finalize(y: DoubleArray, w: DoubleArray, asset: Int = 0): CoreResult? = nFinalize(asset, y, w)

    // ---- marks
    /** Load an extra asset by name (assets/<name>.json/.bin); returns its handle, or -1. */
    @Synchronized fun loadAsset(ctx: Context, name: String): Int {
        init(ctx)
        return try {
            val j = ctx.assets.open("$name.json").bufferedReader().use { it.readText() }
            val b = ctx.assets.open("$name.bin").use { it.readBytes() }
            nInit(j, b)
        } catch (e: java.io.FileNotFoundException) { Log.w(TAG, "asset $name missing"); -1 }
    }
    /** Mark detector for the geometry of asset [handle] (rotation-averaged template, polar search tables). */
    @Synchronized fun markInit(handle: Int): Int = nMarkInit(handle)

    /**
     * Detect the mark in [m] (uint8 frame or view) and decode it against [assets] (same geometry, different keys).
     * flags: bit0 wide registration, bit1 fast fit, bit2 no refine retry, bit3 detect only. ppm range = frame px per mm. [exclude] = triples (cx, cy, r) in px.
     */
    fun decodeMark(det: Int, assets: IntArray, m: Mat, ppmLo: Double, ppmHi: Double, sym: Int = 1, flags: Int = 0, exclude: DoubleArray? = null): MarkOutcome? =
        nMarkDecode(det, assets, m.dataAddr(), m.cols(), m.rows(), m.step1().toInt(), flags, ppmLo, ppmHi, sym, exclude)

    /** Whole-design decode: [seed] = composite page mm -> frame px (3x3 row-major); [shifts] = region page -> composite page (x, y per region). */
    fun demoDecode(comp: Int, regions: IntArray, shifts: DoubleArray, m: Mat, seed: DoubleArray, fast: Boolean = false): DemoOutcome? =
        nDemoDecode(comp, regions, shifts, m.dataAddr(), m.cols(), m.rows(), m.step1().toInt(), seed, if (fast) 2 else 0)

    /** cpus whose cpuinfo_max_freq equals the maximum (prime cluster); fallback: the top two cpus of an 8-core. 0 = no restriction. */
    fun primeCpuMask(): Long {
        val n = Runtime.getRuntime().availableProcessors()
        try {
            val f = LongArray(n) { java.io.File("/sys/devices/system/cpu/cpu$it/cpufreq/cpuinfo_max_freq").readText().trim().toLong() }
            val mx = f.max()
            var m = 0L; for (i in 0 until n) if (f[i] == mx) m = m or (1L shl i)
            if (m != 0L && m != (1L shl n) - 1) return m
        } catch (e: Throwable) { }
        return if (n >= 8) 0xC0L else 0L
    }

    /** Pin thread [tid] to the cpus in [mask]; 0 = ok, else errno. */
    fun setAffinity(tid: Int, mask: Long): Int = nSetAffinity(tid, mask)
    fun affinity(tid: Int): Long = nGetAffinity(tid)

    private external fun nInit(json: String, bin: ByteArray): Int
    private external fun nDemoDecode(comp: Int, regions: IntArray, shifts: DoubleArray, addr: Long, w: Int, h: Int, stride: Int, seed: DoubleArray, flags: Int): DemoOutcome?
    private external fun nMarkInit(asset: Int): Int
    private external fun nMarkDecode(det: Int, assets: IntArray, addr: Long, w: Int, h: Int, stride: Int, flags: Int, ppmLo: Double, ppmHi: Double, sym: Int, exclude: DoubleArray?): MarkOutcome?
    private external fun nDecode(asset: Int, addr: Long, w: Int, h: Int, stride: Int, flags: Int): CoreResult?
    private external fun nDecodeBuf(asset: Int, buf: ByteBuffer, off: Int, w: Int, h: Int, stride: Int, flags: Int): CoreResult?
    private external fun nDecodeMulti(asset: Int, addrs: LongArray, ws: IntArray, hs: IntArray, strides: IntArray, flags: Int): CoreResult?
    private external fun nFinalize(asset: Int, y: DoubleArray, w: DoubleArray): CoreResult?
    private external fun nSetAffinity(tid: Int, mask: Long): Int
    private external fun nGetAffinity(tid: Int): Long
}
