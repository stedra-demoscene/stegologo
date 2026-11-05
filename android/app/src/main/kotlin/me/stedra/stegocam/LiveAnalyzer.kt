package me.stedra.stegocam

import android.media.Image
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

/**
 * Live text-row / block detector (mirrors tools/decode_capture.py detect_rows) on the YUV Y plane.
 * Runs on its own thread; frames are dropped while busy. Boxes are returned normalised (0..1) in the
 * upright (portrait, display) frame.
 */
class LiveAnalyzer(ctx: android.content.Context, private val onResult: (Result) -> Unit) {
    /** colour: 0 accepted, 1 CRC ok but pilot z too low, 2 CRC failed */
    class Label(val hex: String, val sub: String, val colour: Int, val stable: Boolean)
    class Result(val boxes: List<FloatArray>, val labels: List<Label?>, val ms: Float, val fps: Float, val nBoxes: Int, val summary: String = "")

    private val decoder = LiveDecoder(ctx)
    /** Run the sheet decoder on detected blocks (broadcast LIVE with --ez decode false to turn it off). */
    @Volatile var decodeOn = true

    /** Scanner mode: frames go to the scanner engine (QR + stego sources) instead of the block detector. */
    @Volatile var scanner: FrameSink? = null

    /** Gather mode: when set, frames go to the pose estimator instead of the block detector / decoder. */
    @Volatile var poseCb: ((PoseEstimator.Pose?) -> Unit)? = null
    @Volatile var poseExpected = 1
    /** Gather mode on demo cards: pose from the card matcher instead of the page matcher (poseExpected = card pct, 0 = any). */
    @Volatile var poseDemo: DemoStego? = null
    @Volatile var poseK = doubleArrayOf(3000.0, 3000.0, 2048.0, 1536.0)
    private var poseRuns = 0
    private var lastPose: PoseEstimator.Pose? = null
    private var lastPoseStart = 0L

    companion object {
        private const val TAG = "StegoCam"
        const val DS = 4
        @Volatile private var cvOk = false
        fun ensureCv(): Boolean {
            if (!cvOk) cvOk = try { OpenCVLoader.initLocal() } catch (e: Throwable) { Log.e(TAG, "OpenCV load failed", e); false }
            return cvOk
        }
    }

    private val busy = AtomicBoolean(false)
    @Volatile private var accepting = true
    private val gate = java.util.concurrent.locks.ReentrantLock()
    private val exec = Executors.newSingleThreadExecutor { Thread(it, "live-analysis") }
    @Volatile private var summary = ""
    private var lastDone = 0L
    private var emaFps = 0f
    private var emaMs = 0f
    private var lastLog = 0L
    private var logMsSum = 0f; private var logN = 0

    /** Takes ownership of [img] (closes it) when it returns true; returns false (img untouched) if busy. */
    fun submit(img: Image, sensorOrientation: Int): Boolean {
        if (!accepting || !ensureCv()) return false
        if (poseCb != null && System.currentTimeMillis() - lastPoseStart < 90) return false   // pose at <= ~10 Hz
        if (!busy.compareAndSet(false, true)) return false
        try {
            exec.execute {
                gate.lock()
                try { process(img, sensorOrientation) }
                catch (e: Throwable) { Log.e(TAG, "live analysis failed", e) }
                finally { img.close(); busy.set(false); gate.unlock() }
            }
        } catch (e: Exception) { busy.set(false); return false }
        return true
    }

    /** Stop accepting frames and wait for the in-flight analysis (it reads camera memory). Call before closing the ImageReader. */
    fun quiesce(timeoutMs: Long = 3000): Boolean {
        accepting = false
        val ok = try { gate.tryLock(timeoutMs, java.util.concurrent.TimeUnit.MILLISECONDS) } catch (e: InterruptedException) { false }
        if (ok) gate.unlock()
        return ok
    }

    fun resume() { accepting = true }

    /** Forget all tracks (live switched on/off). */
    fun reset() { decoder.reset() }

    fun shutdown() { exec.shutdown(); decoder.shutdown() }

    private fun process(img: Image, orient: Int) {
        val t0 = System.nanoTime()
        val p = img.planes[0]
        val w = img.width; val h = img.height
        val full = Mat(h, w, CvType.CV_8UC1, p.buffer, p.rowStride.toLong())   // no copy
        val pcb = poseCb
        if (pcb != null) {
            lastPoseStart = System.currentTimeMillis()
            try {
                val dm = poseDemo
                if (dm != null) { val pose = dm.pose(full, poseK, poseExpected); lastPose = pose; pcb(pose); return }
                val lp = lastPose
                val hint = if (lp != null && lp.found && lp.pageOk && lp.inliers >= 15 && System.currentTimeMillis() - lp.t < 1000) lp.h else null
                val all = (poseRuns++ % 6 == 0)
                var pose = PoseEstimator.estimate(full, poseK, poseExpected, all, hint)
                if (pose != null && !all && !pose.found && hint == null) pose = PoseEstimator.estimate(full, poseK, poseExpected, true)
                lastPose = pose
                pcb(pose)
            } finally { full.release() }
            return
        }
        val sc = scanner
        if (sc != null) {
            try { sc.onFrame(full, orient) } finally { full.release() }
            return
        }
        val boxes = ArrayList<FloatArray>()
        val labels = ArrayList<Label?>()
        try {
            val blocks = BlockDetector.detect(full).filter { max(it.x1 - it.x0, it.y1 - it.y0) >= 200 }   // ignore noise specks in the dark
            val (views, sum) = if (decodeOn) decoder.update(full, blocks) else Pair(blocks.map { null }, "")
            summary = sum
            for ((i, b) in blocks.withIndex()) {
                boxes.add(toUpright(b.x0.toFloat(), b.y0.toFloat(), b.x1.toFloat(), b.y1.toFloat(), w, h, orient))
                val v = views[i]
                labels.add(if (v == null) null else {
                    val sub = if (v.colour == 0) "z %.1f  %.1fs  %.0fms".format(v.z, v.ttaMs / 1000.0, v.latencyMs)
                              else "z %.1f  %s  %.0fms  x%d".format(v.z, if (v.colour == 1) "weak z" else "rej", v.latencyMs, v.nFrames)
                    Label(v.hex ?: "", sub, v.colour, v.stable)
                })
            }
        } finally {
            full.release()
        }
        val now = System.nanoTime()
        val ms = (now - t0) / 1e6f
        val nowMs = now / 1_000_000
        if (lastDone != 0L) {
            val inst = 1000f / max(1L, nowMs - lastDone)
            emaFps = if (emaFps == 0f) inst else 0.8f * emaFps + 0.2f * inst
        }
        lastDone = nowMs
        emaMs = if (emaMs == 0f) ms else 0.8f * emaMs + 0.2f * ms
        logMsSum += ms; logN++
        if (nowMs - lastLog >= 1000) {
            Log.i(TAG, "LIVE n_boxes=${boxes.size} ms=%.1f fps=%.2f".format(logMsSum / logN, emaFps))
            lastLog = nowMs; logMsSum = 0f; logN = 0
        }
        onResult(Result(boxes, labels, emaMs, emaFps, boxes.size, summary))
    }

    /** Sensor-orientation box (pixels of sw x sh) to upright normalised [x0,y0,x1,y1]. */
    private fun toUpright(x0: Float, y0: Float, x1: Float, y1: Float, sw: Int, sh: Int, orient: Int): FloatArray {
        val fw = sw.toFloat(); val fh = sh.toFloat()
        return when (orient) {
            90 -> floatArrayOf(1f - y1 / fh, x0 / fw, 1f - y0 / fh, x1 / fw)
            270 -> floatArrayOf(y0 / fh, 1f - x1 / fw, y1 / fh, 1f - x0 / fw)
            180 -> floatArrayOf(1f - x1 / fw, 1f - y1 / fh, 1f - x0 / fw, 1f - y0 / fh)
            else -> floatArrayOf(x0 / fw, y0 / fh, x1 / fw, y1 / fh)
        }
    }
}
