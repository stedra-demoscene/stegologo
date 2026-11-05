package me.stedra.stegocam

import android.content.Context
import android.util.Log
import org.opencv.core.Mat
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Engine of the public scanner: continuous, multi-object. Every frame goes to ML Kit (QR, URL payloads only) and, one frame at a time, to the demo stego
 * decoder (DemoStego.identify + decodeIdent, the scanner-mode code path). Each detection is reported with the object's centre in normalised upright
 * coordinates (the preview's) and the time the source frame entered the engine, for latency accounting.
 */
class PublicScanner(ctx: Context, private val onDet: (Det) -> Unit) : FrameSink {
    class Det(val kind: String, val url: String, val cx: Float, val cy: Float, val t0Ns: Long, val z: Double = 0.0, val ms: Double = 0.0)

    companion object {
        private const val TAG = "StegoCam"
        const val Z_PUBLIC = 8.0                       // combined z needed to show a stego link (real signal >= 15.9, 0 um control up to 3.4)
        private val TOKEN = Regex("[0-9A-Za-z]{6,7}")  // token length 6 or 7
        fun isUrl(s: String) = s.startsWith("http://", true) || s.startsWith("https://", true)
    }

    val demo = DemoStego(ctx)
    private val qr = QrScanner()
    private val busy = AtomicBoolean(false)
    @Volatile var enabled = true
    @Volatile private var sw = 4096
    @Volatile private var sh = 3072
    @Volatile private var orient = 90
    private val pool = Executors.newSingleThreadExecutor { r -> Thread({ android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND - 2); r.run() }, "public-stego") }
    private var affinityDone = false

    init {
        qr.onAll = { list ->
            for (d in list) if (isUrl(d.text.trim())) {
                val b = Upright.box(d.box[0].toFloat(), d.box[1].toFloat(), d.box[2].toFloat(), d.box[3].toFloat(), sw, sh, orient)
                onDet(Det("qr", d.text.trim(), (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, d.t0Ns))
            }
        }
        pool.execute { try { demo.ensureLoaded() } catch (e: Throwable) { Log.e(TAG, "demo load failed", e) } }   // warm the card references off the first frame
    }

    override fun onFrame(full: Mat, orient: Int) {
        if (!enabled) return
        this.orient = orient; sw = full.cols(); sh = full.rows()
        qr.offer(full)
        if (!busy.compareAndSet(false, true)) return
        val frame = full.clone()
        val t0 = System.nanoTime()
        try { pool.execute { work(frame, t0) } } catch (e: Exception) { busy.set(false); frame.release() }
    }

    private fun work(frame: Mat, t0: Long) {
        try {
            if (!affinityDone) { affinityDone = true; val m = StegoCore.primeCpuMask(); if (m != 0L) StegoCore.setAffinity(android.os.Process.myTid(), m) }
            val id = demo.identify(frame) ?: return
            val c = id.card; val h = id.hCard
            if (c == null || h == null || id.inliers < DemoStego.MIN_INLIERS) return
            val o = demo.decodeIdent(frame, id) ?: return
            Log.i(TAG, "PUBLIC stego card ${c.pct} inl=${id.inliers} z=%.1f acc=${o.accept} tok=${o.token} %.0fms".format(o.z, (System.nanoTime() - t0) / 1e6))
            if (!o.accept || o.z <= Z_PUBLIC || !TOKEN.matches(o.token)) return
            val p = demo.centrePx(h)
            val b = Upright.box(p[0].toFloat(), p[1].toFloat(), p[0].toFloat(), p[1].toFloat(), frame.cols(), frame.rows(), orient)
            onDet(Det("logo", DemoConfig.prefix + o.token, b[0], b[1], t0, o.z, (System.nanoTime() - t0) / 1e6))
        } catch (e: Throwable) { Log.e(TAG, "public stego failed", e) }
        finally { frame.release(); busy.set(false) }
    }

    fun shutdown() { enabled = false; pool.shutdown(); qr.close(); demo.shutdown() }
}
