package me.stedra.stegocam

import android.content.Context
import android.util.Log
import org.opencv.core.Mat
import java.io.File

/** Sensor-orientation box (pixels of sw x sh) to upright normalised [x0, y0, x1, y1] (the preview's coordinates). */
object Upright {
    fun box(x0: Float, y0: Float, x1: Float, y1: Float, sw: Int, sh: Int, orient: Int): FloatArray {
        val fw = sw.toFloat(); val fh = sh.toFloat()
        return when (orient) {
            90 -> floatArrayOf(1f - y1 / fh, x0 / fw, 1f - y0 / fh, x1 / fw)
            270 -> floatArrayOf(y0 / fh, 1f - x1 / fw, y1 / fh, 1f - x0 / fw)
            180 -> floatArrayOf(1f - x1 / fw, 1f - y1 / fh, 1f - x0 / fw, 1f - y0 / fh)
            else -> floatArrayOf(x0 / fw, y0 / fh, x1 / fw, y1 / fh)
        }
    }
}

/** What the card shows plus what the hidden reveal panel shows. */
class ScanResult(val text: String, val source: Source, val detail: String, val z: Double, val tMs: Long, val box: FloatArray?, val raw: String) {
    enum class Source { QR, STEGO }
}

/**
 * Scanner mode: a standard QR decoder and the stego decoders run on the same frames; whichever resolves first becomes the result
 * ([winner]); the other source keeps running so the reveal panel can show both ([qr], [stego]). Resets with [reset].
 * Threading: [onFrame] on the analysis thread; [QrScanner] callbacks on the main thread; [onChange] is called from either (post to the UI).
 */
/** Receiver of every analysis frame in scanner-like modes (Y plane in sensor orientation + the sensor orientation in degrees). */
interface FrameSink { fun onFrame(full: Mat, orient: Int) }

class ScannerEngine(ctx: Context, val sources: List<StegoSource>, val resolver: PayloadResolver = DemoResolver()) : FrameSink {
    companion object { private const val TAG = "StegoCam" }

    val qrScanner = QrScanner()
    @Volatile var useQr = true
    @Volatile var useStego = true
    @Volatile var winner: ScanResult? = null
        private set
    @Volatile var qr: ScanResult? = null
        private set
    @Volatile var stego: ScanResult? = null
        private set
    @Volatile var onChange: (() -> Unit)? = null
    @Volatile var startMs = System.currentTimeMillis()
        private set
    @Volatile private var sw = 4096
    @Volatile private var sh = 3072
    @Volatile private var orient = 90
    private val logFile = File(ctx.getExternalFilesDir(null), "scan_events.log")
    private fun log(line: String) { Log.i(TAG, line); try { logFile.appendText(line + "\n") } catch (_: Throwable) { } }

    init {
        qrScanner.onHit = { h ->
            if (useQr && qr == null) {
                val b = h.box
                val ub = if (b[2] > b[0]) Upright.box(b[0].toFloat(), b[1].toFloat(), b[2].toFloat(), b[3].toFloat(), sw, sh, orient) else null
                val tNow = System.currentTimeMillis() - startMs
                val r = ScanResult(h.text, ScanResult.Source.QR, "QR (ML Kit bundled, ${qrScanner.frames} frames)", 0.0, tNow, ub, h.text)
                qr = r
                log("SCAN_QR t_ms=$tNow text=${h.text}")
                settle(r)
            }
        }
    }

    @Synchronized private fun settle(r: ScanResult) {
        if (winner == null) { winner = r; log("SCAN_WINNER source=${r.source} t_ms=${r.tMs} text=${r.text} detail=${r.detail}") }
        onChange?.invoke()
    }

    fun reset() {
        synchronized(this) { winner = null; qr = null; stego = null; startMs = System.currentTimeMillis() }
        qrScanner.reset()
        for (s in sources) s.reset()
        log("SCAN_RESET")
        onChange?.invoke()
    }

    /** Analysis thread. [full] = Y plane (sensor orientation), [orient] = sensor orientation in degrees. */
    override fun onFrame(full: Mat, orient: Int) {
        this.orient = orient; sw = full.cols(); sh = full.rows()
        if (useQr && qr == null) qrScanner.offer(full)
        if (useStego && stego == null) {
            for (s in sources) {
                val hits = s.onFrame(full)
                val h = hits.firstOrNull() ?: continue
                val text = resolver.resolve(h) ?: continue
                val b = h.box
                val ub = Upright.box(b[0].toFloat(), b[1].toFloat(), b[2].toFloat(), b[3].toFloat(), sw, sh, orient)
                val tNow = System.currentTimeMillis() - startMs
                val r = ScanResult(text, ScanResult.Source.STEGO, "${s.name}/${h.kind}: ${h.detail}", h.z, tNow, ub, h.payload)
                stego = r
                log("SCAN_STEGO t_ms=$tNow kind=${h.kind} z=%.1f payload=${h.payload} text=$text".format(h.z))
                settle(r)
                break
            }
        }
    }

    fun shutdown() { for (s in sources) s.shutdown(); qrScanner.close() }
}
