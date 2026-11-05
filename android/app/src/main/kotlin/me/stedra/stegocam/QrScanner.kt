package me.stedra.stegocam

import android.util.Log
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Standard QR decoder (ML Kit barcode scanning, the BUNDLED model: no Play-services download) on the same frames as the stego decoders.
 * One frame in flight: a frame is half-sized (INTER_AREA), packed as NV21 (gray chroma) and handed to ML Kit; the result callback runs on the main thread.
 */
class QrScanner {
    class Hit(val text: String, val box: IntArray, val tMs: Long)
    /** One barcode of one frame; [box] in full-frame px (sensor orientation); [t0Ns] = System.nanoTime() when the frame was offered. */
    class Det(val text: String, val box: IntArray, val t0Ns: Long)

    companion object { private const val TAG = "StegoCam"; const val SCALE = 0.5 }

    private val client = BarcodeScanning.getClient(BarcodeScannerOptions.Builder().setBarcodeFormats(Barcode.FORMAT_QR_CODE).build())
    private val busy = AtomicBoolean(false)
    private var nv21: ByteArray? = null
    private var ybuf: ByteArray? = null
    @Volatile var startMs = System.currentTimeMillis()
    @Volatile var onHit: ((Hit) -> Unit)? = null
    @Volatile var enabled = true
    /** Continuous mode (public scanner): every barcode of every processed frame. */
    @Volatile var onAll: ((List<Det>) -> Unit)? = null
    var lastMs = 0L
        private set
    var frames = 0
        private set

    fun reset() { startMs = System.currentTimeMillis() }

    /** Analysis thread. Copies what it needs before returning. */
    fun offer(full: Mat) {
        if (!enabled || !busy.compareAndSet(false, true)) return
        try {
            val w = ((full.cols() * SCALE).toInt()) and 1.inv(); val h = ((full.rows() * SCALE).toInt()) and 1.inv()
            val small = Mat()
            Imgproc.resize(full, small, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            val ysz = w * h
            var buf = nv21
            if (buf == null || buf.size != ysz * 3 / 2) { buf = ByteArray(ysz * 3 / 2).also { java.util.Arrays.fill(it, ysz, it.size, 128.toByte()) }; nv21 = buf; ybuf = ByteArray(ysz) }
            small.get(0, 0, ybuf!!)
            System.arraycopy(ybuf!!, 0, buf, 0, ysz)
            small.release()
            val t0 = System.currentTimeMillis(); val t0n = System.nanoTime()
            val img = InputImage.fromByteArray(buf, w, h, 0, InputImage.IMAGE_FORMAT_NV21)
            client.process(img)
                .addOnSuccessListener { list ->
                    lastMs = System.currentTimeMillis() - t0; frames++
                    onAll?.let { cb ->
                        val ds = list.filter { !it.rawValue.isNullOrEmpty() && it.boundingBox != null }.map { bc ->
                            val bb = bc.boundingBox!!
                            Det(bc.rawValue!!, intArrayOf((bb.left / SCALE).toInt(), (bb.top / SCALE).toInt(), (bb.right / SCALE).toInt(), (bb.bottom / SCALE).toInt()), t0n)
                        }
                        if (ds.isNotEmpty()) cb(ds)
                    }
                    val b = list.firstOrNull { !it.rawValue.isNullOrEmpty() }
                    if (b != null) {
                        val bb = b.boundingBox
                        val box = if (bb != null) intArrayOf((bb.left / SCALE).toInt(), (bb.top / SCALE).toInt(), (bb.right / SCALE).toInt(), (bb.bottom / SCALE).toInt()) else intArrayOf(0, 0, 0, 0)
                        onHit?.invoke(Hit(b.rawValue!!, box, System.currentTimeMillis() - startMs))
                    }
                    busy.set(false)
                }
                .addOnFailureListener { e -> Log.w(TAG, "QR scan failed: $e"); busy.set(false) }
        } catch (e: Throwable) { Log.e(TAG, "QR offer failed", e); busy.set(false) }
    }

    /** Blocking one-shot scan of a full-res Y frame (call off the main thread). Returns the first QR text or null. */
    fun scanBlocking(full: Mat): String? {
        val w = ((full.cols() * SCALE).toInt()) and 1.inv(); val h = ((full.rows() * SCALE).toInt()) and 1.inv()
        val small = Mat()
        Imgproc.resize(full, small, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val ysz = w * h
        val buf = ByteArray(ysz * 3 / 2).also { java.util.Arrays.fill(it, ysz, it.size, 128.toByte()) }
        val y = ByteArray(ysz); small.get(0, 0, y); System.arraycopy(y, 0, buf, 0, ysz); small.release()
        val list = com.google.android.gms.tasks.Tasks.await(client.process(InputImage.fromByteArray(buf, w, h, 0, InputImage.IMAGE_FORMAT_NV21)), 15, java.util.concurrent.TimeUnit.SECONDS)
        return list.firstOrNull { !it.rawValue.isNullOrEmpty() }?.rawValue
    }

    fun close() { try { client.close() } catch (_: Throwable) { } }
}
