package me.stedra.stegocam

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
import android.hardware.camera2.params.MeteringRectangle
import android.media.Image
import android.media.ImageReader
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Log
import android.util.Size
import android.view.Surface
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ConcurrentSkipListMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * All Camera2 work. Everything mutable runs on [handler] (the camera thread).
 * Streams to a preview surface + a YUV_420_888 ImageReader; no still captures.
 */
class CameraController(private val ctx: Context, private val listener: Listener) {

    interface Listener {
        fun onMessage(msg: String)
        fun onStatusLine(line: String)
        fun onLive(boxes: List<FloatArray>, labels: List<LiveAnalyzer.Label?>, ms: Float, fps: Float, summary: String) {}
        /** previewSize is in sensor orientation (landscape-ish); sensorOrientation in degrees. */
        fun onPreviewSize(size: Size, sensorOrientation: Int)
    }

    companion object {
        const val TAG = "StegoCam"
        private const val MAX_PREVIEW_LONG = 1920
        private const val MAX_FRAME_NS = 66_700_000L // 1/15 s
        private const val READER_IMAGES = 6
        private const val QUEUE_CAP = 6
        private const val NO_FRAME_TIMEOUT_MS = 6000L
    }

    private val manager = ctx.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val mainHandler = Handler(Looper.getMainLooper())

    private var thread: HandlerThread? = null
    private var handler: Handler? = null

    // settings (set from any thread via the setters)
    @Volatile var requestedCamera: String = "auto"
    @Volatile var procMode: String = "off"      // "off" | "default"

    private data class Cand(val id: String, val size: Size)
    private val cands = ArrayDeque<Cand>()
    private var cur: Cand? = null
    private var attempt = 0                       // invalidates stale callbacks

    private var surfaceTexture: SurfaceTexture? = null
    private var previewSurface: Surface? = null
    private var device: CameraDevice? = null
    private var deviceId: String? = null
    private var chars: CameraCharacteristics? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var templateName = "RECORD"
    private var previewSize: Size? = null

    // request state
    private var afMode = CaptureRequest.CONTROL_AF_MODE_AUTO
    private var aeLock = false
    private var lockEnabled = false
    private var regions: Array<MeteringRectangle>? = null
    private var tapToken = 0

    // streaming stats
    private var framesSinceTick = 0
    private var lastTickMs = 0L
    private var fps = 0f
    private var totalFrames = 0L
    private var gotFrame = false
    @Volatile private var lastResult: TotalCaptureResult? = null
    private val results = ConcurrentSkipListMap<Long, TotalCaptureResult>()
    private var captureFailures = 0

    // dump
    private var dump: Dump? = null
    private var pendingDump: PendingDump? = null
    private var settleFrames = 0

    // live analysis
    @Volatile var liveOn = false
        set(v) { if (field != v) analyzer.reset(); field = v; if (!v) liveStatus = "" }
    @Volatile private var liveStatus = ""
    /** Gather mode (pose estimation on the analysis thread). */
    @Volatile var gatherOn = false
    @Volatile var intrinsics = doubleArrayOf(3000.0, 3000.0, 2048.0, 1536.0)
    @Volatile var intrinsicsSrc = "default"
    fun setPoseSink(cb: ((PoseEstimator.Pose?) -> Unit)?, expected: Int) {
        analyzer.poseExpected = expected
        analyzer.poseCb = cb
        gatherOn = cb != null
    }
    fun setPoseExpected(page: Int) { analyzer.poseExpected = page }
    fun setPoseDemo(d: DemoStego?) { analyzer.poseDemo = d }
    /** Scanner mode (QR + stego sources on the analysis thread). */
    @Volatile var scannerOn = false
    fun setScanner(e: FrameSink?) { analyzer.scanner = e; scannerOn = e != null }
    var decodeOn: Boolean get() = analyzer.decodeOn; set(v) { analyzer.decodeOn = v }
    @Volatile private var sensorOrient = 90
    private val analyzer = LiveAnalyzer(ctx) { r ->
        liveStatus = "live %d boxes %.0f ms %.1f afps".format(r.nBoxes, r.ms, r.fps)
        listener.onLive(r.boxes, r.labels, r.ms, r.fps, r.summary)
    }

    private class PendingDump(val n: Int, val tag: String?, val extras: JSONObject? = null, val parent: File? = null, val dirName: String? = null,
                              val cb: ((File, String?) -> Unit)? = null)

    private class Item(val frame: Int, val ts: Long, val data: ByteArray)

    private class Dump(val n: Int, val tag: String, val dir: File, val proc: String, val extras: JSONObject? = null, val cb: ((File, String?) -> Unit)? = null) {
        var received = 0
        val dropped = ArrayList<Pair<Int, Long>>()     // frame number, timestamp
        val timestamps = ArrayList<Pair<Int, Long>>()  // all frames (written + dropped)
        val written = ArrayList<Int>()
        val queue = ArrayBlockingQueue<Item>(QUEUE_CAP)
        @Volatile var allReceived = false
        var error: String? = null
    }

    // ---------------------------------------------------------------- lifecycle

    val hasStarted: Boolean get() = thread != null

    fun start() {
        if (thread != null) return
        val t = HandlerThread("camera"); t.start()
        thread = t; handler = Handler(t.looper)
        handler?.post(ticker)
    }

    fun stop() {
        val h = handler ?: return
        val latch = CountDownLatch(1)
        h.post {
            h.removeCallbacksAndMessages(null)
            attempt++
            closeAll()
            latch.countDown()
        }
        latch.await(4, TimeUnit.SECONDS)
        thread?.quitSafely()
        try { thread?.join(1000) } catch (_: InterruptedException) {}
        thread = null; handler = null
    }

    fun open(st: SurfaceTexture) {
        handler?.post {
            if (surfaceTexture === st && device != null && session != null) return@post
            surfaceTexture = st
            restart()
        }
    }

    /** Re-run candidate selection from scratch (camera id changed, or resume). */
    fun restart() {
        handler?.post {
            val st = surfaceTexture ?: return@post
            attempt++
            closeAll()
            buildCandidates()
            tryNext()
        }
    }

    private fun closeSession() {
        analyzer.quiesce()
        try { session?.close() } catch (_: Exception) {}
        session = null
        try { reader?.close() } catch (_: Exception) {}
        reader = null
        try { previewSurface?.release() } catch (_: Exception) {}
        previewSurface = null
    }

    private fun closeDevice() {
        val d = device ?: return
        device = null; deviceId = null
        try { d.close() } catch (_: Exception) {}
        // closing is async; give the HAL a moment before reopening
        try { Thread.sleep(250) } catch (_: InterruptedException) {}
    }

    private fun closeAll() {
        closeSession(); closeDevice()
        cur = null
        results.clear()
    }

    // ---------------------------------------------------------------- candidates

    private fun minDuration(map: android.hardware.camera2.params.StreamConfigurationMap, s: Size) =
        map.getOutputMinFrameDuration(ImageFormat.YUV_420_888, s)

    private fun buildCandidates() {
        cands.clear()
        val ids = manager.cameraIdList.toSet()
        val req = requestedCamera
        val order = if (req == "auto" || req.isBlank()) listOf("2", "0") else listOf(req)
        for (id in order) {
            if (id !in ids) { Log.w(TAG, "camera $id not in cameraIdList $ids"); continue }
            val c = manager.getCameraCharacteristics(id)
            val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: continue
            val sizes = (map.getOutputSizes(ImageFormat.YUV_420_888)?.toList() ?: emptyList()) +
                (map.getHighResolutionOutputSizes(ImageFormat.YUV_420_888)?.toList() ?: emptyList())
            val uniq = sizes.distinct().sortedByDescending { it.width.toLong() * it.height }
            Log.i(TAG, "camera $id YUV_420_888 sizes (w x h : min frame duration ms / max fps):")
            for (s in uniq) {
                val d = minDuration(map, s)
                Log.i(TAG, "  cam $id ${s.width}x${s.height} : ${"%.2f".format(d / 1e6)} ms / ${"%.1f".format(1e9 / d)} fps" +
                    (if (d <= MAX_FRAME_NS) "" else "  (too slow)"))
            }
            val ok = uniq.filter { minDuration(map, it) <= MAX_FRAME_NS }
            val list = if (ok.isNotEmpty()) ok else uniq.sortedBy { minDuration(map, it) }
            if (ok.isEmpty()) Log.w(TAG, "camera $id: no YUV size reaches 15 fps; using all sizes by speed")
            for (s in list) cands.addLast(Cand(id, s))
        }
        Log.i(TAG, "candidate order: " + cands.joinToString { "${it.id}/${it.size.width}x${it.size.height}" })
    }

    private fun tryNext() {
        closeSession()
        val c = cands.removeFirstOrNull()
        if (c == null) {
            listener.onMessage("No working camera configuration")
            Log.e(TAG, "no working camera configuration; all candidates failed")
            return
        }
        cur = c
        val my = ++attempt
        gotFrame = false; captureFailures = 0
        Log.i(TAG, "trying camera ${c.id} YUV ${c.size.width}x${c.size.height} proc=$procMode")
        if (device != null && deviceId != c.id) closeDevice()
        if (device != null) { configure(my); return }
        openDevice(c.id, my)
    }

    private fun fail(my: Int, why: String) {
        if (my != attempt) return
        Log.w(TAG, "FALLBACK: camera ${cur?.id} ${cur?.size} failed: $why")
        tryNext()
    }

    @SuppressLint("MissingPermission")
    private fun openDevice(id: String, my: Int) {
        try {
            manager.openCamera(id, object : CameraDevice.StateCallback() {
                override fun onOpened(cam: CameraDevice) {
                    handler?.post {
                        if (my != attempt) { cam.close(); return@post }
                        device = cam; deviceId = id
                        chars = manager.getCameraCharacteristics(id)
                        configure(my)
                    }
                }
                override fun onDisconnected(cam: CameraDevice) {
                    cam.close()
                    handler?.post {
                        if (device === cam) { device = null; deviceId = null }
                        fail(my, "disconnected")
                    }
                }
                override fun onError(cam: CameraDevice, error: Int) {
                    Log.e(TAG, "CameraDevice error $error on $id")
                    cam.close()
                    handler?.post {
                        if (device === cam) { device = null; deviceId = null }
                        fail(my, "device error $error")
                    }
                }
                override fun onClosed(cam: CameraDevice) { Log.i(TAG, "camera $id closed") }
            }, mainHandler)
        } catch (e: Exception) {
            Log.e(TAG, "openCamera($id) failed", e)
            fail(my, "open exception ${e.message}")
        }
    }

    private fun choosePreview(c: CameraCharacteristics, yuv: Size): Size? {
        val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return null
        val sizes = map.getOutputSizes(SurfaceTexture::class.java)?.toList() ?: return null
        val ar = yuv.width.toDouble() / yuv.height
        val fits = sizes.filter { maxOf(it.width, it.height) <= MAX_PREVIEW_LONG }
        val pool = if (fits.isNotEmpty()) fits else sizes
        val exact = pool.filter { Math.abs(it.width.toDouble() / it.height - ar) < 0.005 }
        return (if (exact.isNotEmpty()) exact else pool.sortedBy { Math.abs(it.width.toDouble() / it.height - ar) }.take(3))
            .maxByOrNull { it.width.toLong() * it.height }
    }

    private fun configure(my: Int) {
        val cam = device ?: return fail(my, "no device")
        val st = surfaceTexture ?: return
        val c = cur ?: return
        val ch = chars ?: return
        val ps = choosePreview(ch, c.size) ?: return fail(my, "no preview size")
        previewSize = ps
        Log.i(TAG, "preview size ${ps.width}x${ps.height}")
        sensorOrient = ch.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
        try {
            val arr = ch.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE) ?: ch.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)!!
            val pa = ch.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
            val phys = ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
            val (k, src) = PoseEstimator.intrinsics(ch.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION),
                ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull(), phys?.width, phys?.height,
                arr.width(), arr.height(), c.size.width, c.size.height)
            intrinsics = k; intrinsicsSrc = src; analyzer.poseK = k
            Log.i(TAG, "INTRINSICS fx=%.1f fy=%.1f cx=%.1f cy=%.1f src=%s array=%dx%d pixelArray=%s calib=%s".format(k[0], k[1], k[2], k[3], src,
                arr.width(), arr.height(), pa, ch.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)?.joinToString(",")))
        } catch (e: Exception) { Log.w(TAG, "intrinsics failed", e) }
        listener.onPreviewSize(ps, sensorOrient)
        st.setDefaultBufferSize(ps.width, ps.height)
        val surf = Surface(st)
        previewSurface = surf
        analyzer.resume()
        val r = ImageReader.newInstance(c.size.width, c.size.height, ImageFormat.YUV_420_888, READER_IMAGES)
        reader = r
        r.setOnImageAvailableListener({ onImage(it, my) }, handler)
        try {
            @Suppress("DEPRECATION")
            cam.createCaptureSession(listOf(surf, r.surface), object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(s: CameraCaptureSession) {
                    if (my != attempt) { s.close(); return }
                    session = s
                    Log.i(TAG, "session configured: camera ${c.id} YUV ${c.size.width}x${c.size.height}")
                    afMode = if (afModes().contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO))
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO
                    else if (afModes().contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE))
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE else CaptureRequest.CONTROL_AF_MODE_AUTO
                    aeLock = false; lockEnabled = false; regions = null
                    if (applyRepeating()) {
                        Log.i(TAG, "StegoCam STREAMING camera=${c.id} size=${c.size.width}x${c.size.height} template=$templateName proc=$procMode")
                        handler?.postDelayed({
                            if (my == attempt && !gotFrame) fail(my, "no YUV frame within ${NO_FRAME_TIMEOUT_MS} ms")
                        }, NO_FRAME_TIMEOUT_MS)
                    } else fail(my, "setRepeatingRequest failed")
                }
                override fun onConfigureFailed(s: CameraCaptureSession) {
                    handler?.post { fail(my, "onConfigureFailed") }
                }
            }, handler)
        } catch (e: Exception) {
            Log.e(TAG, "createCaptureSession failed", e)
            fail(my, "createCaptureSession exception ${e.message}")
        }
    }

    private fun afModes(): IntArray = chars?.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES) ?: intArrayOf()

    // ---------------------------------------------------------------- requests

    private fun buildRequest(trigger: Int? = null): CaptureRequest? {
        val cam = device ?: return null
        val ch = chars ?: return null
        val r = reader ?: return null
        val ps = previewSurface ?: return null
        val b = try {
            templateName = "RECORD"
            cam.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
        } catch (e: Exception) {
            Log.w(TAG, "TEMPLATE_RECORD failed (${e.message}); using PREVIEW")
            templateName = "PREVIEW"
            cam.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
        }
        b.addTarget(ps); b.addTarget(r.surface)
        b.set(CaptureRequest.CONTROL_AF_MODE, afMode)
        b.set(CaptureRequest.CONTROL_AE_LOCK, aeLock)
        regions?.let {
            if ((ch.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF) ?: 0) > 0) b.set(CaptureRequest.CONTROL_AF_REGIONS, it)
            if ((ch.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AE) ?: 0) > 0) b.set(CaptureRequest.CONTROL_AE_REGIONS, it)
        }
        if (trigger != null) b.set(CaptureRequest.CONTROL_AF_TRIGGER, trigger)
        if (procMode == "off") {
            fun has(key: CameraCharacteristics.Key<IntArray>, v: Int) = ch.get(key)?.contains(v) == true
            if (has(CameraCharacteristics.EDGE_AVAILABLE_EDGE_MODES, CaptureRequest.EDGE_MODE_OFF))
                b.set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_OFF)
            if (has(CameraCharacteristics.NOISE_REDUCTION_AVAILABLE_NOISE_REDUCTION_MODES, CaptureRequest.NOISE_REDUCTION_MODE_OFF))
                b.set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_OFF)
            if (has(CameraCharacteristics.HOT_PIXEL_AVAILABLE_HOT_PIXEL_MODES, CaptureRequest.HOT_PIXEL_MODE_OFF))
                b.set(CaptureRequest.HOT_PIXEL_MODE, CaptureRequest.HOT_PIXEL_MODE_OFF)
            // stabilisation warps the image; off as well in "off" mode
            if (has(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF))
                b.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
            if (has(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF))
                b.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF)
        }
        return b.build()
    }

    private fun applyRepeating(): Boolean {
        val s = session ?: return false
        return try {
            val req = buildRequest() ?: return false
            s.setRepeatingRequest(req, captureCallback, handler)
            true
        } catch (e: Exception) { Log.w(TAG, "setRepeatingRequest failed", e); false }
    }

    private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(s: CameraCaptureSession, req: CaptureRequest, res: TotalCaptureResult) {
            lastResult = res
            val ts = res.get(CaptureResult.SENSOR_TIMESTAMP)
            if (ts != null) {
                results[ts] = res
                while (results.size > 400) results.pollFirstEntry()
            }
        }
        override fun onCaptureFailed(s: CameraCaptureSession, req: CaptureRequest, f: CaptureFailure) {
            captureFailures++
            Log.w(TAG, "capture failed reason=${f.reason} frame=${f.frameNumber} (#$captureFailures)")
            if (!gotFrame && captureFailures >= 10) fail(attempt, "10 capture failures before first frame")
        }
    }

    fun setProc(mode: String) {
        handler?.post {
            if (mode != "off" && mode != "default") return@post
            procMode = mode
            Log.i(TAG, "proc mode -> $mode")
            if (session != null) applyRepeating()
        }
    }

    // ---------------------------------------------------------------- focus / lock

    /** nx, ny: normalised position in the upright (portrait) preview view, 0..1. */
    fun focusAt(nx: Float, ny: Float) {
        handler?.post {
            val c = chars ?: return@post
            val s = session ?: return@post
            val active = c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE) ?: return@post
            val o = c.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
            val (sx, sy) = when (o) {
                90 -> Pair(ny, 1f - nx)
                180 -> Pair(1f - nx, 1f - ny)
                270 -> Pair(1f - ny, nx)
                else -> Pair(nx, ny)
            }
            val half = (active.width() * 0.05f).toInt()
            val cx = active.left + (sx * active.width()).toInt()
            val cy = active.top + (sy * active.height()).toInt()
            val rect = Rect(
                (cx - half).coerceAtLeast(active.left), (cy - half).coerceAtLeast(active.top),
                (cx + half).coerceAtMost(active.right - 1), (cy + half).coerceAtMost(active.bottom - 1))
            regions = arrayOf(MeteringRectangle(rect, MeteringRectangle.METERING_WEIGHT_MAX - 1))
            Log.i(TAG, "tap view=($nx,$ny) -> sensor=($sx,$sy) rect=$rect")
            aeLock = false
            val my = ++tapToken
            try {
                if (afModes().contains(CaptureRequest.CONTROL_AF_MODE_AUTO)) {
                    afMode = CaptureRequest.CONTROL_AF_MODE_AUTO
                    s.capture(buildRequest(CaptureRequest.CONTROL_AF_TRIGGER_CANCEL)!!, null, handler)
                    s.capture(buildRequest(CaptureRequest.CONTROL_AF_TRIGGER_START)!!, null, handler)
                }
                applyRepeating()
            } catch (e: Exception) { Log.w(TAG, "focusAt failed", e) }
            listener.onMessage("Focusing...")
            handler?.postDelayed({
                if (my != tapToken) return@postDelayed
                if (lockEnabled) {
                    aeLock = true               // stay in AF_AUTO at the focused position
                } else if (afModes().contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)) {
                    afMode = CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO
                }
                applyRepeating()
            }, 2500)
        }
    }

    fun setLock(on: Boolean) {
        handler?.post {
            lockEnabled = on
            tapToken++
            if (on) {
                aeLock = true
                if (afModes().contains(CaptureRequest.CONTROL_AF_MODE_AUTO)) afMode = CaptureRequest.CONTROL_AF_MODE_AUTO
            } else {
                aeLock = false
                if (afModes().contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO))
                    afMode = CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO
                try { session?.capture(buildRequest(CaptureRequest.CONTROL_AF_TRIGGER_CANCEL)!!, null, handler) } catch (_: Exception) {}
            }
            applyRepeating()
            listener.onMessage(if (on) "AE/AF locked" else "Unlocked")
        }
    }

    // ---------------------------------------------------------------- stream + status

    private val ticker = object : Runnable {
        override fun run() {
            val now = System.currentTimeMillis()
            if (lastTickMs != 0L) {
                val dt = (now - lastTickMs) / 1000f
                if (dt > 0) fps = framesSinceTick / dt
            }
            framesSinceTick = 0; lastTickMs = now
            val c = cur
            val fd = lastResult?.get(CaptureResult.LENS_FOCUS_DISTANCE)
            val focus = if (fd == null) "?" else if (fd > 0.01f) "%.2f dpt (%.0f mm)".format(fd, 1000f / fd) else "inf"
            listener.onStatusLine(
                if (c == null || session == null) "starting camera..."
                else "cam ${c.id} | ${c.size.width}x${c.size.height} | proc $procMode | focus $focus | %.1f fps".format(fps) + (if (liveOn && liveStatus.isNotEmpty()) " | $liveStatus" else ""))
            handler?.postDelayed(this, 1000)
        }
    }

    private fun onImage(r: ImageReader, my: Int) {
        val img = try { r.acquireNextImage() } catch (e: Exception) { null } ?: return
        var handedOff = false
        try {
            if (my != attempt) return
            gotFrame = true
            framesSinceTick++; totalFrames++
            val pd = pendingDump
            if (pd != null) {
                if (settleFrames > 0) { settleFrames--; if (settleFrames == 0) beginDump(pd) }
                else beginDump(pd)
            }
            dump?.let { handleDumpFrame(it, img) }
            if (liveOn || gatherOn || scannerOn) handedOff = analyzer.submit(img, sensorOrient)   // analyzer closes img
        } finally {
            if (!handedOff) img.close()
        }
    }

    // ---------------------------------------------------------------- dump

    fun requestDump(n: Int, tag: String?, proc: String?, extras: JSONObject? = null, parent: File? = null, dirName: String? = null,
                    cb: ((File, String?) -> Unit)? = null) {
        handler?.post {
            if (dump != null || pendingDump != null) { Log.w(TAG, "StegoCam DUMP_FAILED busy"); cb?.invoke(parent ?: File("."), "busy"); return@post }
            if (session == null || reader == null) { Log.w(TAG, "StegoCam DUMP_FAILED camera not ready"); cb?.invoke(parent ?: File("."), "camera not ready"); return@post }
            pendingDump = PendingDump(n.coerceAtLeast(1), tag, extras, parent, dirName, cb)
            if (proc != null && proc != procMode && (proc == "off" || proc == "default")) {
                setProc(proc)
                settleFrames = 10
            } else settleFrames = 0
            if (settleFrames == 0) pendingDump?.let { /* starts on next frame */ }
        }
    }

    private fun sanitize(tag: String?): String =
        (tag?.replace(Regex("[^A-Za-z0-9_-]"), "")?.take(40) ?: "").ifEmpty { "manual" }

    private fun beginDump(pd: PendingDump) {
        pendingDump = null
        val tag = sanitize(pd.tag)
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val base = ctx.getExternalFilesDir("captures") ?: ctx.filesDir
        val dir = if (pd.parent != null) File(pd.parent, pd.dirName ?: "${stamp}_$tag") else File(base, "${stamp}_$tag")
        dir.mkdirs()
        val d = Dump(pd.n, tag, dir, procMode, pd.extras, pd.cb)
        dump = d
        Log.i(TAG, "dump start n=${pd.n} tag=$tag proc=$procMode dir=$dir")
        Thread({ writerLoop(d) }, "dump-writer").start()
    }

    private fun copyY(img: Image): ByteArray {
        val p = img.planes[0]
        val w = img.width; val h = img.height
        val rs = p.rowStride; val ps = p.pixelStride
        val buf = p.buffer
        val out = ByteArray(w * h)
        if (ps == 1) {
            if (rs == w) { buf.position(0); buf.get(out, 0, w * h) }
            else for (r in 0 until h) { buf.position(r * rs); buf.get(out, r * w, w) }
        } else {
            val tmp = ByteArray(rs)
            for (r in 0 until h) {
                buf.position(r * rs)
                buf.get(tmp, 0, minOf(rs, buf.remaining()))
                for (x in 0 until w) out[r * w + x] = tmp[x * ps]
            }
        }
        return out
    }

    private fun handleDumpFrame(d: Dump, img: Image) {
        val frame = d.received++
        val ts = img.timestamp
        d.timestamps.add(Pair(frame, ts))
        if (d.queue.remainingCapacity() == 0) {
            d.dropped.add(Pair(frame, ts))
            Log.w(TAG, "dump: writer queue full, dropped frame $frame")
        } else {
            d.queue.offer(Item(frame, ts, copyY(img)))
        }
        if (d.received >= d.n) {
            d.allReceived = true
            dump = null
        }
    }

    private fun writerLoop(d: Dump) {
        val w = reader?.width ?: cur?.size?.width ?: 0
        val h = reader?.height ?: cur?.size?.height ?: 0
        val size = cur?.size ?: Size(w, h)
        try {
            val header = "P5\n${size.width} ${size.height}\n255\n".toByteArray(Charsets.US_ASCII)
            while (true) {
                val it = d.queue.poll(200, TimeUnit.MILLISECONDS)
                if (it == null) {
                    if (d.allReceived && d.queue.isEmpty()) break
                    continue
                }
                FileOutputStream(File(d.dir, "frame_%03d.pgm".format(it.frame))).use { o ->
                    o.write(header); o.write(it.data)
                }
                d.written.add(it.frame)
            }
        } catch (e: Exception) {
            d.error = e.toString()
            Log.e(TAG, "dump writer failed", e)
        }
        try { writeMeta(d, size) } catch (e: Exception) { Log.e(TAG, "meta failed", e); d.error = e.toString() }
        if (d.error != null) Log.e(TAG, "StegoCam DUMP_FAILED ${d.error}")
        Log.i(TAG, "StegoCam DUMPED ${d.dir.absolutePath}")
        d.cb?.invoke(d.dir, d.error)
    }

    private fun jn(v: Any?): Any = v ?: JSONObject.NULL

    private fun writeMeta(d: Dump, size: Size) {
        // results may arrive slightly after the images; wait briefly
        val deadline = System.currentTimeMillis() + 3000
        while (System.currentTimeMillis() < deadline && d.timestamps.any { !results.containsKey(it.second) }) Thread.sleep(50)

        val ch = chars
        val root = JSONObject()
        root.put("camera_id", cur?.id)
        root.put("yuv_size", JSONArray(listOf(size.width, size.height)))
        root.put("pgm_orientation", "sensor (no rotation)")
        root.put("proc_mode", d.proc)
        root.put("template", templateName)
        root.put("tag", d.tag)
        root.put("n_requested", d.n)
        root.put("device_model", Build.MODEL)
        root.put("device_manufacturer", Build.MANUFACTURER)
        root.put("android_version", Build.VERSION.RELEASE)
        root.put("android_sdk", Build.VERSION.SDK_INT)
        val frames = JSONArray()
        var prevTs = 0L
        val gaps = JSONArray()
        val sorted = d.timestamps.sortedBy { it.first }
        for ((fr, ts) in sorted) {
            val r = results[ts]
            val f = JSONObject()
            f.put("frame", fr)
            f.put("file", if (fr in d.written) "frame_%03d.pgm".format(fr) else JSONObject.NULL)
            f.put("sensor_timestamp_ns", ts)
            f.put("result_matched", r != null)
            f.put("exposure_time_ns", jn(r?.get(CaptureResult.SENSOR_EXPOSURE_TIME)))
            f.put("iso", jn(r?.get(CaptureResult.SENSOR_SENSITIVITY)))
            f.put("frame_duration_ns", jn(r?.get(CaptureResult.SENSOR_FRAME_DURATION)))
            f.put("focus_distance_dpt", jn(r?.get(CaptureResult.LENS_FOCUS_DISTANCE)?.toDouble()))
            f.put("af_state", jn(r?.get(CaptureResult.CONTROL_AF_STATE)))
            f.put("af_mode", jn(r?.get(CaptureResult.CONTROL_AF_MODE)))
            f.put("ae_state", jn(r?.get(CaptureResult.CONTROL_AE_STATE)))
            f.put("edge_mode", jn(r?.get(CaptureResult.EDGE_MODE)))
            f.put("noise_reduction_mode", jn(r?.get(CaptureResult.NOISE_REDUCTION_MODE)))
            f.put("hot_pixel_mode", jn(r?.get(CaptureResult.HOT_PIXEL_MODE)))
            f.put("video_stabilization_mode", jn(r?.get(CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE)))
            f.put("optical_stabilization_mode", jn(r?.get(CaptureResult.LENS_OPTICAL_STABILIZATION_MODE)))
            frames.put(f)
            if (prevTs != 0L) {
                val dt = ts - prevTs
                val fd = r?.get(CaptureResult.SENSOR_FRAME_DURATION) ?: 0L
                if (fd > 0 && dt > fd * 1.5) gaps.put(JSONObject().put("before_frame", fr).put("dt_ns", dt).put("frame_duration_ns", fd))
            }
            prevTs = ts
        }
        root.put("frames", frames)
        root.put("timestamp_gaps", gaps)
        root.put("dropped_frames", JSONArray(d.dropped.map { JSONObject().put("frame", it.first).put("sensor_timestamp_ns", it.second) }))
        root.put("write_error", jn(d.error))
        if (d.extras != null) root.put("scenario", d.extras)
        root.put("intrinsics_stream_fx_fy_cx_cy", JSONArray(intrinsics.toList())).put("intrinsics_source", intrinsicsSrc)

        val st = JSONObject()
        if (ch != null) {
            fun fa(a: FloatArray?) = if (a == null) JSONObject.NULL else JSONArray(a.map { it.toDouble() })
            st.put("lens_intrinsic_calibration", fa(ch.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)))
            if (Build.VERSION.SDK_INT >= 28) st.put("lens_distortion", fa(ch.get(CameraCharacteristics.LENS_DISTORTION)))
            @Suppress("DEPRECATION")
            st.put("lens_radial_distortion", fa(ch.get(CameraCharacteristics.LENS_RADIAL_DISTORTION)))
            st.put("lens_pose_rotation", fa(ch.get(CameraCharacteristics.LENS_POSE_ROTATION)))
            st.put("lens_pose_translation", fa(ch.get(CameraCharacteristics.LENS_POSE_TRANSLATION)))
            val ps = ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
            st.put("sensor_physical_size_mm", if (ps == null) JSONObject.NULL else JSONArray(listOf(ps.width.toDouble(), ps.height.toDouble())))
            val pa = ch.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
            st.put("pixel_array_size", if (pa == null) JSONObject.NULL else JSONArray(listOf(pa.width, pa.height)))
            val aa = ch.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            st.put("active_array_size", if (aa == null) JSONObject.NULL else JSONArray(listOf(aa.left, aa.top, aa.right, aa.bottom)))
            st.put("sensor_orientation", jn(ch.get(CameraCharacteristics.SENSOR_ORIENTATION)))
            st.put("focal_lengths_mm", fa(ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)))
            st.put("min_focus_distance_dpt", jn(ch.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)?.toDouble()))
            st.put("lens_poses_reference", jn(ch.get(CameraCharacteristics.LENS_POSE_REFERENCE)))
        }
        root.put("characteristics", st)
        File(d.dir, "meta.json").writeText(root.toString(2))
    }
}
