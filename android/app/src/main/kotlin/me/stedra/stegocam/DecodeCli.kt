package me.stedra.stegocam

import android.content.Context
import android.util.Log
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import java.io.File

/**
 * Offline decode of crops of a pushed PGM (bit-exactness test against Python):
 *   adb push frame_000.pgm /sdcard/Android/data/me.stedra.stegocam/files/in/
 *   adb shell am broadcast -a me.stedra.stegocam.DECODE_FILE -p me.stedra.stegocam --es path <abs path> --es boxes "x0,y0,x1,y1;..."
 * Boxes are the crop boxes (decode_capture.crop_box) in full-frame pixels. Logs `StegoCam DECODE_RESULT {json}` per box.
 */
object DecodeCli {
    private const val TAG = "StegoCam"

    fun readPgm(f: File): Mat {
        val b = f.readBytes()
        var pos = 0
        fun token(): String {
            while (b[pos].toInt().toChar().isWhitespace()) pos++
            if (b[pos].toInt().toChar() == '#') { while (b[pos].toInt().toChar() != '\n') pos++; return token() }
            val st = pos; while (!b[pos].toInt().toChar().isWhitespace()) pos++
            return String(b, st, pos - st)
        }
        require(token() == "P5")
        val w = token().toInt(); val h = token().toInt(); token(); pos++
        val m = Mat(h, w, CvType.CV_8UC1); m.put(0, 0, b, pos, w * h)
        return m
    }

    /** opts: "pin=1" (prime cores) / "pin=0", "reps=N" (repeat each decode N times, log all ms), "prio=-8" (thread priority). */
    fun run(ctx: Context, path: String?, boxes: String?, opts: String? = null) {
        if (path == null || boxes == null) { Log.e(TAG, "DECODE_RESULT {\"error\":\"missing path/boxes\"}"); return }
        val o = (opts ?: "").split(",").filter { it.contains("=") }.associate { it.substringBefore("=") to it.substringAfter("=") }
        Thread {
            o["prio"]?.let { android.os.Process.setThreadPriority(it.toInt()) }
            try {
                StegoCore.init(ctx)
                if (!LiveAnalyzer.ensureCv()) throw RuntimeException("opencv")
                if (o["pin"] == "1") Log.i(TAG, "AFFINITY rc=${StegoCore.setAffinity(android.os.Process.myTid(), 0xC0L)} now=0x%x".format(StegoCore.affinity(android.os.Process.myTid())))
                val img = readPgm(File(path))
                if (boxes == "diag") {   // detector + grid of (crop margin x asset x wide windows) per block, for debugging block decoding
                    val t = System.nanoTime()
                    val bl = BlockDetector.detect(img).filter { maxOf(it.w, it.h) >= 200 }
                    Log.i(TAG, "DIAG file=${File(path).name} detect_ms=%.0f n=%d".format((System.nanoTime() - t) / 1e6, bl.size))
                    for ((bi, b) in bl.withIndex()) {
                        Log.i(TAG, "DIAG blk=$bi box=${b.x0},${b.y0},${b.x1},${b.y1} w=${b.w} h=${b.h} g=%.1f long_over_g=%.2f aspect=%.2f clipped=${b.clipped}".format(b.g, maxOf(b.w, b.h) / b.g, maxOf(b.w, b.h).toDouble() / minOf(b.w, b.h)))
                        for (mg in doubleArrayOf(0.08, 0.04)) {
                            val cb = BlockDetector.cropBox(b, img.cols(), img.rows(), mg)
                            val crop = img.submat(Rect(cb[0], cb[1], cb[2] - cb[0], cb[3] - cb[1]))
                            for (a in 0 until StegoCore.nAssets) for (wide in booleanArrayOf(false, true)) {
                                val x = StegoCore.decode(crop, a, wide, a == 1 && o["fast"] != "0") ?: continue
                                Log.i(TAG, "DIAG blk=$bi mg=$mg asset=$a wide=$wide acc=${x.accept} crc=${x.crcOk} zcal=%.1f reg=%.2f core_ms=%.0f ${x.stages} hex=${x.hex}".format(x.zCal, x.regScore, x.totalMs))
                            }
                        }
                    }
                    img.release(); return@Thread
                }
                if (boxes == "marks") {   // detect + decode every mark spec on the pushed frame (repeat for exclusions), timing per call
                    val specs = MarkSpec.parse(ctx)
                    val reps = o["reps"]?.toInt() ?: 1
                    for (sp in specs) {
                        val hs = sp.assetNames.map { StegoCore.loadAsset(ctx, it) }.filter { it >= 0 }
                        if (hs.isEmpty()) continue
                        sp.assets = hs.toIntArray(); sp.det = StegoCore.markInit(hs[0])
                        val ex = ArrayList<Double>()
                        for (round in 0 until (o["rounds"]?.toInt() ?: 4)) {
                            var out: MarkOutcome? = null; val all = ArrayList<Double>()
                            repeat(reps) {
                                val t0 = System.nanoTime()
                                out = StegoCore.decodeMark(sp.det, sp.assets, img, sp.ppmLo, sp.ppmHi, sp.sym, if (o["fast"] == "0") 0 else 2, ex.toDoubleArray())
                                all.add((System.nanoTime() - t0) / 1e6)
                            }
                            val r = out
                            if (r == null || !r.found) { Log.i(TAG, "MARKS_RESULT {\"spec\":\"${sp.name}\",\"round\":$round,\"found\":false}"); break }
                            val c = r.core
                            val acc = c != null && c.accept && c.zCal >= MarkLive.Z_ACCEPT
                            val d = r.detMs
                            Log.i(TAG, "MARKS_RESULT {\"file\":\"${File(path).name}\",\"spec\":\"${sp.name}\",\"round\":$round,\"score\":%.3f,\"iso\":%.3f,\"ppm\":%.3f,\"rot\":%.2f,\"cx\":%.1f,\"cy\":%.1f,".format(r.score, r.iso, r.ppm, r.rotDeg, r.cx, r.cy) +
                                "\"asset\":${r.asset},\"hyp\":${r.hyp},\"accept\":$acc,\"hex\":\"${c?.hex}\",\"z_cal\":%.2f,\"ms\":%.1f,\"ms_min\":%.1f,\"detect_ms\":%.1f,\"decode_ms\":%.1f,".format(c?.zCal ?: 0.0, all.sorted()[all.size / 2], all.min(), r.msDetect, r.msDecode) +
                                "\"stages_ms\":\"prep=%.1f stageA=%.1f polar=%.1f fine=%.1f ecc=%.1f\",\"n_decodes\":${r.nDecodes}}".format(d[0], d[1], d[2], d[3], d[4]))
                            if (r.score < MarkLive.DETECT_OK) break
                            ex.add(r.cx); ex.add(r.cy); ex.add(0.5 * maxOf(r.crop[2] - r.crop[0], r.crop[3] - r.crop[1]) * 0.8)
                        }
                    }
                    img.release(); return@Thread
                }
                if (boxes == "scan") {   // replay a pushed frame (opts qr=<second pgm> alternates) through the scanner engine; events go to scan_events.log
                    val e = ScannerEngine(ctx, Sources.create(ctx, o["profile"] ?: "demo"))
                    e.useQr = o["src"] != "stego"; e.useStego = o["src"] != "qr"
                    val frames = listOf(img) + (o["qr"]?.let { listOf(readPgm(File(it))) } ?: emptyList())
                    e.reset()
                    val t0 = System.currentTimeMillis(); val secs = (o["secs"] ?: "10").toDouble(); var n = 0
                    while ((System.currentTimeMillis() - t0) / 1000.0 < secs && (e.winner == null || o["all"] == "1")) {
                        val tt = System.nanoTime()
                        e.onFrame(frames[n++ % frames.size], 90)
                        Thread.sleep(maxOf(1L, 33 - ((System.nanoTime() - tt) / 1e6).toLong()))
                    }
                    Thread.sleep(((o["tail"] ?: "0").toDouble() * 1000).toLong())
                    Log.i(TAG, "SCAN_DONE file=${File(path).name} winner=${e.winner?.source} qr=${e.qr?.tMs} stego=${e.stego?.tMs} text=${e.winner?.text}")
                    e.shutdown(); img.release(); return@Thread
                }
                if (boxes == "live") {   // replay one pushed frame through the live path (detector -> LiveDecoder) at ~30 fps for secs=N; events go to live_events.log
                    val ld = LiveDecoder(ctx); ld.reset()
                    val t0 = System.currentTimeMillis(); val secs = (o["secs"] ?: "8").toDouble()
                    while ((System.currentTimeMillis() - t0) / 1000.0 < secs) {
                        val tt = System.nanoTime()
                        val bl = BlockDetector.detect(img).filter { maxOf(it.w, it.h) >= 200 }
                        ld.update(img, bl)
                        val dt = (System.nanoTime() - tt) / 1e6
                        Thread.sleep(maxOf(1L, 33 - dt.toLong()))
                    }
                    Log.i(TAG, "DIAG live done file=${File(path).name}"); ld.shutdown(); img.release(); return@Thread
                }
                var list: String = boxes.trim().trim('\'', '"')
                if (boxes == "auto") {   // run the live block detector on the file and decode its crops
                    val t = System.nanoTime()
                    val bl = BlockDetector.detect(img).sortedWith(compareBy({ Math.round(((it.y0 + it.y1) / 2.0) / 450) }, { it.x0 }))
                    Log.i(TAG, "DETECT ms=%.0f n=%d".format((System.nanoTime() - t) / 1e6, bl.size))
                    list = bl.joinToString(";") { BlockDetector.cropBox(it, img.cols(), img.rows()).joinToString(",") }
                    Log.i(TAG, "DETECT boxes=$list")
                }
                val reps = o["reps"]?.toInt() ?: 1
                for ((i, bx) in list.split(";").withIndex()) {
                    val v = bx.split(",").map { it.trim().toInt() }
                    val crop = img.submat(Rect(v[0], v[1], v[2] - v[0], v[3] - v[1]))   // in place, no copy
                    var r: CoreResult? = null; val all = ArrayList<Double>()
                    repeat(reps) {
                        val t0 = System.nanoTime()
                        r = StegoCore.decode(crop)
                        all.add((System.nanoTime() - t0) / 1e6)
                    }
                    val x = r
                    if (x == null) { Log.i(TAG, "DECODE_RESULT {\"file\":\"${File(path).name}\",\"i\":$i,\"error\":\"null\"}"); continue }
                    val ms = all.sorted()[all.size / 2]
                    Log.i(TAG, "DECODE_RESULT {\"file\":\"${File(path).name}\",\"i\":$i,\"hex\":\"${x.hex}\",\"crc_ok\":${x.crcOk}," +
                        "\"accept\":${x.accept},\"z\":${x.z},\"amp_um\":${x.ampUm},\"amp_raw_um\":${x.ampRawUm},\"n_valid\":${x.nValid}," +
                        "\"reg_score\":${x.regScore},\"H\":${x.H.joinToString(",", "[", "]")},\"ms\":$ms,\"core_ms\":${x.totalMs}," +
                        "\"stages\":\"${x.stages}\"}")
                }
                img.release()
            } catch (e: Throwable) { Log.e(TAG, "DECODE_RESULT {\"error\":\"$e\"}", e) }
        }.start()
    }
}
