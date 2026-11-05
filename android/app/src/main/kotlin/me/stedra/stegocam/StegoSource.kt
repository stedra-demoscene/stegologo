package me.stedra.stegocam

import android.content.Context
import org.opencv.core.Mat
import kotlin.math.max

/** A payload recovered from the geometry of the printed page (not from a barcode). [box] = sensor-px rectangle [x0, y0, x1, y1] of the region. */
class StegoHit(val payload: String, val z: Double, val tAcceptMs: Long, val box: IntArray, val kind: String, val detail: String)

/**
 * Pluggable stego decoder for the scanner mode. [onFrame] runs on the analysis thread for every camera frame (sensor orientation, Y plane) and must
 * return quickly: sources that need long computations do them on their own threads and return what has been accepted so far.
 * Add the token decoder (6 base-62 chars + CRC-16, two regions combined) by implementing this interface and registering it in [Sources.create].
 */
interface StegoSource {
    val name: String
    fun onFrame(full: Mat): List<StegoHit>
    fun reset()
    fun shutdown()
}

/** Maps a decoded stego payload to the text/URL shown on the result card. Replace when the token format is final. */
fun interface PayloadResolver {
    fun resolve(hit: StegoHit): String?
}

/** Stand-in resolver: a payload that already is a URL is shown as is; a hex payload becomes https://stedra.me/s/<first 12 hex>. */
class StandInResolver : PayloadResolver {
    override fun resolve(hit: StegoHit): String? =
        if (hit.payload.startsWith("http://") || hit.payload.startsWith("https://")) hit.payload
        else if (hit.payload.isNotEmpty()) "https://stedra.me/s/" + hit.payload.take(12) else null
}

/** 128-bit text blocks (sheet-4 page 6 B_15 etc.): the existing LiveDecoder path (block detector + core decode). */
class BlockStego(ctx: Context) : StegoSource {
    override val name = "blocks"
    private val decoder = LiveDecoder(ctx)
    private val hits = ArrayList<StegoHit>()
    private val seen = HashSet<String>()

    override fun onFrame(full: Mat): List<StegoHit> {
        val blocks = BlockDetector.detect(full).filter { max(it.x1 - it.x0, it.y1 - it.y0) >= 200 }
        val (views, _) = decoder.update(full, blocks)
        for ((i, v) in views.withIndex()) {
            if (v == null || v.hex == null || !v.stable) continue
            val b = blocks[i]
            val key = v.hex
            if (seen.add(key)) hits.add(StegoHit(v.hex, v.z, v.ttaMs, intArrayOf(b.x0, b.y0, b.x1, b.y1), "block", "block ${v.mode} z=%.1f".format(v.z)))
        }
        return hits.toList()
    }

    override fun reset() { decoder.reset(); hits.clear(); seen.clear() }
    override fun shutdown() { decoder.shutdown() }
}

/** Marks (rings / opart / guilloche / cartouche / swirl ...): C++ detector + decode on a worker thread (MarkLive). */
class MarkStego(ctx: Context) : StegoSource {
    override val name = "marks"
    val live = MarkLive(ctx).also { it.loadSpecs() }
    init { live.enabled = true }

    override fun onFrame(full: Mat): List<StegoHit> {
        live.offer(full)
        return live.current().filter { it.accepted }.map {
            StegoHit(it.hex, it.z, it.tMs, it.box, "mark:" + it.spec.name, "${it.spec.label} (${it.assetName}) z=%.1f %.0f ms".format(it.z, it.ms))
        }
    }

    override fun reset() { live.reset() }
    override fun shutdown() { live.shutdown() }
}

object Sources {
    /** The stego sources of the scanner. profile "demo" (default): the demo cards; "lab": sheet-4 marks and the 128-bit block. */
    fun create(ctx: Context, profile: String = "demo"): List<StegoSource> =
        if (profile == "lab") listOf(MarkStego(ctx), BlockStego(ctx)) else listOf(DemoStego(ctx))
}
