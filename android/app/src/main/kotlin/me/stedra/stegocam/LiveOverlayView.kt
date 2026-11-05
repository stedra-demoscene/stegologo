package me.stedra.stegocam

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.view.View

/** Draws normalised upright boxes (+ decode labels inside the box, clipped to its width) over the preview view's bounds. */
class LiveOverlayView(context: Context, private val preview: View) : View(context) {
    @Volatile private var boxes: List<FloatArray> = emptyList()
    @Volatile private var labels: List<LiveAnalyzer.Label?> = emptyList()
    private val d = resources.displayMetrics.density
    private val sp = resources.displayMetrics.scaledDensity
    private val colours = intArrayOf(Color.argb(240, 0, 255, 80), Color.argb(240, 255, 210, 0), Color.argb(240, 255, 60, 60))
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 2f * d }
    private val big = Paint(Paint.ANTI_ALIAS_FLAG).apply { typeface = Typeface.MONOSPACE; isFakeBoldText = true }
    private val small = Paint(Paint.ANTI_ALIAS_FLAG).apply { typeface = Typeface.DEFAULT }
    private val bg = Paint().apply { color = Color.argb(185, 0, 0, 0) }

    fun showBoxes(b: List<FloatArray>, l: List<LiveAnalyzer.Label?>) { boxes = b; labels = l; postInvalidate() }

    /** Largest text size <= max (and >= min) whose width fits [avail]. */
    private fun fit(p: Paint, text: String, maxSp: Float, minSp: Float, avail: Float) {
        p.textSize = maxSp * sp
        val w = p.measureText(text)
        if (w > avail) p.textSize = (maxSp * sp * avail / w).coerceAtLeast(minSp * sp)
    }

    override fun onDraw(canvas: Canvas) {
        if (preview.width == 0) return
        val l = preview.left.toFloat(); val t = preview.top.toFloat()
        val w = preview.width.toFloat(); val h = preview.height.toFloat()
        val lb = labels
        for ((i, b) in boxes.withIndex()) {
            val lab = lb.getOrNull(i)
            val col = if (lab == null) Color.argb(240, 0, 200, 255) else colours[lab.colour]
            paint.color = col
            val x0 = l + b[0] * w; val y0 = t + b[1] * h; val x1 = l + b[2] * w; val y1 = t + b[3] * h
            canvas.drawRect(x0, y0, x1, y1, paint)
            if (lab == null) continue
            val avail = (x1 - x0) - 6f * d
            if (avail < 20f * d) continue
            val accepted = lab.colour == 0
            big.color = col; small.color = Color.WHITE
            // payload line: prominent when accepted
            fit(big, lab.hex.ifEmpty { "--------" }, if (accepted) 24f else 13f, 8f, avail)
            fit(small, lab.sub, 10f, 6f, avail)
            val hBig = big.textSize; val hSmall = small.textSize
            val boxH = hBig + hSmall + 10f * d
            val inside = (y1 - y0) > boxH * 1.5f
            val top = if (inside) y1 - boxH - 2f * d else y1 + 2f * d
            canvas.save()
            canvas.clipRect(x0, if (inside) y0 else top, x1, if (inside) y1 else top + boxH)
            canvas.drawRect(x0, top, x1, top + boxH, bg)
            canvas.drawText(lab.hex.ifEmpty { "--------" }, x0 + 3f * d, top + hBig + 1f * d, big)
            canvas.drawText(lab.sub, x0 + 3f * d, top + hBig + hSmall + 4f * d, small)
            canvas.restore()
        }
    }
}
