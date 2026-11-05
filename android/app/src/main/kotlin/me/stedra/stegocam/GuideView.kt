package me.stedra.stegocam

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View

/** Draws a centred 85:14 guide rectangle (horizontal or vertical) over the preview view. */
class GuideView(context: Context, private val preview: View) : View(context) {
    var vertical = false
        set(v) { field = v; invalidate() }

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(230, 255, 230, 0)
        style = Paint.Style.STROKE
        strokeWidth = 3f * resources.displayMetrics.density
    }
    private val cross = Paint(paint).apply { strokeWidth = 1f * resources.displayMetrics.density }

    override fun onDraw(canvas: Canvas) {
        if (preview.width == 0) return
        val cx = preview.left + preview.width / 2f
        val cy = preview.top + preview.height / 2f
        val long = (if (vertical) preview.height else preview.width) * 0.8f
        val short = long * 14f / 85f
        val w = if (vertical) short else long
        val h = if (vertical) long else short
        canvas.drawRect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, paint)
        canvas.drawLine(cx - w / 2, cy, cx + w / 2, cy, cross)
        canvas.drawLine(cx, cy - h / 2, cx, cy + h / 2, cross)
    }
}
