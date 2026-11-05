package me.stedra.stegocam

import android.content.Context
import android.view.TextureView

/** TextureView that letterboxes itself to a given aspect ratio (no stretching). */
class AutoFitTextureView(context: Context) : TextureView(context) {
    private var aw = 0
    private var ah = 0

    fun setAspect(w: Int, h: Int) {
        aw = w; ah = h
        requestLayout()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val maxW = MeasureSpec.getSize(widthMeasureSpec)
        val maxH = MeasureSpec.getSize(heightMeasureSpec)
        if (aw == 0 || ah == 0) {
            setMeasuredDimension(maxW, maxH)
            return
        }
        // fit inside (maxW, maxH)
        if (maxW * ah <= maxH * aw) {
            setMeasuredDimension(maxW, maxW * ah / aw)
        } else {
            setMeasuredDimension(maxH * aw / ah, maxH)
        }
    }
}
