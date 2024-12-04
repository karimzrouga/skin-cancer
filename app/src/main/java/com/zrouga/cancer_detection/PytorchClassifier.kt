package com.zrouga.cancer_detection

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.nio.FloatBuffer
import java.util.*

class PytorchClassifier(assetManager: AssetManager, modelPath: String, labelPath: String, inputSize: Int) {
    private var module: Module
    private var LABEL_LIST: List<String>
    private val INPUT_SIZE: Int = inputSize
    private val PIXEL_SIZE: Int = 3
    private val IMAGE_MEAN = 0.0f
    private val IMAGE_STD = 255.0f
    private val MAX_RESULTS = 3
    private val THRESHOLD = 0.4f

    data class Recognition(
        var id: String = "",
        var title: String = "",
        var confidence: Float = 0F
    ) {
        override fun toString(): String {
            return "Title = $title, Confidence = $confidence)"
        }
    }

    init {
        module = Module.load(assetManager.openFd(modelPath).fileDescriptor.toString())
        LABEL_LIST = loadLabelList(assetManager, labelPath)
    }

    private fun loadLabelList(assetManager: AssetManager, labelPath: String): List<String> {
        return assetManager.open(labelPath).bufferedReader().useLines { it.toList() }
    }

    fun recognizeImage(bitmap: Bitmap): List<Recognition> {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)
        val inputTensor = convertBitmapToTensor(scaledBitmap)
        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
        val scores = outputTensor.dataAsFloatArray
        return getSortedResult(scores)
    }

    private fun convertBitmapToTensor(bitmap: Bitmap): Tensor {
        val floatBuffer = FloatBuffer.allocate(INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE)
        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)

        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                val value = intValues[pixel++]
                floatBuffer.put(((value shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                floatBuffer.put(((value shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                floatBuffer.put(((value and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
            }
        }

        return Tensor.fromBlob(floatBuffer.array(), longArrayOf(1, PIXEL_SIZE.toLong(), INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))
    }

    private fun getSortedResult(scores: FloatArray): List<Recognition> {
        Log.d("Classifier", "Scores size: ${scores.size}")

        val pq = PriorityQueue(
            MAX_RESULTS,
            Comparator<Recognition> { (_, _, confidence1), (_, _, confidence2) ->
                confidence2.compareTo(confidence1)
            }
        )

        for (i in scores.indices) {
            val confidence = scores[i]
            if (confidence >= THRESHOLD) {
                pq.add(
                    Recognition(
                        id = "$i",
                        title = if (LABEL_LIST.size > i) LABEL_LIST[i] else "Unknown",
                        confidence = confidence
                    )
                )
            }
        }

        Log.d("Classifier", "pq size: ${pq.size}")

        val recognitions = ArrayList<Recognition>()
        val recognitionsSize = Math.min(pq.size, MAX_RESULTS)
        for (i in 0 until recognitionsSize) {
            recognitions.add(pq.poll())
        }
        return recognitions
    }
}
