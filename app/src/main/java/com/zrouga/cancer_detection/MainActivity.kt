package com.zrouga.cancer_detection

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.pytorch.IValue
import org.pytorch.Module
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

import org.pytorch.Tensor
class MainActivity : AppCompatActivity() {

    private lateinit var mCameraButton: Button
    private lateinit var mGalleryButton: Button
    private lateinit var mDetectButton: Button
    private lateinit var mPhotoImageView: ImageView
    private lateinit var mResultTextView: TextView

    private val mCameraRequestCode = 101
    private val mGalleryRequestCode = 102

    private lateinit var mBitmap: Bitmap

    private lateinit var mClassifier: TensorClassifier


    private val mInputSize = 224
    private val mModelPath = "model.tflite"
    private val mLabelPath = "labels.txt"
    private val mSamplePath = "skin-icon.jpg"

    private lateinit var efficientnetModule: Module
    private lateinit var mobilenetModule: Module

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize UI elements
        mCameraButton = findViewById(R.id.mCameraButton)
        mGalleryButton = findViewById(R.id.mGalleryButton)
        mDetectButton = findViewById(R.id.mDetectButton)
        mPhotoImageView = findViewById(R.id.mPhotoImageView)
        mResultTextView = findViewById(R.id.mResultTextView)



        try {
          //  efficientnetModule = Module.load(assetFilePath(this, "EfficientNet.pt"))
          //  mobilenetModule = Module.load(assetFilePath(this, "MobileNet.pt"))

            mClassifier = TensorClassifier(assets, mModelPath, mLabelPath, mInputSize)

            resources.assets.open(mSamplePath).use {
                mBitmap = BitmapFactory.decodeStream(it)
                mBitmap = Bitmap.createScaledBitmap(mBitmap, mInputSize, mInputSize, true)
                mPhotoImageView.setImageBitmap(mBitmap)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Error loading image: ${e.message}", Toast.LENGTH_SHORT).show()
        }      // Set up button listeners
        mCameraButton.setOnClickListener {
            if (checkAndRequestPermissions()) {
                val callCameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(callCameraIntent, mCameraRequestCode)
            }
        }

        mGalleryButton.setOnClickListener {
            if (checkAndRequestPermissions()) {
                val callGalleryIntent = Intent(Intent.ACTION_PICK)
                callGalleryIntent.type = "image/*"
                startActivityForResult(callGalleryIntent, mGalleryRequestCode)
            }
        }

        mDetectButton.setOnClickListener {
            val results = mClassifier.recognizeImage(mBitmap).firstOrNull()
            mResultTextView.text = results?.title + "\n Confidence: " + results?.confidence
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == mCameraRequestCode && resultCode == RESULT_OK && data != null) {
            mBitmap = data.extras!!.get("data") as Bitmap
            mBitmap = scaleImage(mBitmap)
            mPhotoImageView.setImageBitmap(mBitmap)
            mResultTextView.text = "Photo captured and set successfully!"
        } else if (requestCode == mGalleryRequestCode && resultCode == RESULT_OK && data != null) {
            val uri = data.data
            try {
                mBitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
                mBitmap = scaleImage(mBitmap)
                mPhotoImageView.setImageBitmap(mBitmap)
                mResultTextView.text = "Gallery image set successfully!"
            } catch (e: Exception) {
                e.printStackTrace()
                Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
            }
        }
    }


    // Method to load the model file from assets
    private fun assetFilePath(context: Context, assetName: String): String? {
        try {
            val file = File(context.filesDir, assetName)
            if (!file.exists()) {
                context.assets.open(assetName).use { inputStream ->
                    FileOutputStream(file).use { outputStream ->
                        val buffer = ByteArray(4 * 1024)
                        var bytesRead: Int
                        while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                            outputStream.write(buffer, 0, bytesRead)
                        }
                    }
                }
            }
            return file.absolutePath
        } catch (e: IOException) {
            Log.e("MainActivity", "Error reading asset file", e)
            return null
        }
    }
    // Check and request permissions
    private fun checkAndRequestPermissions(): Boolean {
        val cameraPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
        val storagePermission =
            ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)

        val listPermissionsNeeded = mutableListOf<String>()

        if (cameraPermission != PackageManager.PERMISSION_GRANTED) {
            listPermissionsNeeded.add(Manifest.permission.CAMERA)
        }
        if (storagePermission != PackageManager.PERMISSION_GRANTED) {
            listPermissionsNeeded.add(Manifest.permission.READ_EXTERNAL_STORAGE)
        }

        if (listPermissionsNeeded.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, listPermissionsNeeded.toTypedArray(), 100)
            return false
        }
        return true
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100) {
            val perms = mutableMapOf<String, Int>()
            permissions.forEachIndexed { index, permission ->
                perms[permission] = grantResults[index]
            }

            val allGranted = perms.values.all { it == PackageManager.PERMISSION_GRANTED }
            if (!allGranted) {
                Toast.makeText(this, "Permissions Denied. Cannot proceed!", Toast.LENGTH_SHORT)
                    .show()
            }
        }
    }

    // Scale image to desired size
    private fun scaleImage(bitmap: Bitmap): Bitmap {
        val desiredWidth = 224  // Replace with your desired width
        val desiredHeight = 224 // Replace with your desired height
        return Bitmap.createScaledBitmap(bitmap, desiredWidth, desiredHeight, true)
    }
}
