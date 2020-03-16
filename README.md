# Smart Bill Manager [Android App, Flask Server (Image Recognition, Image Processing, OCR)

Smart Bill Manager is a solution to digitize the process of traditional bill management by providing an automated solution to scan,
extract, store and manage bills.

This repository contains the Back-End Flask Server for Smart Bill Manager.

The main purpose of this server is to receive a bill image and extract relevant text out of it and send it back to the server.
The received image goes through following processes in the server:
1) Received image goes into a VGG Classifier which classifies the received bill image.
2) After successful classification, the image is sent to the relevant YOLO Object Detection Model, which then detects segments in the image.
3) After text segments have been detected in the image, the segments are sent to the Image Processing Module, onto which number of techniques
   are applied to better the quality of segmented images.
4) The segmented images are then sent to the OCR module where text is finally extracted from the image.
5) The extracted text is converted to JSON Obj and sent back to the Android client.


Language used: Python

Major Libraries used: PyTesseract, TensorFlow, Keras, OpenCV

