# Real-Time Face Recognition System using CNN

## 🎯 Overview
A comprehensive **Real-Time Face Recognition System** using CNNs for webcam-based face recognition. Suitable for attendance systems, security access control, and identity verification.

---

## 🏗️ System Architecture

### Three Main Components:

**1. Data Collection (`Scraping Data.ipynb`)**
- Automated web scraping using Selenium
- Multi-source image extraction with fallback mechanisms
- Organized data storage with proper naming

**2. Preprocessing & Training (`Real_time_face_recognition.ipynb`)**
- MTCNN for accurate face detection
- Comprehensive data augmentation (rotation, shifts, zoom, flipping)
- Automatic train/test split
- Face standardization to 160x160 pixels

**3. CNN Model**
- Custom CNN architecture for face recognition
- Real-time processing with minimal latency
- Scalable to multiple face classes

---

## 🔧 Tech Stack

**Core Technologies:**
- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Face Detection**: MTCNN (Multi-task Cascade CNN)
- **Web Scraping**: Selenium WebDriver
- **Image Processing**: PIL, NumPy

---

## 🚀 Key Features

- **Automated Data Pipeline**: End-to-end from scraping to recognition
- **Advanced Face Processing**: MTCNN-based detection with high accuracy
- **Data Augmentation**: 10+ techniques generating 10x training data
- **Real-time Recognition**: Optimized for live video streams
- **Robust Error Handling**: Automatic recovery and data validation

---

## 📊 Workflow

```
Data Collection → Face Detection → Data Augmentation → CNN Training → Real-time Recognition
   (Selenium)       (MTCNN)      (ImageDataGen)     (Custom CNN)    (Webcam Feed)
