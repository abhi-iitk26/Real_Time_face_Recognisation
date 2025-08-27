# Real-Time Face Recognition System using CNN

## 🎯 Project Overview

This project implements a comprehensive **Real-Time Face Recognition System** using Convolutional Neural Networks (CNN). The system is designed to recognize faces in real-time through webcam feed, making it suitable for applications like attendance systems, security access control, and identity verification.

## 🏗️ Project Architecture

The project consists of three main components:

### 1. **Data Collection & Web Scraping** (`Scraping Data.ipynb`)
- **Automated Data Collection**: Web scraping implementation using Selenium to collect student photos from institutional websites
- **Multi-source Image Extraction**: Extracts images from multiple URL sources with fallback mechanisms
- **Data Organization**: Automatically downloads and organizes images with proper naming conventions
- **Error Handling**: Robust error handling for failed downloads and invalid image formats

### 2. **Data Preprocessing & Model Training** (`Real_time_face_recognition.ipynb`)
- **Face Detection**: Uses MTCNN (Multi-task Cascade CNN) for accurate face detection and extraction
- **Data Augmentation**: Implements comprehensive data augmentation using ImageDataGenerator:
  - Rotation (20°)
  - Width/Height shifts (20%)
  - Shear and zoom transformations
  - Horizontal flipping
  - Fill mode handling
- **Dataset Organization**: Automatically splits data into training and testing sets
- **Face Preprocessing**: Standardizes face images to 160x160 pixels for consistent input

### 3. **CNN Model Architecture**
- **Face Extraction Pipeline**: Custom FACELOADING class for efficient face processing
- **Model Training**: CNN-based architecture for face recognition
- **Real-time Processing**: Optimized for real-time face recognition with minimal latency

## 🔧 Technical Stack

### **Core Technologies:**
- **Deep Learning Framework**: TensorFlow/Keras
- **Computer Vision**: OpenCV (cv2)
- **Face Detection**: MTCNN (Multi-task Cascade CNN)
- **Web Scraping**: Selenium WebDriver
- **Image Processing**: PIL (Python Imaging Library)
- **Data Manipulation**: NumPy, Pandas

### **Development Environment:**
- **Language**: Python 3.x
- **Environment**: Jupyter Notebook
- **Browser Automation**: Chrome WebDriver
- **Package Management**: pip

## 🚀 Key Features

### **1. Automated Data Collection**
- Web scraping from multiple sources
- Automatic fallback URL handling
- Bulk image downloading with proper organization
- Student information extraction (name, roll number, branch)

### **2. Advanced Face Processing**
- MTCNN-based face detection for high accuracy
- Automatic face extraction and cropping
- Face standardization to consistent dimensions
- Real-time face detection capabilities

### **3. Data Augmentation Pipeline**
- Generates multiple variations of training images
- Implements 10+ augmentation techniques
- Automatic train/test split generation
- Maintains data quality and consistency

### **4. CNN Model Features**
- Custom CNN architecture for face recognition
- Optimized for real-time performance
- Scalable to multiple face classes
- Efficient memory usage

## 📊 Project Workflow

```
1. Data Collection (Scraping Data.ipynb)
   ├── Web scraping setup
   ├── Student data extraction
   ├── Image downloading
   └── Data organization

2. Preprocessing (Real_time_face_recognition.ipynb)
   ├── Face detection using MTCNN
   ├── Face extraction and cropping
   ├── Data augmentation
   └── Dataset preparation

3. Model Training
   ├── CNN architecture design
   ├── Model compilation
   ├── Training process
   └── Model evaluation

4. Real-time Recognition
   ├── Webcam integration
   ├── Real-time face detection
   ├── Face recognition
   └── Result display
```

## 🎯 Applications

### **Educational Institutions:**
- Automated attendance systems
- Student identification
- Access control for labs/libraries

### **Security Systems:**
- Building access control
- Employee verification
- Visitor management

### **Commercial Applications:**
- Customer identification
- Personalized services
- Security monitoring

## 🔍 Technical Highlights

### **1. Face Detection Accuracy**
- Uses state-of-the-art MTCNN for robust face detection
- Handles various lighting conditions
- Detects faces at different angles and scales

### **2. Data Augmentation Strategy**
- Comprehensive augmentation pipeline
- Generates 10+ variations per original image
- Maintains facial feature integrity
- Improves model generalization

### **3. Real-time Performance**
- Optimized CNN architecture
- Efficient face preprocessing
- Minimal latency for real-time applications
- Scalable to multiple simultaneous users

### **4. Robust Data Collection**
- Multi-source web scraping
- Automatic error recovery
- Data validation and cleaning
- Organized file structure

## 📈 Performance Metrics

- **Face Detection Accuracy**: High precision using MTCNN
- **Real-time Processing**: Optimized for live video streams
- **Data Augmentation**: 10x increase in training data
- **Model Efficiency**: Balanced accuracy and speed

## 🛠️ Installation & Setup

### **Prerequisites:**
```bash
pip install tensorflow opencv-python mtcnn selenium webdriver-manager pillow numpy
```

### **WebDriver Setup:**
- Chrome WebDriver (automatically managed by webdriver-manager)
- Chrome browser installation required

## 💡 Innovation Aspects

1. **Automated Data Pipeline**: End-to-end automation from data collection to model training
2. **Multi-source Scraping**: Robust data collection with fallback mechanisms
3. **Real-time Processing**: Optimized for live applications
4. **Scalable Architecture**: Easily extensible to new face classes
5. **Production Ready**: Comprehensive error handling and validation

## 🔮 Future Enhancements

- **Model Optimization**: Implement transfer learning with pre-trained models
- **Mobile Deployment**: Convert to mobile-friendly format (TensorFlow Lite)
- **Database Integration**: Add database support for large-scale deployments
- **API Development**: REST API for integration with other systems
- **Performance Monitoring**: Add metrics tracking and logging

## 📋 Project Structure

```
Real_Time_face_Recognisation/
├── Real_time_face_recognition.ipynb    # Main training and recognition notebook
├── Scraping Data.ipynb                 # Data collection and web scraping
├── LICENSE                             # License file
├── LICENSE.md                          # License documentation
├── README.md                           # Project documentation
└── PROJECT_DESCRIPTION.md              # This detailed description
```

---

**Developed by**: Abhishek (abhi-iitk26)  
**Repository**: [Real_Time_face_Recognisation](https://github.com/abhi-iitk26/Real_Time_face_Recognisation)  
**Technology Stack**: Python, TensorFlow, OpenCV, MTCNN, Selenium
