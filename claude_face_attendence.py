import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, date
import os
import pickle
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
import time
import logging
import threading
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config with custom styling
st.set_page_config(
    page_title="IITK Students Attendence System", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📹"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    .status-stopped {
        color: #dc3545;
        font-weight: bold;
    }
    .attendance-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎯 IITK Students Attendence System</h1>
    <p>Real-time attendance tracking with AI-powered face recognition</p>
</div>
""", unsafe_allow_html=True)

# --- Enhanced Initialization with Error Handling ---
@st.cache_resource
def load_models():
    """Load models with error handling and progress indication"""
    try:
        with st.spinner("🔄 Loading AI models..."):
            facenet = FaceNet()
            detector = MTCNN()
            
            # Check if model files exist
            if not os.path.exists("svm_model_160x160.pkl"):
                st.error("❌ SVM model file not found! Please ensure 'svm_model_160x160.pkl' exists.")
                return None, None, None, None
                
            if not os.path.exists("filtered_faces_embeddings.npz"):
                st.error("❌ Face embeddings file not found! Please ensure 'filtered_faces_embeddings.npz' exists.")
                return None, None, None, None
            
            model = pickle.load(open("svm_model_160x160.pkl", "rb"))
            faces_embeddings = np.load("filtered_faces_embeddings.npz")
            Y = faces_embeddings['arr_1']
            encoder = LabelEncoder()
            encoder.fit(Y)
            
            st.success("✅ Models loaded successfully!")
            return facenet, detector, model, encoder
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None

# Load models
facenet, detector, model, encoder = load_models()

# Check if models loaded successfully
if any(x is None for x in [facenet, detector, model, encoder]):
    st.stop()

# --- Configuration Section ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Threshold setting
    threshold = st.slider(
        "Recognition Threshold", 
        min_value=0.01, 
        max_value=0.99, 
        value=0.5, 
        step=0.01,
        help="Higher values = more strict recognition"
    )
    
    # Camera settings
    st.subheader("📹 Camera Settings")
    camera_index = st.selectbox("Camera Index", [0, 1, 2], index=0)
    frame_rate = st.slider("Frame Rate (FPS)", 1, 30, 10)
    
    # File settings
    st.subheader("📁 File Settings")
    attendance_file = st.text_input("Attendance File", value="attendance.xlsx")
    
    # Auto-save setting
    auto_save = st.checkbox("Auto-save attendance", value=True)

# --- Initialize Attendance Data ---
def load_attendance_data():
    """Load existing attendance data with error handling"""
    try:
        if os.path.exists(attendance_file):
            df_existing = pd.read_excel(attendance_file)
            # Ensure required columns exist
            if 'Name' not in df_existing.columns or 'Time' not in df_existing.columns:
                df_existing = pd.DataFrame(columns=["Name", "Time", "Date"])
        else:
            df_existing = pd.DataFrame(columns=["Name", "Time", "Date"])
        return df_existing
    except Exception as e:
        logger.error(f"Error loading attendance data: {e}")
        st.warning(f"⚠️ Could not load existing attendance data: {e}")
        return pd.DataFrame(columns=["Name", "Time", "Date"])

# --- Enhanced Functions ---
def get_embedding(face_img):
    """Extract face embedding with error handling"""
    try:
        face_img = face_img.astype('float32')
        face_img = np.expand_dims(face_img, axis=0)
        yhat = facenet.embeddings(face_img)
        return yhat[0]
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None

def save_attendance(name, df_existing):
    """Save attendance with enhanced data tracking"""
    try:
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if person already marked attendance today
        today_attendance = df_existing[df_existing['Date'] == date_str]
        if name in today_attendance['Name'].values:
            return False, "Already marked today"
        
        # Add new attendance record
        new_record = pd.DataFrame([[name, time_str, date_str, datetime_str]], 
                                columns=["Name", "Time", "Date", "DateTime"])
        df_updated = pd.concat([df_existing, new_record], ignore_index=True)
        
        if auto_save:
            df_updated.to_excel(attendance_file, index=False)
        
        return True, f"Attendance marked at {time_str}"
    except Exception as e:
        logger.error(f"Error saving attendance: {e}")
        return False, f"Error: {str(e)}"

def recognize_faces(frame, df_existing):
    """Enhanced face recognition with better error handling"""
    try:
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_img)
        recognized_faces = []
        
        for face in faces:
            try:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)  # Ensure positive coordinates
                confidence = face['confidence']
                
                # Skip low-confidence detections
                if confidence < 0.9:
                    continue
                
                face_img = rgb_img[y:y+h, x:x+w]
                
                if face_img.size == 0:
                    continue
                    
                face_img = cv2.resize(face_img, (160, 160))
                embedding = get_embedding(face_img)
                
                if embedding is None:
                    continue
                
                probabilities = model.predict_proba([embedding])[0]
                max_prob = np.max(probabilities)
                
                if max_prob < threshold:
                    name = "Unknown"
                    status = ""
                else:
                    name = encoder.inverse_transform([np.argmax(probabilities)])[0]
                    success, message = save_attendance(name, df_existing)
                    status = "✅" if success else "⚠️"
                
                recognized_faces.append((x, y, w, h, name, max_prob, status))
                
            except Exception as e:
                logger.error(f"Error processing face: {e}")
                continue
                
        return recognized_faces
    except Exception as e:
        logger.error(f"Error in face recognition: {e}")
        return []

# --- Session State Initialization ---
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'attendance_data' not in st.session_state:
    st.session_state.attendance_data = load_attendance_data()
if 'recognition_stats' not in st.session_state:
    st.session_state.recognition_stats = {
        'total_recognitions': 0,
        'unique_people': 0,
        'session_start': None
    }

# --- Main Interface ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    
    # Camera controls
    control_col1, control_col2, control_col3 = st.columns(3)
    
    with control_col1:
        start_btn = st.button(
            "🟢 Start Camera", 
            disabled=st.session_state.camera_running,
            use_container_width=True
        )
    
    with control_col2:
        stop_btn = st.button(
            "🔴 Stop Camera", 
            disabled=not st.session_state.camera_running,
            use_container_width=True
        )
    
    with control_col3:
        refresh_btn = st.button(
            "🔄 Refresh Data",
            use_container_width=True
        )
    
    # Status indicator
    if st.session_state.camera_running:
        st.markdown('<p class="status-running">🔴 LIVE - Camera is running</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-stopped">⚫ Camera is stopped</p>', unsafe_allow_html=True)
    
    # Camera display area
    FRAME_WINDOW = st.empty()

with col2:
    st.subheader("📊 Session Statistics")
    
    # Statistics display
    stats = st.session_state.recognition_stats
    
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("👥 People Today", len(st.session_state.attendance_data[
            st.session_state.attendance_data['Date'] == date.today().strftime("%Y-%m-%d")
        ]) if 'Date' in st.session_state.attendance_data.columns else 0)
    
    with metric_col2:
        st.metric("🎯 Total Records", len(st.session_state.attendance_data))
    
    # Recent recognitions
    st.subheader("🕒 Recent Activity")
    if not st.session_state.attendance_data.empty and 'DateTime' in st.session_state.attendance_data.columns:
        recent = st.session_state.attendance_data.tail(5)[['Name', 'Time']].sort_values('Time', ascending=False)
        for _, row in recent.iterrows():
            st.text(f"• {row['Name']} - {row['Time']}")
    else:
        st.text("No recent activity")

# Handle button clicks
if start_btn:
    st.session_state.camera_running = True
    st.session_state.recognition_stats['session_start'] = datetime.now()
    st.rerun()

if stop_btn:
    st.session_state.camera_running = False
    st.rerun()

if refresh_btn:
    st.session_state.attendance_data = load_attendance_data()
    st.rerun()

# --- Enhanced Camera Logic ---
if st.session_state.camera_running:
    try:
        cap = cv2.VideoCapture(camera_index)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, frame_rate)
        
        if not cap.isOpened():
            st.error("❌ Could not open webcam. Please check your camera connection and permissions.")
            st.session_state.camera_running = False
        else:
            # Create placeholders for dynamic updates
            status_placeholder = st.empty()
            
            frame_count = 0
            last_update = time.time()
            
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to grab frame from camera. Camera may be disconnected.")
                    break
                
                frame_count += 1
                
                # Process every nth frame for performance
                if frame_count % 3 == 0:  # Process every 3rd frame
                    recognized_faces = recognize_faces(frame, st.session_state.attendance_data)
                    
                    # Draw rectangles and labels
                    for (x, y, w, h, name, confidence, status) in recognized_faces:
                        # Color coding: Green for known, Red for unknown
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        
                        # Draw rectangle
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        
                        # Prepare text with confidence
                        display_text = f"{name} ({confidence:.2f})"
                        if status:
                            display_text += f" {status}"
                        
                        # Draw text background for better readability
                        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (x, y-30), (x + text_size[0], y), color, -1)
                        cv2.putText(frame, display_text, (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display frame
                FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)
                
                # Update status every 2 seconds
                current_time = time.time()
                if current_time - last_update > 2:
                    st.session_state.attendance_data = load_attendance_data()
                    last_update = current_time
                
                # Frame rate control
                time.sleep(1/frame_rate)
                
                # Check if stop button was pressed (streamlit state update)
                if not st.session_state.camera_running:
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            FRAME_WINDOW.empty()
            st.success("✅ Camera session ended successfully.")
            
    except Exception as e:
        st.error(f"❌ Camera error: {str(e)}")
        logger.error(f"Camera error: {e}")
        if 'cap' in locals():
            cap.release()
        st.session_state.camera_running = False

# --- Enhanced Attendance Management ---
st.markdown("---")

# Attendance section
attendance_col1, attendance_col2 = st.columns([3, 1])

with attendance_col1:
    st.subheader("📋 Attendance Management")

with attendance_col2:
    # Export options
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        if st.button("📥 Export CSV"):
            if not st.session_state.attendance_data.empty:
                csv = st.session_state.attendance_data.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"attendance_{date.today().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    with export_col2:
        if st.button("🗑️ Clear Today"):
            today_str = date.today().strftime("%Y-%m-%d")
            if 'Date' in st.session_state.attendance_data.columns:
                st.session_state.attendance_data = st.session_state.attendance_data[
                    st.session_state.attendance_data['Date'] != today_str
                ]
                if auto_save:
                    st.session_state.attendance_data.to_excel(attendance_file, index=False)
                st.success("Today's attendance cleared!")
                st.rerun()

# Attendance display with filtering
if not st.session_state.attendance_data.empty:
    # Date filter
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        if 'Date' in st.session_state.attendance_data.columns:
            available_dates = sorted(st.session_state.attendance_data['Date'].unique(), reverse=True)
            selected_date = st.selectbox("Filter by Date", ["All Dates"] + list(available_dates))
        else:
            selected_date = "All Dates"
    
    with filter_col2:
        if 'Name' in st.session_state.attendance_data.columns:
            available_names = sorted(st.session_state.attendance_data['Name'].unique())
            selected_name = st.selectbox("Filter by Name", ["All Names"] + list(available_names))
        else:
            selected_name = "All Names"
    
    with filter_col3:
        records_per_page = st.selectbox("Records per page", [10, 25, 50, 100], index=1)
    
    # Apply filters
    filtered_data = st.session_state.attendance_data.copy()
    
    if selected_date != "All Dates":
        filtered_data = filtered_data[filtered_data['Date'] == selected_date]
    
    if selected_name != "All Names":
        filtered_data = filtered_data[filtered_data['Name'] == selected_name]
    
    # Display filtered data
    if not filtered_data.empty:
        st.dataframe(
            filtered_data.sort_values('DateTime' if 'DateTime' in filtered_data.columns else 'Time', ascending=False),
            use_container_width=True,
            height=400
        )
        
        # Summary statistics
        st.markdown("### 📈 Summary")
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            st.metric("Total Records", len(filtered_data))
        
        with summary_col2:
            st.metric("Unique People", filtered_data['Name'].nunique() if 'Name' in filtered_data.columns else 0)
        
        with summary_col3:
            if 'Date' in filtered_data.columns:
                st.metric("Date Range", f"{len(filtered_data['Date'].unique())} days")
            else:
                st.metric("Date Range", "N/A")
        
        with summary_col4:
            if selected_date != "All Dates":
                today_count = len(filtered_data)
                st.metric("Today's Count", today_count)
    else:
        st.info("📝 No attendance records found for the selected filters.")
else:
    st.info("📝 No attendance records available. Start the camera to begin tracking!")

# --- System Status and Information ---
st.markdown("---")
st.subheader("💡 System Information")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown("""
    **🎯 Features:**
    - Real-time face detection and recognition
    - Automatic attendance logging
    - Duplicate prevention (one entry per day)
    - Configurable recognition threshold
    - Export to CSV/Excel
    - Session statistics tracking
    """)

with info_col2:
    st.markdown("""
    **📋 Instructions:**
    1. Ensure your camera is connected
    2. Adjust recognition threshold if needed
    3. Click 'Start Camera' to begin
    4. Look directly at the camera for best results
    5. Attendance is automatically saved
    6. Use filters to view specific records
    """)

# --- Error Recovery Section ---
with st.expander("🔧 Troubleshooting & Manual Controls"):
    st.markdown("""
    **Common Issues:**
    - **Camera not working**: Check camera permissions and try different camera index
    - **Low recognition accuracy**: Increase threshold or ensure good lighting
    - **File errors**: Check file permissions and paths
    """)
    
    # Manual attendance entry
    st.subheader("✍️ Manual Attendance Entry")
    manual_col1, manual_col2 = st.columns(2)
    
    with manual_col1:
        manual_name = st.text_input("Name")
    
    with manual_col2:
        if st.button("➕ Add Manual Entry"):
            if manual_name.strip():
                success, message = save_attendance(manual_name.strip(), st.session_state.attendance_data)
                if success:
                    st.session_state.attendance_data = load_attendance_data()
                    st.success(f"✅ {message}")
                else:
                    st.warning(f"⚠️ {message}")
                st.rerun()
            else:
                st.error("Please enter a valid name")
    
    # System reset
    if st.button("🔄 Reset System", type="secondary"):
        st.session_state.camera_running = False
        st.session_state.recognition_stats = {
            'total_recognitions': 0,
            'unique_people': 0,
            'session_start': None
        }
        st.success("System reset successfully!")
        st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🤖 Powered by AI • Built with Streamlit • Face Recognition Technology</p>
    <p><small>Ensure proper lighting and face the camera directly for best results</small></p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for live updates (when camera is running)
if st.session_state.camera_running:
    time.sleep(0.1)  # Small delay to prevent excessive CPU usage