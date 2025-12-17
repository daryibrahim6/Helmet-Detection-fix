import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io
import tempfile
import os
import subprocess

# Page config
st.set_page_config(
    page_title="Helmet Detection AI",
    page_icon="🏍️",
    layout="wide"
)

# CSS - Enhanced
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
    }
    h1 {
        color: #ffffff;
        text-align: center;
        font-size: 3em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: fadeIn 1s;
    }
    .subtitle {
        text-align: center;
        color: #f0f0f0;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .metric-card {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 10px;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

st.title("🏍️ Helmet Detection System")
st.markdown('<p class="subtitle">🎓 Tugas Besar AI | YOLOv8 + Real-time Webcam</p>', unsafe_allow_html=True)

# Sidebar - Enhanced
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    detection_mode = st.radio(
        "📍 Detection Mode",
        ["📷 Image", "🎥 Video", "📹 Webcam"],
        help="Pilih mode input"
    )
    
    st.markdown("---")
    
    # Advanced settings
    with st.expander("🎚️ Advanced Settings"):
        confidence = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.5, 0.05,
            help="Semakin tinggi = lebih strict (lebih sedikit false positive)"
        )
        
        iou_threshold = st.slider(
            "IOU Threshold",
            0.0, 1.0, 0.45, 0.05,
            help="Untuk menghilangkan duplikat deteksi"
        )
        
        image_size = st.select_slider(
            "Image Size (inference)",
            options=[320, 416, 640, 800, 1024],
            value=640,
            help="Lebih besar = lebih akurat tapi lebih lambat"
        )
    
    st.markdown("---")
    
    with st.expander("ℹ️ About System"):
        st.write("""
        **Helmet Detection AI**
        
        🎯 **Model:**
        - Algorithm: YOLOv8 Nano
        - Training: 150 epochs
        - Classes: helmet, no helmet
        - Target Accuracy: 85%+
        
        ✨ **Features:**
        - ✅ Image detection
        - ✅ Video processing (H.264)
        - ✅ Webcam real-time
        - ✅ Advanced settings
        - ✅ Download results
        
        👥 **Team:**
        - Chairani Nayu Nainggolan
        - Esa Canoe Alvian Karim
        - Dary Ibrahim Akram
        
        🏫 Politeknik Negeri Indramayu
        📚 Kecerdasan Buatan - Semester 5
        """)
    
    st.markdown("---")
    st.success("🟢 System Online")
    st.info(f"🎯 Confidence: {int(confidence*100)}%")

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

with st.spinner("🔄 Loading AI model..."):
    model = load_model()

if model is None:
    st.error("❌ Cannot load model! Pastikan file 'best.pt' ada di folder yang sama.")
    st.stop()

# ============================================================
# MODE 1: IMAGE DETECTION (IMPROVED)
# ============================================================
if detection_mode == "📷 Image":
    st.markdown("## 📷 Image Detection Mode")
    
    # File uploader
    uploaded = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload gambar pengendara motor (JPG, JPEG, PNG)"
    )
    
    if uploaded:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📸 Original Image")
            img = Image.open(uploaded)
            st.image(img, use_container_width=True)
            st.caption(f"Size: {img.size[0]}x{img.size[1]} pixels")
        
        with st.spinner("🔍 Analyzing image..."):
            # Run detection with custom settings
            results = model(
                np.array(img),
                conf=confidence,
                iou=iou_threshold,
                imgsz=image_size,
                verbose=False
            )
            
            # Get annotated image
            annotated = Image.fromarray(
                cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
            )
        
        with col2:
            st.markdown("### 🎯 Detection Results")
            st.image(annotated, use_container_width=True)
            
            # Show detection count
            total_detections = len(results[0].boxes)
            st.caption(f"Total detections: {total_detections}")
        
        # Statistics section
        st.markdown("---")
        st.markdown("## 📊 Detection Statistics")
        
        with_h = without_h = 0
        compliance = 0.0 
        confidences = []
        detections_list = []
        
        for idx, box in enumerate(results[0].boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = results[0].names[cls]
            bbox = box.xyxy[0].cpu().numpy()
            name_check = class_name.lower()
            
            confidences.append(conf)
            detections_list.append({
                'id': idx + 1,
                'class': class_name,
                'confidence': f"{conf*100:.1f}%",
                'bbox': f"({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])})"
            })
            
            if name_check in ['helmet', 'with helmet']:
                with_h += 1
            elif name_check in ['no helmet', 'without helmet']:
                without_h += 1
        
        # Metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "✅ With Helmet",
                with_h,
                help="Jumlah pengendara memakai helm"
            )
        
        with col2:
            st.metric(
                "❌ Without Helmet",
                without_h,
                delta=f"-{without_h}" if without_h > 0 else "Good!",
                delta_color="inverse",
                help="Jumlah pengendara TIDAK memakai helm"
            )
        
        with col3:
            st.metric(
                "📍 Total Detected",
                with_h + without_h,
                help="Total pengendara terdeteksi"
            )
        
        with col4:
            avg_conf = np.mean(confidences) if confidences else 0
            st.metric(
                "🎯 Avg Confidence",
                f"{int(avg_conf*100)}%",
                help="Rata-rata confidence score"
            )
        
        # Compliance rate
        if (with_h + without_h) > 0:
            compliance = (with_h / (with_h + without_h)) * 100
            
            st.markdown("### 📈 Helmet Compliance Rate")
            st.progress(compliance / 100)
            
            if compliance >= 80:
                st.success(f"🎉 Excellent! {compliance:.1f}% compliance rate")
            elif compliance >= 50:
                st.warning(f"⚠️ Moderate: {compliance:.1f}% compliance rate")
            else:
                st.error(f"❌ Poor: {compliance:.1f}% compliance - Need enforcement!")
        
        # Detailed detections table
        if detections_list:
            st.markdown("---")
            st.markdown("### 🔍 Detailed Detections")
            
            import pandas as pd
            df = pd.DataFrame(detections_list)
            st.dataframe(df, use_container_width=True)
        
        # Download section
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            buf = io.BytesIO()
            annotated.save(buf, format='PNG')
            st.download_button(
                label="💾 Download Result Image",
                data=buf.getvalue(),
                file_name="helmet_detection_result.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col2:
            # Save statistics to text
            stats_text = f"""HELMET DETECTION RESULTS
========================
With Helmet: {with_h}
Without Helmet: {without_h}
Total: {with_h + without_h}
Compliance Rate: {compliance:.1f}%
Avg Confidence: {avg_conf*100:.1f}%

Settings:
- Confidence Threshold: {confidence}
- IOU Threshold: {iou_threshold}
- Image Size: {image_size}
"""
            st.download_button(
                label="📄 Download Statistics",
                data=stats_text,
                file_name="detection_stats.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    else:
        st.info("👆 Upload gambar untuk memulai deteksi")
        st.markdown("""
        **💡 Tips untuk hasil terbaik:**
        - Gunakan gambar dengan resolusi minimal 640x640
        - Pastikan pengendara terlihat jelas
        - Hindari gambar blur atau gelap
        - Multiple riders OK - akan dideteksi semua
        """)

# ============================================================
# MODE 2: VIDEO DETECTION (IMPROVED)
# ============================================================
elif detection_mode == "🎥 Video":
    st.markdown("## 🎥 Video Detection Mode")
    
    uploaded = st.file_uploader(
        "Upload video...",
        type=['mp4', 'avi', 'mov'],
        help="Format: MP4, AVI, MOV (max 2 menit recommended)"
    )
    
    if uploaded:
        # Save to temp
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded.read())
        tfile.close()
        
        st.markdown("### 📹 Original Video")
        st.video(tfile.name)
        
        # Get video info
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        st.caption(f"📊 Duration: {duration:.1f}s | Frames: {total_frames} | FPS: {fps}")
        
        # Processing options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            process_btn = st.button(
                "🚀 Process Video",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            speed_mode = st.selectbox(
                "Speed Mode",
                ["Fast (skip 2)", "Normal (skip 1)", "Quality (all frames)"],
                help="Fast: lebih cepat tapi skip beberapa frame"
            )
        
        with col3:
            output_size = st.selectbox(
                "Output Size",
                ["640x360 (Fast)", "854x480 (Medium)", "1280x720 (HD)"],
                index=1
            )
        
        if process_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Parse settings
            if "Fast" in speed_mode:
                frame_skip = 2
            elif "Normal" in speed_mode:
                frame_skip = 1
            else:
                frame_skip = 0
            
            if "640x360" in output_size:
                out_w, out_h = 640, 360
            elif "854x480" in output_size:
                out_w, out_h = 854, 480
            else:
                out_w, out_h = 1280, 720
            
            # Open video
            cap = cv2.VideoCapture(tfile.name)
            
            # Temp output
            temp_avi = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_avi, fourcc, fps, (out_w, out_h))
            
            # Statistics
            with_h_total = 0
            without_h_total = 0
            frame_count = 0
            
            status_text.text("🔄 Processing frames...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Resize
                frame = cv2.resize(frame, (out_w, out_h))
                
                # Skip frames
                if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                    out.write(frame)
                    progress_bar.progress(min(frame_count / total_frames * 0.8, 0.8))
                    continue
                
                # Detect
                results = model(
                    frame,
                    conf=confidence,
                    iou=iou_threshold,
                    imgsz=image_size,
                    verbose=False
                )
                
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
                
                # Count
                  for box in results[0].boxes:
                    class_name = results[0].names[int(box.cls[0])].lower()
                    if class_name in ['helmet', 'with helmet']:
                        with_h_total += 1
                    elif class_name in ['no helmet', 'without helmet']:
                        without_h_total += 1
                
                progress_bar.progress(min(frame_count / total_frames * 0.8, 0.8))
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            cap.release()
            out.release()
            
            # Convert to H264
            status_text.text("🔄 Converting to browser-compatible format...")
            progress_bar.progress(0.85)
            
            final_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            
            try:
                cmd = [
                    'ffmpeg', '-y', '-i', temp_avi,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    final_mp4
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
            except:
                st.warning("⚠️ FFmpeg conversion skipped")
                final_mp4 = temp_avi
                progress_bar.progress(1.0)
            
            st.success("🎉 Video processing completed!")
            
            # Results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎬 Processed Video")
                with open(final_mp4, 'rb') as f:
                    st.video(f.read())
            
            with col2:
                st.markdown("### 📊 Video Statistics")
                
                st.metric("✅ With Helmet (detections)", with_h_total)
                st.metric("❌ Without Helmet (detections)", without_h_total)
                
                total_det = with_h_total + without_h_total
                if total_det > 0:
                    comp_rate = (with_h_total / total_det) * 100
                    st.metric("📈 Overall Compliance", f"{comp_rate:.1f}%")
                    st.progress(comp_rate / 100)
                
                st.info(f"📹 Processed {frame_count} frames")
            
            # Download
            st.markdown("---")
            with open(final_mp4, 'rb') as f:
                st.download_button(
                    "💾 Download Processed Video",
                    f.read(),
                    "helmet_detection_video.mp4",
                    "video/mp4",
                    use_container_width=True
                )
    
    else:
        st.info("👆 Upload video untuk memulai deteksi")
        st.markdown("""
        **💡 Tips:**
        - Video max 2-3 menit untuk hasil optimal
        - Gunakan "Fast" mode untuk video panjang
        - Output size lebih kecil = processing lebih cepat
        """)

# ============================================================
# MODE 3: WEBCAM (IMPROVED)
# ============================================================
else:
    st.markdown("## 📹 Webcam Real-time Detection")
    
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    
    st.warning("""
    ⚠️ **Requirements:**
    - Browser: Chrome atau Edge (recommended)
    - Permission: Allow camera access
    - Close other apps using webcam
    """)
    
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Webcam callback
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Detect with custom settings
        results = model(
            img,
            conf=confidence,
            iou=iou_threshold,
            verbose=False
        )
        
        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
    
    # Start webcam
    st.info("🎥 Klik 'START' untuk mengaktifkan webcam")
    
    webrtc_ctx = webrtc_streamer(
        key="helmet-webcam-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if webrtc_ctx.state.playing:
        st.success("✅ Webcam active - Real-time detection running!")
    
    st.markdown("---")
    st.info("""
    💡 **Tips Webcam:**
    - Arahkan ke foto/gambar pengendara motor
    - Test dengan teman yang pakai/tidak pakai helm
    - Jarak optimal: 1-3 meter
    - Pastikan pencahayaan cukup
    - Gerakan pelan untuk hasil lebih stabil
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#f0f0f0;padding:20px;background:rgba(0,0,0,0.2);border-radius:10px;'>
    <p style='margin:0;font-size:1.1em;'><strong>🎓 Tugas Besar Kecerdasan Buatan</strong></p>
    <p style='margin:5px 0;'>Kelompok 3 | Sistem Informasi Kota Cerdas</p>
    <p style='margin:5px 0;'>Politeknik Negeri Indramayu | 2025</p>
    <p style='margin:10px 0 0 0;font-size:0.9em;opacity:0.8;'>
        Chairani Nayu Nainggolan • Esa Canoe Alvian Karim • Dary Ibrahim Akram
    </p>
</div>
""", unsafe_allow_html=True)
