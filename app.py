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

# CSS
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
    }
    .subtitle {
        text-align: center;
        color: #f0f0f0;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏍️ Helmet Detection System")
st.markdown('<p class="subtitle">🎓 Tugas Besar AI | YOLOv8 + Real-time Webcam</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    detection_mode = st.radio("Mode", ["📷 Image", "🎥 Video", "📹 Webcam"])
    confidence = st.slider("Confidence", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")
    with st.expander("ℹ️ About"):
        st.write("""
        **Helmet Detection AI**

        - Model: YOLOv8 Nano
        - Training: 150 epochs
        - Classes: helmet, no helmet
        - Accuracy: 85%+

        **Features:**
        - ✅ Image detection
        - ✅ Video processing
        - ✅ Webcam real-time

        **Team:**
        - Chairani Nayu Nainggolan
        - Esa Canoe Alvian Karim
        - Dary Ibrahim Akram

        Politeknik Negeri Indramayu
        """)

    st.success("🟢 System Online")

# Load model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

with st.spinner("Loading model..."):
    model = load_model()

# ============================================================
# MODE 1: IMAGE
# ============================================================
if detection_mode == "📷 Image":
    st.markdown("## 📷 Image Detection")

    uploaded = st.file_uploader("Upload Image", type=['jpg','jpeg','png'])

    if uploaded:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Original")
            img = Image.open(uploaded)
            st.image(img, use_container_width=True)

        with st.spinner("🔍 Detecting..."):
            results = model(np.array(img), conf=confidence, verbose=False)
            annotated = Image.fromarray(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))

        with col2:
            st.subheader("🎯 Results")
            st.image(annotated, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Statistics")

        with_h = without_h = 0
        confidences = []

        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = results[0].names[cls]
            confidences.append(conf)

            if class_name == 'helmet':
                with_h += 1
            elif class_name == 'no helmet':
                without_h += 1

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("✅ With Helmet", with_h)
        c2.metric("❌ Without Helmet", without_h)
        c3.metric("📍 Total", with_h + without_h)
        c4.metric("🎯 Avg Conf", f"{int(np.mean(confidences)*100) if confidences else 0}%")

        if (with_h + without_h) > 0:
            compliance = (with_h / (with_h + without_h)) * 100
            st.progress(compliance / 100)

            if compliance >= 80:
                st.success(f"🎉 {compliance:.1f}% compliance")
            elif compliance >= 50:
                st.warning(f"⚠️ {compliance:.1f}% compliance")
            else:
                st.error(f"❌ {compliance:.1f}% compliance")

        buf = io.BytesIO()
        annotated.save(buf, format='PNG')
        st.download_button("💾 Download", buf.getvalue(), "result.png", "image/png")
    else:
        st.info("👆 Upload gambar untuk mulai deteksi")

# ============================================================
# MODE 2: VIDEO (FIX - BISA DI-PLAY!)
# ============================================================
elif detection_mode == "🎥 Video":
    st.markdown("## 🎥 Video Detection")
    uploaded = st.file_uploader("Upload Video", type=['mp4','avi','mov'])
    
    if uploaded:
        # Simpan video upload ke file sementara
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded.read())
        
        if st.button("🚀 Process Video"):
            status = st.empty()
            progress = st.progress(0)
            
            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # File sementara untuk hasil mentah (AVI)
            temp_avi = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_avi, fourcc, fps, (640, 360)) # Resize ke 640x360 agar RAM aman
            
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Resize & Detect
                frame = cv2.resize(frame, (640, 360))
                results = model(frame, conf=confidence, verbose=False)
                annotated_frame = results[0].plot()
                
                out.write(annotated_frame)
                count += 1
                progress.progress(count / total_frames)
                status.text(f"Processing frame {count}/{total_frames}...")
            
            cap.release()
            out.release()
            
            # --- TAHAP KRUSIAL: KONVERSI KE MP4 (H.264) AGAR BISA DI-PLAY ---
            status.text("🔄 Mengonversi format agar bisa diputar di browser...")
            final_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            
            # Jalankan FFMPEG
            cmd = f'ffmpeg -y -i "{temp_avi}" -c:v libx264 -pix_fmt yuv420p "{final_mp4}"'
            subprocess.run(cmd, shell=True)
            
            status.text("✅ Selesai! Menampilkan video...")
            
            # Tampilkan Video
            with open(final_mp4, 'rb') as f:
                video_bytes = f.read()
                st.video(video_bytes)
            
            # Tombol Download
            st.download_button("💾 Download Hasil Video", video_bytes, "hasil_deteksi.mp4", "video/mp4")

# ============================================================
# MODE 3: WEBCAM REAL-TIME
# ============================================================
else:
    st.markdown("## 📹 Webcam Real-time Detection")
    
    # Import khusus di sini agar tidak berat saat start-up
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Jalankan deteksi YOLO (Gunakan variabel model yang sudah di-load di atas)
        results = model(img, conf=confidence, verbose=False)
        annotated_frame = results[0].plot()
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    st.info("Klik 'Start' di bawah untuk mengaktifkan kamera laptop Anda.")

    webrtc_streamer(
        key="helmet-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.write("Catatan: Pastikan browser memberikan izin (allow) akses kamera.")

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
