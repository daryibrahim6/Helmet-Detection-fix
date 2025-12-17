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
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded.read())
        tfile.close()

        st.subheader("📹 Original Video")
        st.video(tfile.name)

        col1, col2 = st.columns([2,1])
        with col1:
            process = st.button("🚀 Process Video", type="primary", use_container_width=True)
        with col2:
            skip = st.selectbox("Speed", ["Fast", "Normal", "Quality"])

        if process:
            progress = st.progress(0)
            status = st.empty()

            # Open video
            cap = cv2.VideoCapture(tfile.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Temporary output (raw)
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='_temp.avi').name

            # Use XVID codec (more compatible)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (w, h))

            skip_frames = 2 if skip == "Fast" else 1 if skip == "Normal" else 0

            count = with_h = without_h = 0

            status.text("🔄 Processing frames...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640, 360))
                count += 1

                # Skip frames for speed
                if skip_frames > 0 and count % (skip_frames + 1) != 0:
                    out.write(frame)
                    progress.progress(min(count/total * 0.8, 0.8))
                    continue

                # Run detection
                results = model(frame, conf=confidence, verbose=False)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

                # Count detections
                for box in results[0].boxes:
                    class_name = results[0].names[int(box.cls[0])]
                    if class_name == 'helmet':
                        with_h += 1
                    elif class_name == 'no helmet':
                        without_h += 1

                progress.progress(min(count/total * 0.8, 0.8))
                status.text(f"Processing frame {count}/{total}")

            cap.release()
            out.release()

            # Convert to H264 (browser-compatible)
            status.text("🔄 Converting video for browser playback...")
            progress.progress(0.85)

            final_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

            try:
                # Use ffmpeg to convert
                subprocess.run([
                    'ffmpeg', '-y', '-i', temp_output,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    final_output
                ], check=True, capture_output=True)

                progress.progress(1.0)
                status.text("✅ Processing complete!")

            except subprocess.CalledProcessError:
                # Fallback: just use the temp output
                st.warning("⚠️ Video conversion skipped (ffmpeg not available)")
                final_output = temp_output
                progress.progress(1.0)
                status.text("✅ Processing complete!")

            st.success("🎉 Video processing completed!")

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🎬 Processed Video")
                st.video(final_output)

            with col2:
                st.markdown("### 📊 Statistics")
                st.metric("✅ With Helmet", with_h)
                st.metric("❌ Without Helmet", without_h)

                total_detections = with_h + without_h
                if total_detections > 0:
                    compliance_rate = (with_h / total_detections) * 100
                    st.metric("📈 Compliance Rate", f"{compliance_rate:.1f}%")
                    st.progress(compliance_rate / 100)

                st.info(f"📹 Total frames: {total}")

            # Download button
            st.markdown("---")
            with open(final_output, 'rb') as f:
                st.download_button(
                    label="💾 Download Processed Video",
                    data=f,
                    file_name="helmet_detection_result.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

            # Cleanup temp files
            try:
                os.unlink(temp_output)
            except:
                pass

    else:
        st.info("👆 Upload video untuk mulai deteksi")
        st.markdown("""
        **Tips:**
        - Video max 2 menit untuk hasil optimal
        - Mode "Fast" untuk video panjang
        - Format: MP4, AVI, MOV
        """)

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
