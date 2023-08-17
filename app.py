import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

def process(image):
    out_image = image.copy()

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        results = face_mesh.process(image)

        (image_height, image_width) = image.shape[:2]

        blank = np.zeros((image_height, image_width, 3))
        blank += 255

        if not st.checkbox("show camera image"):
            out_image = blank.copy()
            
        if st.checkbox("draw landmarks"):
            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                   for landmark in face.landmark:               
                        x = int(landmark.x * image_width)
                        y = int(landmark.y * image_height)
                        cv2.circle(out_image, center=(x, y), radius=2, color=(0, 0, 255), thickness=-1)
                        cv2.circle(out_image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)    
               
    return cv2.flip(out_image, 1)
    
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = process(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)
