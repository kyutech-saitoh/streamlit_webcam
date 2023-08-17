import streamlit as st

import tempfile
import time
from PIL import Image

import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Basic App Scaffolding
st.title('Face Mesh App using Streamlit')


def process(image):
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)
    
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        min_detection_confidence=0.1
    ) as face_mesh:

        results = face_mesh.process(image)
        out_image = image.copy()

        st.write("detected")

        for face_landmarks in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image=out_image,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec
            )
    
    return cv2.flip(image, 1)
    
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        st.write("check1")
        img = process(img)
        st.write("check2")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)
