import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=5,
    min_detection_confidence=0.1
)

def process(image):
    out_image = image.copy()

    results = face_mesh.process(image)

    for face_landmarks in results.multi_face_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image=out_image,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS
        )
    
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
