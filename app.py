import cv2
import streamlit as st
import mediapipe as mp
import cv2 as cv
import numpy as np
import tempfile
import time
from PIL import Image

# Basic App Scaffolding
st.title('Face Mesh App using Streamlit')

## Add Sidebar and Main Window style
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

## Create Sidebar
st.sidebar.title('FaceMesh Sidebar')
st.sidebar.subheader('Parameter')

## Define available pages in selection box
app_mode = st.sidebar.selectbox(
    'App Mode',
    ['About','Image','Video']
)

# Resize Images to fit Container
@st.cache()
# Get Image Dimensions
def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    dim = None
    # grab the image size
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image
    # calculate the ratio of the height and construct the
    # dimensions
    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)
    # calculate the ratio of the width and construct the
    # dimensions
    else:
        r = width/float(w)
        dim = width, int(h*r)

    # Resize image
    resized = cv.resize(image,dim,interpolation=inter)

    return resized


drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

st.sidebar.markdown('---')

## Add Sidebar and Window style
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("**Detected Faces**")
kpil_text = st.markdown('0')

max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1)
st.sidebar.markdown('---')

detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0,max_value=1.0,value=0.5)
st.sidebar.markdown('---')

## Output
st.markdown('## Output')
img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))

else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))

st.sidebar.text('Original Image')
st.sidebar.image(image)

face_count=0

## Dashboard
with mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, #Set of unrelated images
    max_num_faces=max_faces,
    min_detection_confidence=detection_confidence
) as face_mesh:

        results = face_mesh.process(image)
        out_image=image.copy()

        #Face Landmark Drawing
        for face_landmarks in results.multi_face_landmarks:
            face_count += 1

            mp.solutions.drawing_utils.draw_landmarks(
                image=out_image,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec
            )

            kpil_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)

        st.subheader('Output Image')
        st.image(out_image, use_column_width=True)
