import streamlit as st
import pickle
import numpy as np
from PIL import Image
import face_recognition
import firebase_admin
from firebase_admin import credentials, db
from keras.models import load_model
import cv2 

# Page configuration
st.set_page_config(
    page_title="CineFace: Actor Recognition and Emotion Detection System",
    page_icon="🎭",
    layout="centered"  # Changed to wide for a more spacious layout
)

# Firebase initialization (uncomment if needed)
"""cred = credentials.Certificate("actor-and-actress-firebase-adminsdk-8skz5-0a638a7bf7.json")
firebase_admin.initialize_app(cred, {
'databaseURL': 'https://actor-and-actress-default-rtdb.firebaseio.com/'
 })"""

@st.cache_data
def load_encoded_faces(file_path):
    """Load and return encoded face data from a pickle file."""
    try:
        with open(file_path, "rb") as file:
            encode_list_known_with_names = pickle.load(file)
            return encode_list_known_with_names
    except FileNotFoundError:
        st.error("Encoding file not found. Please check the file path.")
        st.stop()

def fetch_actor_info(name):
    """Retrieve actor information from Firebase based on the name."""
    ref = db.reference('face_encodings')
    actors = ref.get()
    return next((actor['actor_info'] for actor in actors.values() if actor['name'] == name), None)

def display_actor_info(actor_info):
    """Display the actor's biography, birthday, and place of birth in a table format."""
    if actor_info:
        st.markdown("### Actor Information")
        col1, col2 = st.columns(2)
        col3, col4=st.columns(2)
        col5,col6=st.columns(2)
        col1.write("**Biography:**")
        col2.write(actor_info.get('biography', 'N/A'))
        
        col3.write("**Birthday:**")
        col4.write(actor_info.get('birthday', 'N/A'))
        
        col5.write("**Place of Birth:**")
        col6.write(actor_info.get('place_of_birth', 'N/A'))
    else:
        st.warning("No additional information found for this actor in Firebase.")
        
emotion_model_path = 'emotion_model.h5'
emotion_model = load_model(emotion_model_path)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion(face):
    """Detect emotion from a face image."""
    face_resized = cv2.resize(face, (48, 48))
    face_normalized = face_resized / 255.0
    face_reshaped = np.reshape(face_normalized, (1, 48, 48, 3))  # RGB format
    prediction = emotion_model.predict(face_reshaped)
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion

# Application title
st.title("🎭 CineFace")
st.markdown("#### Actor Recognition , Emotion Detection and Information  System")
st.markdown("---")

# Upload image section
st.markdown("### Upload an Image")
uploaded_image = st.file_uploader("", type=["jpg", "png", "jpeg"])

# Load face encodings
st.info("Loading encoded face data...")
encode_list_known, names = load_encoded_faces("EncodeFile.p")
st.success("Encoding data loaded successfully.")
st.markdown("---")

# Process uploaded image
if uploaded_image is not None:
    st.markdown("### Analyzing Image")
    with st.spinner("Processing..."):
        try:
            img = Image.open(uploaded_image)
            img_array = np.array(img)

            face_locations = face_recognition.face_locations(img_array)
            face_encodings = face_recognition.face_encodings(img_array)
            
            if face_encodings:
                encode_face = face_encodings[0]
                face_distances = face_recognition.face_distance(encode_list_known, encode_face)
                match_index = np.argmin(face_distances)

                col1, col2 = st.columns([2, 3])
                col1.image(img, caption="Uploaded Image", use_column_width=True)
                
                top, right, bottom, left = face_locations[0]
                face_img = img_array[top:bottom, left:right]
                emotion = detect_emotion(face_img)

                if face_distances[match_index] <= 0.65:
                    actor_name = names[match_index]
                    actor_info = fetch_actor_info(actor_name)

                    col2.subheader(f"🎬 Match Found: {actor_name}")
                    col2.write(f"**Confidence:** {round((1 - face_distances[match_index]) * 100, 2)}%")
                    col2.write(f"**Detected Emotion:** {emotion}")

                    display_actor_info(actor_info)
                else:
                    col2.error("No match found.")
                    col2.write(f"**Closest match distance:** {round(face_distances[match_index], 2)}")
                    st.info("Try uploading a different image or increasing the encoding data.")
            else:
                st.warning("No face detected in the uploaded image.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload an image to start the analysis.")


    


