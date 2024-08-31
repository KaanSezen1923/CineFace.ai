Here's a GitHub README description for your project:

---

# 🎭 CineFace: Actor Recognition , Emotion Detection and Information System

**CelebScan** is an advanced image recognition system designed to identify actors from images and provide detailed information about them. The system leverages face recognition, emotion detection, and a connection to The Movie Database (TMDb) API to retrieve comprehensive actor profiles, including their biography, birthday, place of birth, and known works. The project also utilizes Firebase Realtime Database to store and manage actor data.

## Features

- **Actor Recognition**: Upload an image, and CelebScan will identify the actor(s) in the photo by comparing the face with a pre-encoded database.
- **Emotion Detection**: The system can detect the emotional state of the recognized actor based on their facial expression.
- **Actor Information Retrieval**: Fetch detailed information about recognized actors from Firebase and TMDb, including biography, birthday, place of birth, and notable movies.
- **Real-time Analysis**: Quickly process and analyze uploaded images to provide immediate results.

## Project Structure

- **Face Encoding**: The project encodes faces from a dataset of actor images and stores them in a pickle file and Firebase Realtime Database.
- **Image Upload and Analysis**: Users can upload images via the Streamlit interface, where the system will analyze the image to detect and recognize faces.
- **Firebase Integration**: The project is integrated with Firebase Realtime Database to store face encodings and retrieve actor information.
- **Emotion Detection**: A Keras-based model is used to detect emotions from the detected faces in the uploaded image.

## Technologies Used

- **Streamlit**: For building the user interface and image upload functionality.
- **Face Recognition**: For detecting and recognizing actors in uploaded images.
- **Keras**: For emotion detection using a pre-trained deep learning model.
- **Firebase**: To store face encodings and retrieve actor information.
- **TMDb API**: To fetch additional details about recognized actors.

## How to Use

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/CineFace.ai.git
   ```
2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```
4. **Upload an image** and let the system analyze it to recognize the actor and detect their emotion.

## Future Enhancements

- **Enhanced Actor Database**: Expanding the database with more actors and detailed information.
- **Multi-Face Recognition**: Support for recognizing multiple actors in a single image.
- **Real-time Video Analysis**: Adding functionality to analyze video streams in real-time.

## Contributions

Contributions are welcome! Feel free to open an issue or submit a pull request to improve CelebScan.

---

This README provides an overview of your project, instructions on how to use it, and a glimpse of possible future enhancements.

Results:

![image](https://github.com/user-attachments/assets/e0c910ba-84c9-4f1c-97fb-3fe2a7e43907)

![image](https://github.com/user-attachments/assets/aca54203-fe9d-4027-9489-6020d5a579b3)

![image](https://github.com/user-attachments/assets/a5c340b1-d435-45da-a0aa-7ff0b2409c7c)

![image](https://github.com/user-attachments/assets/7bb1b795-b318-4b4d-8721-7f8b5f35c3c4)

![image](https://github.com/user-attachments/assets/e42cd191-eda6-48e9-b478-3922fbf01be1)



