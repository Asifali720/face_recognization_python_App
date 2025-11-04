# Face Recognition Attendance System Multiple Cameras

This project is a web application built with Flask that utilizes face recognition technology for managing attendance. It allows multiple cameras to stream video over HTTP, enabling real-time attendance tracking. The application supports three user roles: Students, Teachers, and Admins, each with specific functionalities.

## 🚀 Features

- **Face Recognition**: Automatically recognizes students' faces for attendance logging using the `face_recognition` Python library. 📸
- **Multiple Camera Support**: Stream from multiple cameras over HTTP, allowing flexibility in attendance monitoring. 🎥
- **Role-Based Access**:
  - **Students**: View their own attendance records. 📅
  - **Teachers**: Access attendance data for their classes. 📊
  - **Admins**: Add or remove face data and manage student records. ⚙️
- **MySQL Database**: Efficiently stores and manages user data and attendance records. 🗄️

## 🛠️ Technologies Used

- **Flask**: The web framework for building the application. ⚗️
- **face_recognition**: For facial recognition capabilities. 👤
- **OpenCV**: For video capturing and processing. 🖥️
- **MySQL**: For database management. 🐬
- **HTML/CSS/JavaScript**: For front-end development. 🌐


## 📌 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RaY8118/Flask_Face_Recognition--multiple_Cameras.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Flask_Face_Recognition--multiple_Cameras
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up the MySQL database and configure the connection in the application.
5. Run the Flask application:
   ```bash
   python app.py
   ```
6. Access the web application at `http://127.0.0.1:5000`.


## 🌟 Acknowledgments

- [face_recognition](https://github.com/ageitgey/face_recognition) for the facial recognition capabilities.
- [OpenCV](https://opencv.org/) for image and video processing support.
