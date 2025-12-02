# Face Recognition Attendance System Multiple Cameras

This project is a web application built with Flask that utilizes face recognition technology for managing attendance. It allows multiple cameras to stream video over HTTP, enabling real-time attendance tracking. The application supports three user roles: Students, Teachers, and Admins, each with specific functionalities.

## 🚀 Features

### Core Features
- **Face Recognition**: Automatically recognizes students' faces for attendance logging using the `face_recognition` Python library. 📸
- **Multiple Camera Support**: Stream from multiple cameras over HTTP, allowing flexibility in attendance monitoring. 🎥
- **Live Face Capture**: Register users with live webcam face capture during registration. 📷
- **Image Recognition**: Recognize faces from uploaded images on the dashboard. 🖼️
- **MySQL Database**: Efficiently stores and manages user data and attendance records. 🗄️

### Role-Based Access Control
- **Admin Users**:
  - Full system management and dashboard access
  - Manage all users, students, and attendance records
  - Download CSV exports for students, attendance, and faces data
  - Add/remove face data and manage student database
  - View registered users and their profiles
  - Admin dropdown menu with logout option only
  
- **Student Users**:
  - View personal attendance records and history
  - Start/Stop attendance recording
  - View live camera stream
  - Access profile settings with logout
  - User dropdown menu with profile and logout options

- **Teacher Users**:
  - Similar to students with additional attendance monitoring
  - Access to live face recognition dashboard
  - User dropdown menu with profile and logout options

### Dashboard Features
- **Admin Dashboard**: 
  - Users management table with delete functionality
  - Students database with downloadable CSV
  - Attendance records with timestamps and CSV export
  - Face recognition database with image management
  - Color-coded role badges and status indicators
  - Gradient styling and modern glassmorphism design

- **Student Dashboard**:
  - Real-time attendance start/stop recording
  - Attendance status with color-coded badges (Complete, In-Progress, Absent)
  - Personal attendance history
  - Animated interface with smooth transitions

### Modern UI/UX
- **Responsive Design**: Mobile-first approach with Bootstrap 5
- **Modern Aesthetics**: 
  - Gradient backgrounds and text effects
  - Glassmorphism design with backdrop filters
  - Smooth CSS animations and transitions
  - Color-coded status indicators and badges
  - Professional typography and spacing

- **Interactive Elements**:
  - Animated hero sections with floating background elements
  - Hover effects on cards and buttons
  - Smooth dropdown animations
  - Real-time status updates

### File Management
- **CSV Downloads**:
  - Export students database as CSV
  - Export attendance records with timestamps
  - Export face recognition database
  - Admin-only access with proper authentication

- **User Image Uploads**:
  - Store and retrieve user profile images
  - Face thumbnails from registration

### Authentication & Security
- **User Registration**: Create new user accounts with email validation
- **Login System**: Secure login with bcrypt password hashing
- **Session Management**: Flask-Login for secure session handling
- **Role-Based Permissions**: Conditional access based on user roles
- **Admin Icon Display**: Admin users see icon-only menu, non-admins see profile dropdown

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
