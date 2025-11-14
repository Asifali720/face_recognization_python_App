from flask import Flask, render_template, Response, flash, request, redirect, url_for, session, flash, send_from_directory, jsonify
from dataclasses import dataclass, asdict
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import asc, or_
from werkzeug.utils import secure_filename
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.exceptions import NotFound
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.exc import IntegrityError
import cv2
import pickle
import numpy as np
import face_recognition
import cvzone
import datetime
from datetime import time as datetime_time
import time
import threading
import os
import csv
import io
import logging
import json
import re
import base64
import random


# Opening all the necessary files needed
with open('config.json') as p:
    params = json.load(p)['params']
encoding_file_path = params['encoding_file_path']
file = open(encoding_file_path, 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds

print(f"✓ Loaded {len(encodeListKnown)} face encodings")
print(f"✓ Student IDs: {studentIds[:5]}..." if len(studentIds) > 5 else f"✓ Student IDs: {studentIds}")

# App configs
app = Flask(__name__)
app.config['SECRET_KEY'] = params['secret_key']
app.config['SQLALCHEMY_DATABASE_URI'] = params['sql_url']
db = SQLAlchemy(app)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = params['upload_folder']
hostedapp = Flask(__name__)
hostedapp.wsgi_app = DispatcherMiddleware(
    NotFound(), {"/Attendance_system": app})
cert_path = params['cert_path']
key_path = params['key_path']
bcrypt = Bcrypt()
migrate = Migrate(app, db)
login_manager = LoginManager()
login_manager.init_app(app)

# Models used to connect in SQL Alchemy
# Model of students data table
class Student_data(db.Model):
    __tablename__ = 'student_data'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    rollno = db.Column(db.String(120), unique=True, nullable=False)
    division = db.Column(db.String(80), nullable=False)
    branch = db.Column(db.String(80), nullable=False)
    regid = db.Column(db.String(80), unique=True, nullable=False)


# Model of Attendance table
class Attendance(db.Model):
    __tablename__ = 'attendance'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    start_time = db.Column(db.String(20))
    end_time = db.Column(db.String(20))
    date = db.Column(db.Date, default=datetime.date.today)
    roll_no = db.Column(db.String(20), nullable=False, unique=False)
    division = db.Column(db.String(10))
    branch = db.Column(db.String(100))
    reg_id = db.Column(db.String(100))


# Model of users table
class Users(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column('psw', db.String(128), nullable=False)
    reg_id = db.Column(db.String(20), nullable=False)
    role = db.Column(db.String(20))
    age = db.Column(db.Integer)
    email = db.Column(db.String(200))

    def __repr__(self):
        return f'<User: {self.username}, Role: {self.role}, Age: {self.age}>'

    def get_id(self):
        return str(self.id)


# New model to store face records and embeddings for faster management
class Face(db.Model):
    __tablename__ = 'faces'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200))
    reg_id = db.Column(db.String(50), index=True)
    email = db.Column(db.String(200))
    image_path = db.Column(db.String(500))
    # Store pickled embedding bytes
    embedding = db.Column(db.LargeBinary)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f'<Face id={self.id} reg_id={self.reg_id} name={self.name}>'


class DetectionLog(db.Model):
    __tablename__ = 'detection_logs'
    id = db.Column(db.Integer, primary_key=True)
    reg_id = db.Column(db.String(50))
    name = db.Column(db.String(200))
    detected_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f'<DetectionLog id={self.id} reg_id={self.reg_id} name={self.name} detected_at={self.detected_at}>'


# random reg id func utility
def generate_unique_reg_id():
    while True:
        reg_id = random.randint(1000, 9999)
        if not Users.query.filter_by(reg_id=str(reg_id)).first():
            return str(reg_id)

def update_last_detection(reg_id, name, age):
    """Update detection log in database within app context"""
    try:
        with app.app_context():
            try:
                # Try to find the most recent detection log
                record = DetectionLog.query.order_by(DetectionLog.detected_at.desc()).first()
                
                if not record:
                    record = DetectionLog()
                else:
                    # Create a new record instead of updating the old one
                    record = DetectionLog()
                
                record.reg_id = reg_id
                record.name = name
                record.detected_at = datetime.datetime.utcnow()
                
                db.session.add(record)
                db.session.commit()
                print(f"Last detection updated in DB: {name} ({reg_id})")
            except Exception as db_error:
                db.session.rollback()
                print(f"Database error updating last detection: {db_error}")
                logging.exception("Failed to update detection log: %s", db_error)
    except Exception as e:
        print(f"Error updating last detection: {e}")
        logging.exception("Error in update_last_detection: %s", e)


# Ensure the users table has an `email` column. If the column is missing, try
# to add it via ALTER TABLE so the registration form can store email addresses.
def ensure_users_email_column():
    try:
        from sqlalchemy import text
        with app.app_context():
            engine = db.get_engine()
            # Check INFORMATION_SCHEMA for the column existence
            query = text("""
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'users'
              AND COLUMN_NAME = 'email'
            """)
            try:
                with engine.connect() as conn:
                    res = conn.execute(query).scalar()
            except Exception:
                # older SQLAlchemy engines may not support scalar(); try fetchone
                with engine.connect() as conn:
                    r = conn.execute(query).fetchone()
                    res = r[0] if r else 0
            if res == 0:
                try:
                    from sqlalchemy import text as _text
                    with engine.begin() as conn:
                        conn.execute(_text("ALTER TABLE users ADD COLUMN email VARCHAR(200) DEFAULT NULL"))
                    logging.info('Added email column to users table')
                except Exception as e:
                    logging.exception('Failed to add email column: %s', e)
    except Exception as e:
        logging.exception('Could not verify/add users.email column: %s', e)

# Do not attempt schema changes before the Flask app and DB are initialized.
# The function will be called after app/db creation below.

# After DB and app are initialized, try to ensure the users.email column exists.
try:
    ensure_users_email_column()
except Exception:
    logging.exception('ensure_users_email_column call failed at startup')

# Ensure new tables exist (Face etc.) -- safe for first-run. Use create_all
# so we don't depend on alembic/migrations for this quick setup step.
try:
    with app.app_context():
        db.create_all()
except Exception:
    logging.exception('db.create_all() failed at startup')


# Ensure a default admin user exists so the project is usable out-of-the-box.
def ensure_default_admin():
    try:
        with app.app_context():
            admin_email = 'admin@gmail.com'
            admin_pass = 'test@12345'
            # Check for an existing admin by username, email or reserved reg_id
            existing = Users.query.filter(
                or_(Users.username == admin_email, Users.email == admin_email, Users.reg_id == 'admin')
            ).first()
            if not existing:
                hashed = bcrypt.generate_password_hash(admin_pass).decode('utf-8')
                # reg_id for admin can be a reserved value
                admin_user = Users(username=admin_email, reg_id='admin', password=hashed, role='admin', age=None, email=admin_email)
                db.session.add(admin_user)
                try:
                    db.session.commit()
                    logging.info('Default admin created: %s', admin_email)
                except Exception:
                    db.session.rollback()
                    logging.exception('Failed to commit default admin')
    except Exception:
        logging.exception('ensure_default_admin failed')


# Create default admin now
try:
    ensure_default_admin()
except Exception:
    logging.exception('ensure_default_admin call failed at startup')


# Variables defined
camera = None  # Global variable to store camera object
# Inline model for detection results. Kept small and local to this file so
# templates / endpoints can easily consume a consistent shape.
@dataclass
class Detection:
    name: str | None = None
    reg_id: str | None = None
    age: str | None = None
    timestamp: str | None = None

# Last detection (populated by gen_frames when a face is recognized)
# Use the inline dataclass model so callers can easily serialize to JSON.
m_last_detection_marker = True
last_detection = Detection()

# Simple in-memory cache to avoid spamming the DB with repeated logs for the
# same identity. Keys are reg_id or display_name; values are datetime of last
# seen during a stream. This cache is used together with the per-stream
# buffer below. It is process-local and resets on restart.
recent_detection_cache: dict = {}

# Per-stream buffer: while the camera is running we collect recognition
# events here and only persist them to the DB when the stream is stopped.
# This matches the requested behaviour: detections are saved when the stream
# is stopped, not continuously while the stream runs.
current_stream_detections: list = []
# Lock to protect access to current_stream_detections from the video thread
detection_list_lock = threading.Lock()
morn_time = datetime_time(int(params['morning_time']))
even_time = datetime_time(int(params['evening_time']))
curr_time = datetime.datetime.now().time()
# Lock to protect writes to the encodings file (in-process only). This avoids
# concurrent-thread corruption when multiple requests try to append encodings.
encode_file_lock = threading.Lock()


# Logic to find what function to call based on the time of day for marking the attendance
if morn_time <= curr_time < even_time:
    morn_attendance = True
    even_attendance = False
else:
    even_attendance = True
    morn_attendance = False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@login_manager.user_loader
def load_user(user_id):
    return db.session.query(Users).get(int(user_id))


# Function to start the camera
def start_camera():
    global camera


# Function to stop the camera
def stop_camera():
    global camera, last_detection
    if camera is not None:
        camera.release()
        camera = None

        try:
            # Flush buffered detections with proper transaction handling
            with detection_list_lock:
                if current_stream_detections:
                    # Create a copy to avoid holding lock during DB operations
                    detections_to_save = current_stream_detections.copy()
                    current_stream_detections.clear()
                    recent_detection_cache.clear()
                else:
                    detections_to_save = []

            # Process detections outside the lock
            if detections_to_save:
                save_detections_safely(detections_to_save)

            # Save last detection if valid
            if (
                last_detection.name 
                and last_detection.name != 'Unknown'
                and last_detection.reg_id 
                and last_detection.reg_id != 'N/A'
            ):
                print(f"Saved last detection to DB: {last_detection.name} ({last_detection.reg_id})")
            else:
                print(f"Skipped saving detection - invalid: name={last_detection.name}, reg_id={last_detection.reg_id}")

        except Exception as e:
            logging.exception('Failed to flush detection buffer on stop_camera: %s', e)

def save_detections_safely(detections):
    """Safely save detections to database with proper error handling"""
    if not detections:
        return
    
    # Deduplicate by latest timestamp
    dedup = {}
    for ev in detections:
        key = ev.get('reg_id') or ev.get('name')
        dt = ev.get('detected_at') or datetime.datetime.utcnow()

        if key in dedup:
            if dt > dedup[key]['detected_at']:
                dedup[key] = {
                    'reg_id': ev.get('reg_id'),
                    'name': ev.get('name'),
                    'detected_at': dt
                }
        else:
            dedup[key] = {
                'reg_id': ev.get('reg_id'),
                'name': ev.get('name'),
                'detected_at': dt
            }

    # Save each detection with individual transaction
    successful_saves = 0
    for key, info in dedup.items():
        try:
            # Use individual transactions for each detection
            with app.app_context():
                log = DetectionLog(
                    reg_id=info['reg_id'],
                    name=info['name'],
                    detected_at=info['detected_at']
                )
                db.session.add(log)
                db.session.commit()
                successful_saves += 1
                print(f"Saved detection log: {info['name']} ({info['reg_id']})")
                
        except Exception as e:
            # Rollback and log error, but continue with other detections
            db.session.rollback()
            logging.error(f"Failed to save detection log for {info['name']}: {str(e)}")
            continue

    print(f"Successfully saved {successful_saves}/{len(dedup)} detection logs")

# Alternative: Batch save with single transaction (faster but all-or-nothing)
def save_detections_batch(detections):
    """Save all detections in a single transaction"""
    if not detections:
        return
    
    dedup = {}
    for ev in detections:
        key = ev.get('reg_id') or ev.get('name')
        dt = ev.get('detected_at') or datetime.datetime.utcnow()

        if key in dedup:
            if dt > dedup[key]['detected_at']:
                dedup[key] = {
                    'reg_id': ev.get('reg_id'),
                    'name': ev.get('name'),
                    'detected_at': dt
                }
        else:
            dedup[key] = {
                'reg_id': ev.get('reg_id'),
                'name': ev.get('name'),
                'detected_at': dt
            }

    try:
        with app.app_context():
            for key, info in dedup.items():
                log = DetectionLog(
                    reg_id=info['reg_id'],
                    name=info['name'],
                    detected_at=info['detected_at']
                )
                db.session.add(log)
            
            db.session.commit()
            print(f"Successfully saved {len(dedup)} detection logs in batch")
            
    except Exception as e:
        db.session.rollback()
        logging.error(f"Batch save of detection logs failed: {str(e)}")
        # Optionally retry individual saves here
        save_detections_safely(list(dedup.values()))

# Function for comparing incoming face with encoded file
def compare(encodeListKnown, encodeFace):
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    matchIndex = np.argmin(faceDis)
    
    # Debug output
    if len(faceDis) > 0:
        min_distance = faceDis[matchIndex]
        print(f"Face comparison - min_distance: {min_distance:.4f}, match found: {matches[matchIndex] if matches else 'No matches'}")
    
    return matches, faceDis, matchIndex


# Function to get name of student from the index given by comparing function
def get_data(matches, matchIndex, studentIds):
    if matches[matchIndex]:
        student_id = studentIds[matchIndex]  # ID from face recognition
        print(f"✓ Face matched: student_id={student_id}")
        return student_id
    print(f"✗ No face match found. matchIndex={matchIndex}, matches={matches}")
    return None  # Return None if no match found


# Function which writes the morning attendance in the database
def morningattendance(name, current_date, roll_no, div, branch, reg_id):
    time.sleep(3)
    try:
        with app.app_context():
            existing_entry = Attendance.query.filter(
                Attendance.name == name,
                Attendance.date == current_date,
                Attendance.start_time != None
            ).first()

            if existing_entry:
                print("Your Attendance is already recorded before")
            else:
                new_attendance = Attendance(
                    name=name,
                    start_time=datetime.datetime.now().strftime("%H:%M:%S"),
                    date=current_date,
                    roll_no=roll_no,
                    division=div,
                    branch=branch,
                    reg_id=reg_id
                )
                db.session.add(new_attendance)
                db.session.commit()
                print("Start time and student data recorded in the database")
    except Exception as e:
        print("Error:", e)


# Function which writes the evening attendance in the database
def eveningattendance(name, current_date):
    time.sleep(3)
    try:
        with app.app_context():
            existing_entry = Attendance.query.filter(
                Attendance.name == name,
                Attendance.date == current_date,
                Attendance.start_time != None
            ).first()
            recorded_entry = Attendance.query.filter(
                Attendance.name == name,
                Attendance.end_time != None
            ).first()

            if existing_entry and not recorded_entry:
                existing_entry.end_time = datetime.datetime.now().strftime("%H:%M:%S")
                db.session.commit()
                print("End time recorded in the database")
            elif recorded_entry:
                print("End time already recorded!")
            else:
                print("No existing entry found for evening attendance")
    except Exception as e:
        print("Error:", e)


# Function which gets data of identified student from the database
def mysqlconnect(student_id):
    if student_id is None:
        return None

    try:
        with app.app_context():
            
            # First try to find detailed student data
            student_data = Student_data.query.filter_by(regid=student_id).first()
            
            if student_data:
                # If student data is found, extract values
                id = student_data.id
                name = student_data.name
                roll_no = student_data.rollno
                division = student_data.division
                branch = student_data.branch
                # student_data may not have age; try Face table for age
                # Prefer getting age from Users (if registration stored age there),
                # otherwise fall back to Face.age if present. This avoids relying
                # on Face having an age column which may not exist in the DB.
                age_val = None
                try:
                    user_rec = Users.query.filter_by(reg_id=student_id).first()
                    if user_rec and getattr(user_rec, 'age', None) is not None:
                        age_val = user_rec.age
                    else:
                        face_rec = Face.query.filter_by(reg_id=student_id).first()
                        if face_rec and getattr(face_rec, 'age', None) is not None:
                            age_val = face_rec.age
                except Exception:
                    age_val = None
                return id, name, roll_no, division, branch, age_val

            # Fallback: some deployments store registrations in `Users` table
            # (created via the registration flow). If a Users record exists for
            # this reg_id, return the username so the front-end can display it.
            user = Users.query.filter_by(reg_id=student_id).first()
            if user:
                # Users may store age directly. As a fallback for roll number
                # (Attendance.roll_no is NOT NULL) use the user's reg_id so
                # attendance rows can still be recorded even when
                # Student_data is not populated.
                age_val = getattr(user, 'age', None)
                roll_fallback = user.reg_id or f"R{user.id}"
                return None, user.username, roll_fallback, None, None, age_val

            # Not found in either table - create a fallback entry
            return None
    except Exception as e:
        print(f"Error in mysqlconnect: {e}")
        logging.exception("mysqlconnect failed: %s", e)
        return None


# Function which does the face recognition and displaying the video feed
def gen_frames(camera):
    global last_detection
    while camera is not None:
        success, frame = camera.read()
        if not success:
            break
        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
        
        # Clear detection at start of frame if no faces detected
        if len(faceCurFrame) == 0:
            last_detection = Detection()

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches, facedis, matchIndex = compare(encodeListKnown, encodeFace)
            student_id = get_data(matches, matchIndex, studentIds)
            data = mysqlconnect(student_id)
            # print("data", data)
            # print("Face distances:", facedis)
            # print("Matches:", matches)
            # print("Best match index:", matchIndex)

            # mysqlconnect returns (id, name, roll_no, division, branch) or
            # (None, None, None, None, None) when not found. Avoid creating
            # attendance rows with a NULL name which violates DB constraints.
            if data is None:
                # safety: treat as unknown
                student_id = None
                name = None
                roll_no = None
                div = None
                branch = None
                age = None
            else:
                name = data[1]
                roll_no = data[2]
                div = data[3]
                branch = data[4]
                age = data[5]
            reg_id = student_id
            # For display, show a readable label even if name is missing
            display_name = name if name else 'Unknown'
            # update last_detection so front-end can poll for recognized faces
            # Only update if we have a valid match (student_id and name are not None)
            try:
                if student_id and name:
                    # Valid match found - update last_detection in memory
                    last_detection.name = display_name
                    last_detection.reg_id = reg_id if reg_id else 'N/A'
                    last_detection.age = age if age is not None else 'N/A'
                    last_detection.timestamp = datetime.datetime.now().isoformat()
                    try:
                        key = reg_id or display_name
                        now_dt = datetime.datetime.utcnow()
                        # short throttle while streaming to avoid repeated identical
                        # entries within the same session; final dedupe happens on stop.
                        THROTTLE_SECS_STREAM = 5
                        last_seen = recent_detection_cache.get(key)
                        if key and (last_seen is None or (now_dt - last_seen).total_seconds() > THROTTLE_SECS_STREAM):
                            with detection_list_lock:
                                current_stream_detections.append({'reg_id': reg_id if reg_id else None, 'name': display_name, 'detected_at': now_dt})
                                recent_detection_cache[key] = now_dt
                    except Exception:
                        # non-fatal; don't let logging break the video loop
                        pass
                else:
                    # Face detected but not recognized - log this as unrecognized
                    print(f"Face detected but not recognized: {display_name}")
            except Exception:
                pass
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bbox = x1, y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(frame, bbox, rt=0)
            # Draw a styled overlay: name (yellow), reg_id (cyan) and age badge (dark blue with white text)
            cv2.putText(frame, display_name, (bbox[0], bbox[1] - 40), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                        (0, 215, 255), 2, lineType=cv2.LINE_AA)

            reg_text = reg_id if reg_id else 'N/A'
            cv2.putText(frame, reg_text, (bbox[0], bbox[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 0), 2, lineType=cv2.LINE_AA)

            age_text = f"Age: {age}" if (age is not None and age != '') else "Age: N/A"
            (text_w, text_h), baseline = cv2.getTextSize(age_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # background rectangle for age badge
            rect_tl = (bbox[0], bbox[1] + 5)
            rect_br = (bbox[0] + text_w + 12, bbox[1] + 5 + text_h + baseline)
            cv2.rectangle(frame, rect_tl, rect_br, (10, 30, 120), -1)  # dark blue
            # white age text inside badge
            cv2.putText(frame, age_text, (bbox[0] + 6, bbox[1] + 5 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, lineType=cv2.LINE_AA)

            current_date = datetime.datetime.now().date()
            # Only mark attendance if we have a valid student id, a resolved
            # name and a non-empty roll number (Attendance.roll_no is NOT NULL
            # in the schema). This avoids IntegrityError when users are stored
            # in `Users` but not present in `Student_data`.
            if student_id and name and roll_no and morn_attendance:
                t = threading.Thread(target=morningattendance, args=(
                    name, current_date, roll_no, div, branch, reg_id))
                t.start()
            # For evening attendance require a recorded roll_no as well so the
            # evening update targets the correct database row.
            if student_id and name and roll_no and even_attendance:
                t = threading.Thread(
                    target=eveningattendance, args=(name, current_date))
                t.start()
        # print("Camera:", camera.isOpened())
        # print("Faces detected:", len(faceCurFrame))
        # print("Loaded known faces:", len(encodeListKnown))
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Route of video feed to flask webpage on index page
@app.route('/video1')
def video1():
    try:
        camera1 = params['camera_index_1']
        try:
            camera_index = int(camera1)
        except Exception:
            camera_index = camera1
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            print(f"video1: failed to open camera index {camera_index}")
            return "Error connecting to the video stream: camera unavailable"
        return Response(gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print("Error:", e)
        return "Error connecting to the video stream"


@app.route('/video2')
def video2():
    try:
        camera2 = params['camera_index_2']
        try:
            camera_index = int(camera2)
        except Exception:
            camera_index = camera2
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            print(f"video2: failed to open camera index {camera_index}")
            return "Error connecting to the video stream: camera unavailable"
        return Response(gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print("Error:", e)
        return "Error connecting to the video stream"


@app.route('/last_detection')
def last_detection_route():
    try:
        return jsonify(asdict(last_detection))
    except Exception:
        return jsonify(asdict(Detection()))


# Route which displays the attendance of all student for that current day
@app.route('/display_attendance', methods=['GET', 'POST'])
@login_required
def display_attendance():
    if current_user.role == 'student':
        stop_camera()
        current_date = datetime.datetime.now().date()
        try:
            input_date = None
            if request.method == 'POST':
                input_date = request.form['date']
            if input_date is None:
                date = current_date
            else:
                date = input_date
            data = Attendance.query.filter_by(date=date).all()
            return render_template('display_data.html', data=data, date=date)
        except Exception as e:
            # Return a more informative error message or handle specific exceptions
            return str(e)
    else:
        return 'UnAuthorized access'

# Route to add new students page for admins


@app.route('/data')
@login_required
def data():
    if current_user.role == 'admin':
        stop_camera()
        return render_template('data.html')
    else:
        return 'UnAuthorized Access'


@app.route('/add_user', methods=['POST'])
@login_required
def add_user():
    name = request.form['name']
    branch = request.form['branch']
    division = request.form['division']
    regid = request.form['reg_id']
    rollno = request.form['roll_no']

    # Check if a student with the same name already exists
    existing_student = Student_data.query.filter_by(name=name).first()

    if existing_student:
        # Student already exists, handle the error (e.g., display a message)
        error_message = 'Student already exists!'
        flash('Student already exists!', 'error')
        return redirect(url_for('data'))
    else:
        # Check if the post request has the file part
        if 'image' not in request.files:
            error_message = 'No file part'
            flash('No file part')
            return redirect(request.url)

        file = request.files['image']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            error_message = 'No selected file'
            flash('No selected file')
            return redirect(request.url)

        # Check if the file extension is allowed
        if file and allowed_file(file.filename):
            # Secure the filename to prevent any malicious activity
            filename = secure_filename(
                regid + '.' + file.filename.rsplit('.', 1)[1].lower())
            # Save the file to the upload folder
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Proceed to add the new student
            user = Student_data(name=name, rollno=rollno,
                                division=division, branch=branch, regid=regid)
            db.session.add(user)
            db.session.commit()
            error_message = 'Student added successfully!'
            flash('Student added successfully!', 'success')
            return render_template('data.html', error=error_message)
        else:
            error_message = 'Invalid file extension. Allowed extensions are: png, jpg, jpeg, gif'
            flash(
                'Invalid file extension. Allowed extensions are: png, jpg, jpeg, gif', 'error')
            return redirect(request.url)


@app.route('/get_attendance', methods=['GET'])
@login_required
def get_attendance():
    if current_user.role == 'teacher':
        stop_camera()
        query_parameters = {}
        for key, value in request.args.items():
            if value:
                query_parameters[key] = value

        if query_parameters:
            attendance_records = Attendance.query.filter_by(
                **query_parameters).order_by(asc(Attendance.reg_id)).all()

            if not attendance_records:
                flash("No records available for the specified criteria")
        else:
            flash("No parameters provided for query")
            attendance_records = []  # Assign an empty list to avoid undefined variable error

        return render_template('results.html', attendance_records=attendance_records)
    else:
        return 'UnAuthorized access'

# Function to download the attendance of particular date in cvs format


@app.route('/download_attendance_csv', methods=['POST'])
def download_attendance_csv():
    try:
        # Assuming the date is submitted via a form
        date = request.form.get('date')
        if not date:
            flash("Date not provided for downloading.")
            return redirect(url_for('get_attendance'))

        # Retrieve attendance records for the specified date
        attendance_records = Attendance.query.filter_by(date=date).all()

        if not attendance_records:
            flash("No attendance records found for the specified date.")
            return redirect(url_for('get_attendance'))

        # Create a CSV string
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Name', 'Start Time', 'End Time', 'Date',
                        'Roll Number', 'Division', 'Branch', 'Registration ID'])
        for record in attendance_records:
            writer.writerow([record.name, record.start_time, record.end_time, record.date,
                            record.roll_no, record.division, record.branch, record.reg_id])

        # Save CSV file to a specified folder
        folder_path = 'downloads'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = os.path.join(folder_path, f"attendance_records_{date}.csv")
        with open(file_path, 'w') as f:
            # Remove trailing newline characters
            f.write(output.getvalue().strip())

        flash("Attendance records downloaded successfully.")
        error_message = 'Attendance records downloaded successfully.'
        return render_template('results.html', error=error_message)
    except Exception as e:
        logging.exception(
            "Error occurred while generating CSV file: %s", str(e))
        flash("An error occurred while generating CSV file.")
        error_message = 'An error occurred while generating CSV file.'
        return render_template('results.html', error=error_message)


# Route to registration page for viewing the attendance
@app.route('/register', methods=['GET', 'POST'])
def register():
    stop_camera()
    error = None  # Initialize error variable
    # Password validation temporarily disabled for easier testing / onboarding.
    if request.method == 'POST':
        username = request.form['username']
        # reg_id = request.form['reg_id']
        password = request.form['password']
        age = request.form['age']
        role = request.form['role']
        hashed_pass = bcrypt.generate_password_hash(password).decode('utf-8')
        # Check if username or reg_id already exists
        existing_user = Users.query.filter_by(username=username).first()
        # existing_reg_id = Users.query.filter_by(reg_id=reg_id).first()
        main_reg_id = generate_unique_reg_id()
        if existing_user:
            error = 'Username already exists!'
            print('Username already exists!')
        # elif existing_reg_id:
        #     error = 'Registration ID already exists!'
        #     print('Registration ID already exists!')
        # Password strength validation disabled
        else:
            # Verify face: we expect a captured live image (base64) submitted from the page.
            try:
                email = request.form.get('email')
                captured_dataurl = request.form.get('captured_image')
                age = request.form.get('age')
                role = request.form.get('role')

                # Normalize age to integer if possible
                try:
                    age_int = int(age) if age is not None and age != '' else None
                except Exception:
                    age_int = None

                if not email:
                    error = 'Email is required.'
                    return render_template('register.html', error=error)

                if not captured_dataurl:
                    error = 'Please capture a live face for verification using the webcam.'
                    return render_template('register.html', error=error)
                if not age:
                    error = 'Age is required.'
                    return render_template('register.html', error=error)
                if role == "student":
                     new_student = Student_data(name=username, rollno=random.randint(0, 99),
                                                division="N/A", branch="N/A", regid=main_reg_id)
                     db.session.add(new_student)
                # helper to decode dataurl -> bytes
                def dataurl_to_bytes(dataurl):
                    header, encoded = dataurl.split(',', 1)
                    return base64.b64decode(encoded)

                def bytes_to_cv2_image(img_bytes):
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    return img

                def get_encoding_from_bytes(img_bytes):
                    img = bytes_to_cv2_image(img_bytes)
                    if img is None:
                        raise ValueError('Could not decode image')
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Try to detect faces at native resolution
                    locs = face_recognition.face_locations(rgb)

                    # If no faces found, try an upscaled version (helpful for small faces)
                    if not locs:
                        try:
                            h, w = img.shape[:2]
                            scale = 1.5
                            resized = cv2.resize(img, (int(w * scale), int(h * scale)))
                            rgb_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                            locs_resized = face_recognition.face_locations(rgb_resized)
                            if locs_resized:
                                # Map the first detected location back to original image coordinates
                                top, right, bottom, left = locs_resized[0]
                                top = int(top / scale)
                                right = int(right / scale)
                                bottom = int(bottom / scale)
                                left = int(left / scale)
                                locs = [(top, right, bottom, left)]
                        except Exception:
                            # ignore upscale errors and continue
                            pass

                    if not locs:
                        raise ValueError('No face found in the image')

                    # If multiple faces are found, pick the largest (most likely the user)
                    if len(locs) > 1:
                        areas = []
                        for (top, right, bottom, left) in locs:
                            areas.append(((bottom - top) * (right - left), (top, right, bottom, left)))
                        areas.sort(reverse=True)
                        chosen = areas[0][1]
                        locs = [chosen]

                    encs = face_recognition.face_encodings(rgb, locs)
                    if not encs:
                        raise ValueError('No face encoding found')
                    return encs[0], img

                try:
                    captured_bytes = dataurl_to_bytes(captured_dataurl)
                except Exception as e:
                    error = 'Failed to decode captured image.'
                    return render_template('register.html', error=error)

                try:
                    enc_captured, decoded_img = get_encoding_from_bytes(captured_bytes)
                except Exception as e:
                    error = f'Captured image problem: {e}'
                    return render_template('register.html', error=error)

                # Use the captured image as the registered photo; save it to uploads

                reg_id = main_reg_id
                filename_ext = 'jpg'
                safe_name = secure_filename(f"{reg_id}.{filename_ext}")
                upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads')
                os.makedirs(upload_folder, exist_ok=True)
                uploaded_path = os.path.join(upload_folder, safe_name)
                # write bytes to file
                with open(uploaded_path, 'wb') as f:
                    f.write(captured_bytes)

                # Create new user in DB (commit with rollback on failure)
                new_user = Users(username=username, reg_id=reg_id,
                                 password=hashed_pass, role=role, age=age_int, email=email)
                db.session.add(new_user)
                try:
                    db.session.commit()
                except IntegrityError as ie:
                    # Handle UNIQUE constraint violations gracefully.
                    db.session.rollback()
                    msg = str(ie)
                    logging.exception('Integrity error creating user: %s', ie)
                    # Some DB dumps have an odd UNIQUE constraint on the `role` column
                    # (single allowed value). If the conflict is on the role, clear
                    # the role and retry once.
                    if 'Duplicate entry' in msg and 'role' in msg:
                        try:
                            new_user.role = None
                            db.session.add(new_user)
                            db.session.commit()
                        except Exception as e2:
                            db.session.rollback()
                            logging.exception('Retry after role-clear failed: %s', e2)
                            error = 'A user with the same username or registration ID already exists.'
                            return render_template('register.html', error=error)
                    else:
                        error = 'A user with the same username or registration ID already exists.'
                        return render_template('register.html', error=error)
                except Exception as e:
                    db.session.rollback()
                    logging.exception('Database error creating user: %s', e)
                    error = 'Database error while creating the user. Please try again.'
                    return render_template('register.html', error=error)

                # Append the new encoding to the encodings file and in-memory lists.
                # Use an in-process lock to avoid concurrent-thread corruption.
                try:
                    global encodeListKnown, studentIds, encoding_file_path
                    encode_file_lock.acquire()
                    # use the encoding extracted from the captured image
                    encodeListKnown.append(enc_captured)
                    studentIds.append(reg_id)
                    # Write to a temporary file first then replace to be a bit safer
                    tmp_path = f"{encoding_file_path}.tmp"
                    with open(tmp_path, 'wb') as ef:
                        pickle.dump([encodeListKnown, studentIds], ef)
                    os.replace(tmp_path, encoding_file_path)
                except Exception as e:
                    # non-fatal: user created but encoding not saved
                    logging.exception('Failed to append encoding: %s', e)
                finally:
                    try:
                        encode_file_lock.release()
                    except RuntimeError:
                        pass

                # Persist face record in the DB for management/search
                try:
                    # store pickled embedding bytes in Face.embedding
                    emb_bytes = pickle.dumps(enc_captured)
                    face_record = Face(name=username, reg_id=reg_id, email=email,
                                       image_path=uploaded_path, embedding=emb_bytes)
                    db.session.add(face_record)
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    logging.exception('Failed to save Face record to DB: %s', e)

                error = 'Registration successfull!'
                flash('Registration successful!', 'success')
                # Redirect to the index page after successful registration to
                # ensure the camera is started and to avoid re-submitting the
                # POST if the user refreshes the page.
                return redirect(url_for('index'))
            except Exception as e:
                logging.exception('Error during registration verification: %s', e)
                error = 'An unexpected error occurred during registration.'
                return render_template('register.html', error=error)

    # Pass error variable to template
    return render_template('register.html', error=error)


@app.route('/login', methods=['GET', 'POST'])
def login():
    error_message = None
    if request.method == 'GET':
        return render_template('login.html')
    elif request.method == 'POST':
        identifier = (request.form.get('username') or '').strip()
        password = request.form.get('password')

        try:
            # Allow login via username OR email
            user = Users.query.filter(or_(Users.username == identifier, Users.email == identifier)).first()

            # If no user found but the identifier matches our default admin email,
            # and the provided password matches the default admin password, create
            # the default admin on-the-fly and reload the user record.
            default_admin_email = 'admin@gmail.com'
            default_admin_pass = 'test@12345'
            if not user and identifier.lower() == default_admin_email and password == default_admin_pass:
                try:
                    ensure_default_admin()
                except Exception:
                    logging.exception('Could not ensure default admin during login')
                # reload user after creation attempt
                user = Users.query.filter(or_(Users.username == identifier, Users.email == identifier, Users.reg_id == 'admin')).first()

            if user and bcrypt.check_password_hash(user.password, password):
                login_user(user)
                session['user_id'] = user.id
                session['username'] = user.username
                session['role'] = user.role
                error_message = 'Welcome back, {}!'.format(user.username)
                flash(error_message, 'success')
                # Redirect based on the user's role
                if user.role == 'admin':
                    flash(error_message, 'success')
                    return redirect(url_for('admin_dashboard'))
                elif user.role == 'teacher':
                    flash(error_message, 'success')
                    return render_template('results.html', error=error_message)
                elif user.role == 'student':
                    flash(error_message, 'success')
                    return redirect(url_for('student_dashboard'))
            else:
                error_message = 'Incorrect username or password. Please try again.',
                flash('Incorrect username or password. Please try again.', 'error')
        except SQLAlchemyError as e:
            error_message = 'An error occurred while processing your request. Please try again later.'
            flash(
                'An error occurred while processing your request. Please try again later.', 'error')
            # Log the exception for further investigation
            print(e)
    # If the request method is not GET or POST, or if the login process fails for any reason
    return render_template('login.html', error=error_message)


def findEncodings(imageslist):
    encodeList = []
    for img in imageslist:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def rebuild_encodings_from_uploads():
    """Regenerate the encoding file and in-memory lists from images in uploads/.
    This helps keep DB and Resources/EncodeFile.p in sync after deletes.
    """
    global encodeListKnown, studentIds, encoding_file_path
    upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads')
    if not os.path.isdir(upload_folder):
        return
    pathList = [p for p in os.listdir(upload_folder) if os.path.isfile(os.path.join(upload_folder, p))]
    imgList = []
    ids = []
    for path in pathList:
        img = cv2.imread(os.path.join(upload_folder, path))
        if img is None:
            continue
        imgList.append(img)
        ids.append(os.path.splitext(path)[0])
    try:
        if imgList:
            newEncodings = findEncodings(imgList)
            encodeListKnown = newEncodings
            studentIds = ids
            tmp_path = f"{encoding_file_path}.tmp"
            with open(tmp_path, 'wb') as ef:
                pickle.dump([encodeListKnown, studentIds], ef)
            os.replace(tmp_path, encoding_file_path)
    except Exception as e:
        logging.exception('Failed to rebuild encodings from uploads: %s', e)

# Route to trigger encoding manually


@app.route('/generate_encodings', methods=['GET', 'POST'])
def generate_encodings():
    if request.method == 'POST':
        # Delete existing encoding file if it exists
        encoding_file_path = "Resources/EncodeFile.p"
        if os.path.exists(encoding_file_path):
            os.remove(encoding_file_path)
            print("File removed")
            flash("File Removed")

        # Importing the student images
        folderPath = 'uploads'
        pathList = os.listdir(folderPath)
        imgList = []
        studentIds = []
        for path in pathList:
            imgList.append(cv2.imread(os.path.join(folderPath, path)))
            studentIds.append(os.path.splitext(path)[0])
            print(os.path.splitext(path)[0])
        # Generate encodings
        try:
            print("Encoding started...")
            error_message = 'Encoding started...'
            flash("Encoding started...", "success")
            encodeListKnown = findEncodings(imgList)
            encodeListKnownWithIds = [encodeListKnown, studentIds]
            print("Encoding complete")
            error_message = 'Encoding complete'
            flash("Encoding complete", "success")
            with open(encoding_file_path, 'wb') as file:
                pickle.dump(encodeListKnownWithIds, file)
            print("File Saved")
            error_message = 'Encodings generated successfully!'
            flash('Encodings generated successfully!', 'success')
        except Exception as e:
            print("Error:", e)
            flash('Error occurred while generating encodings.', 'error')

        # Redirect to homepage or any other page after encoding
        return redirect(url_for('data'))

    return render_template('data.html', error=error_message)


# Function for logout functionality
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    error_message = 'Logout Successfully!!'
    logout_user()
    session.clear()
    return render_template('login.html', error=error_message)


@app.route('/images')
@login_required
def images():
    if current_user.role == 'admin':
        image_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(
            os.path.join(app.config['UPLOAD_FOLDER'], f))]
        image_no = len(image_files)
        print(f"No of images: {image_no}")
        return render_template('image_gallery.html', image_files=image_files, image_no=image_no)
    else:
        return 'UnAuthourized access'


@app.route('/images/<filename>')
def get_image(filename):
    # Serve a specific image file
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/faces')
@login_required
def faces():
    # admin-only page to list stored faces
    if current_user.role != 'admin':
        return 'UnAuthorized access'
    try:
        face_list = Face.query.order_by(Face.created_at.desc()).all()
    except Exception:
        face_list = []
    try:
        detection_logs = DetectionLog.query.order_by(DetectionLog.detected_at.desc()).limit(50).all()
    except Exception:
        detection_logs = []
    # Faces page removed — redirect to admin dashboard
    flash('Faces page is no longer available.', 'info')
    return redirect(url_for('admin_dashboard'))


@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    # admin-only page to manage users, teachers, students
    if current_user.role != 'admin':
        return 'UnAuthorized access'
    try:
        users = Users.query.order_by(Users.username).all()
    except Exception:
        users = []
    try:
        students = Student_data.query.order_by(Student_data.name).all()
    except Exception:
        students = []
    try:
        faces = Face.query.order_by(Face.created_at.desc()).all()
    except Exception:
        faces = []
    return render_template('admin_dashboard.html', users=users, students=students, faces=faces)


@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
def admin_delete_user(user_id):
    if current_user.role != 'admin':
        return 'UnAuthorized access'
    try:
        user = Users.query.get(user_id)
        if not user:
            flash('User not found', 'warning')
            return redirect(url_for('admin_dashboard'))
        # Prevent deleting self
        if user.id == current_user.id:
            flash('You cannot delete your own admin account while logged in.', 'warning')
            return redirect(url_for('admin_dashboard'))
        db.session.delete(user)
        db.session.commit()
        flash('User deleted', 'success')
    except Exception as e:
        db.session.rollback()
        logging.exception('Error deleting user: %s', e)
        flash('Error deleting user', 'error')
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/user/<int:user_id>/set_role', methods=['POST'])
@login_required
def admin_set_role(user_id):
    if current_user.role != 'admin':
        return 'UnAuthorized access'
    new_role = request.form.get('role')
    if new_role not in (None, 'admin', 'teacher', 'student'):
        flash('Invalid role', 'error')
        return redirect(url_for('admin_dashboard'))
    try:
        user = Users.query.get(user_id)
        if not user:
            flash('User not found', 'warning')
            return redirect(url_for('admin_dashboard'))
        previous_role = user.role
        user.role = new_role
        db.session.add(user)
        db.session.commit()

        # If role was changed to 'student', ensure the user appears in the
        # `student_data` table so admins can manage student-specific metadata.
        # We reuse the user's reg_id as the roll number when a roll number is
        # not available. Use safe defaults for division/branch to satisfy the
        # Student_data NOT NULL constraints.
        if new_role == 'student':
            try:
                # only create if no existing student_data for this regid
                existing = Student_data.query.filter_by(regid=user.reg_id).first()
                if not existing:
                    # avoid name/roll collisions; if name exists, append suffix
                    base_name = user.username or f'user_{user.id}'
                    name_to_use = base_name
                    counter = 1
                    while Student_data.query.filter_by(name=name_to_use).first():
                        name_to_use = f"{base_name}_{counter}"
                        counter += 1

                    # Use reg_id as rollno to satisfy NOT NULL unique constraint
                    rollno_to_use = user.reg_id or f"R{user.id}"
                    # safe defaults for division/branch
                    division = 'N/A'
                    branch = 'N/A'
                    new_student = Student_data(name=name_to_use, rollno=rollno_to_use,
                                               division=division, branch=branch, regid=user.reg_id)
                    db.session.add(new_student)
                    db.session.commit()
                    flash('Role updated and student record created', 'success')
                else:
                    flash('Role updated', 'success')
            except Exception as e:
                db.session.rollback()
                logging.exception('Error creating Student_data for user: %s', e)
                flash('Role updated but failed to create student record', 'warning')
        else:
            flash('Role updated', 'success')
    except Exception as e:
        db.session.rollback()
        logging.exception('Error updating role: %s', e)
        flash('Error updating role', 'error')
    return redirect(url_for('admin_dashboard'))


@app.route('/create_default_admin', methods=['POST', 'GET'])
def create_default_admin_route():
    """Create the default admin account on demand.

    This endpoint is intentionally permissive only in development mode (DEV=1).
    In production it requires the current user to be an admin.
    """
    # Allow if DEV=1 or the current user is an admin
    if os.environ.get('DEV') != '1':
        if not current_user.is_authenticated or current_user.role != 'admin':
            return 'Unauthorized', 401
    try:
        ensure_default_admin()
        return 'Default admin ensured', 200
    except Exception as e:
        logging.exception('create_default_admin_route failed: %s', e)
        return 'Failed', 500


@app.route('/faces/<int:face_id>/delete', methods=['POST'])
@login_required
def delete_face(face_id):
    if current_user.role != 'admin':
        return 'UnAuthorized access'
    try:
        face = Face.query.get(face_id)
        if not face:
            flash('Face record not found', 'warning')
            return redirect(url_for('admin_dashboard'))
        # delete image file if present
        try:
            if face.image_path and os.path.exists(face.image_path):
                os.remove(face.image_path)
        except Exception:
            logging.exception('Failed to remove face image file')
        db.session.delete(face)
        db.session.commit()
        # rebuild encodings from uploads to keep EncodeFile.p consistent
        try:
            rebuild_encodings_from_uploads()
        except Exception:
            logging.exception('Failed to rebuild encodings after delete')
        flash('Face record deleted', 'success')
    except Exception as e:
        db.session.rollback()
        logging.exception('Error deleting face: %s', e)
        flash('Error deleting face', 'error')
    return redirect(url_for('admin_dashboard'))


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if current_user.role == 'student':
        name = session['username']
        data = Attendance.query.filter_by(name=name).all()
        no_of_attendance = len(data)
        return render_template('profile.html', data=data, no_of_attendance=no_of_attendance)


# Student dashboard: shows personal attendance and allows the student to attempt
# attendance by verifying their face. The attempt endpoint compares the latest
# recognition (last_detection) to the logged-in student's reg_id and, if it
# matches, records attendance using the existing morning/evening helpers.
@app.route('/student_dashboard')
@login_required
def student_dashboard():
    if current_user.role != 'student':
        return 'UnAuthorized access'
    # stop any global camera feed to avoid conflicts with the dashboard
    stop_camera()
    # fetch attendance records for this student's reg_id (show only their records)
    try:
        attendance_records = Attendance.query.filter_by(reg_id=current_user.reg_id).order_by(Attendance.date.desc()).all()
    except Exception:
        attendance_records = []
    # provide the frontend with the last detection so the page can show the
    # currently recognized face (if any)
    try:
        detection_snapshot = asdict(last_detection)
    except Exception:
        detection_snapshot = None
    return render_template('student_dashboard.html', attendance_records=attendance_records, last_detection=detection_snapshot)


@app.route('/student_attempt_attendance', methods=['POST'])
@login_required
def student_attempt_attendance():
    # Endpoint called when a student clicks "Attempt Attendance" on their dashboard.
    if current_user.role != 'student':
        return ('UnAuthorized access', 403)

    try:
        detected = last_detection
        detected_reg = getattr(detected, 'reg_id', None)
        # normalize common placeholders
        if not detected_reg or detected_reg in ('N/A', ''):
            flash('No face currently recognized. Please open the camera and show your face.', 'warning')
            return redirect(url_for('student_dashboard'))

        # Ensure the detected reg_id matches the logged-in student
        if str(detected_reg) != str(current_user.reg_id):
            flash('The recognized face does not match your account. Please try again.', 'error')
            return redirect(url_for('student_dashboard'))

        # Lookup student details (name, roll_no, division, branch)
        student_info = mysqlconnect(current_user.reg_id)
        if not student_info:
            flash('Could not find your student details. Contact administrator.', 'error')
            return redirect(url_for('student_dashboard'))

        _, name, roll_no, division, branch, _ = student_info
        if not roll_no:
            flash('Your roll number is missing in the system. Contact administrator.', 'error')
            return redirect(url_for('student_dashboard'))

        current_date = datetime.datetime.now().date()

        # Perform the attendance write synchronously so the dashboard shows it immediately.
        with app.app_context():
            if morn_attendance:
                # morning: create new attendance row if not already present
                existing_entry = Attendance.query.filter(
                    Attendance.name == name,
                    Attendance.date == current_date,
                    Attendance.start_time != None
                ).first()
                if existing_entry:
                    flash('Morning attendance already recorded for today.', 'info')
                else:
                    try:
                        new_attendance = Attendance(
                            name=name,
                            start_time=datetime.datetime.now().strftime("%H:%M:%S"),
                            date=current_date,
                            roll_no=roll_no,
                            division=division,
                            branch=branch,
                            reg_id=current_user.reg_id
                        )
                        db.session.add(new_attendance)
                        db.session.commit()
                        flash('Morning attendance recorded successfully.', 'success')
                    except Exception:
                        db.session.rollback()
                        logging.exception('Failed to record morning attendance')
                        flash('Failed to record attendance. Try again or contact admin.', 'error')
            else:
                # evening: update end_time on existing row which has start_time
                existing_entry = Attendance.query.filter(
                    Attendance.name == name,
                    Attendance.date == current_date,
                    Attendance.start_time != None
                ).first()
                start_camera()
                if existing_entry:
                    if existing_entry.end_time:
                        flash('Evening attendance already recorded.', 'info')
                    else:
                        try:
                            existing_entry.end_time = datetime.datetime.now().strftime("%H:%M:%S")
                            db.session.add(existing_entry)
                            db.session.commit()
                            flash('Evening attendance recorded successfully.', 'success')
                        except Exception:
                            db.session.rollback()
                            logging.exception('Failed to record evening attendance')
                            flash('Failed to record attendance. Try again or contact admin.', 'error')
                else:
                    flash('No existing morning attendance found to update for evening.', 'warning')
        start_camera()
        return redirect(url_for('student_dashboard'))
   
    except Exception as e:
        logging.exception('student_attempt_attendance failed: %s', e)
        flash('An internal error occurred while attempting attendance.', 'error')
        return redirect(url_for('student_dashboard'))


# Route to the index page where the camera feed is displayed
@app.route('/')
def index():
    start_camera()
    try:
        # Do not show admin accounts in the main registered-users table
        users = Users.query.filter(Users.role != 'admin').order_by(Users.username).all()
    except Exception:
        users = []
    # gather uploaded filenames to show thumbnails
    uploads = []
    try:
        upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads')
        if os.path.isdir(upload_folder):
            uploads = [f for f in os.listdir(upload_folder) if os.path.isfile(os.path.join(upload_folder, f))]
    except Exception:
        uploads = []
    return render_template('index.html', users=users, uploads=uploads)


# Function to start to the app
if __name__ == '__main__':
    # Development helper:
    # - If you want to run the app mounted at `/Attendance_system` (default behavior), nothing changes.
    # - To run the application at the root (`/`) for local development, set the environment
    #   variable DEV=1 before starting the script. Example:
    #     DEV=1 python3 app.py
    # This keeps backwards compatibility with the original mounting.
    if os.environ.get('DEV') == '1':
        # Run the main Flask app at root for easier local testing
        app.run(debug=True, ssl_context=(cert_path, key_path), host='0.0.0.0')
    else:
        # Default behaviour: mount the app under /Attendance_system
        hostedapp.run(debug=True, ssl_context=(cert_path, key_path), host='0.0.0.0')
