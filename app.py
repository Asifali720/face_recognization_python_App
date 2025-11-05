from flask import Flask, render_template, Response, flash, request, redirect, url_for, session, flash, send_from_directory, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import asc
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


# Opening all the necessary files needed
with open('config.json') as p:
    params = json.load(p)['params']
encoding_file_path = params['encoding_file_path']
file = open(encoding_file_path, 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds


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


# Variables defined
camera = None  # Global variable to store camera object
# Last detection (populated by gen_frames when a face is recognized)
last_detection = {'name': None, 'reg_id': None, 'timestamp': None}
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
    # Align username length with the SQL dump (varchar(100)). Using a too-small
    # length can truncate/produce unexpected behavior when SQLAlchemy emits
    # SQL, so keep it consistent with the DB schema.
    username = db.Column(db.String(100), nullable=False)
    # The DB uses the column name `psw` (from provided SQL dump). Map the model
    # attribute `password` to the existing `psw` column so existing data works.
    # The SQL dump has `psw` as varchar(128) so use 128 here to match.
    password = db.Column('psw', db.String(128), nullable=False)
    reg_id = db.Column(db.String(20), nullable=False)
    role = db.Column(db.String(20))
    # Email column (may be added to DB at startup if missing)
    email = db.Column(db.String(200))

    def __repr__(self):
        return f'<User: {self.username}, Role: {self.role}>'

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


@login_manager.user_loader
def load_user(user_id):
    return db.session.query(Users).get(int(user_id))


# Function to start the camera
def start_camera():
    global camera


# Function to stop the camera
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None


# Function for comparing incoming face with encoded file
def compare(encodeListKnown, encodeFace):
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    # print("matches", matches)
    # print("faceDis", faceDis)
    matchIndex = np.argmin(faceDis)
    return matches, faceDis, matchIndex


# Function to get name of student from the index given by comparing function
def get_data(matches, matchIndex, studentIds):
    if matches[matchIndex]:
        student_id = studentIds[matchIndex]  # ID from face recognition
        return student_id
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
    # If student_id is None, return None for all values
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
                return id, name, roll_no, division, branch

            # Fallback: some deployments store registrations in `Users` table
            # (created via the registration flow). If a Users record exists for
            # this reg_id, return the username so the front-end can display it.
            user = Users.query.filter_by(reg_id=student_id).first()
            if user:
                return None, user.username, None, None, None

            # Not found in either table
            return None
    except Exception as e:
        print("Error:", e)
        return None


# Function which does the face recognition and displaying the video feed
def gen_frames(camera):
    while camera is not None:
        success, frame = camera.read()
        if not success:
            break
        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches, facedis, matchIndex = compare(encodeListKnown, encodeFace)
            student_id = get_data(matches, matchIndex, studentIds)
            data = mysqlconnect(student_id)
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
            else:
                name = data[1]
                roll_no = data[2]
                div = data[3]
                branch = data[4]
            reg_id = student_id
            # For display, show a readable label even if name is missing
            display_name = name if name else 'Unknown'
            print(display_name)
            # update last_detection so front-end can poll for recognized faces
            try:
                global last_detection
                last_detection = {
                    'name': display_name,
                    'reg_id': reg_id if reg_id else 'N/A',
                    'timestamp': datetime.datetime.now().isoformat()
                }
            except Exception:
                pass
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            bbox = x1, y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(frame, bbox, rt=0)
            cv2.putText(frame, display_name, (bbox[0], bbox[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(imgBackground, reg_id if reg_id else 'N/A',
                        (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            current_date = datetime.datetime.now().date()
            # Only mark attendance if we have a valid student id and a resolved name
            if student_id and name and morn_attendance:
                t = threading.Thread(target=morningattendance, args=(
                    name, current_date, roll_no, div, branch, reg_id))
                t.start()
            if student_id and name and even_attendance:
                t = threading.Thread(
                    target=eveningattendance, args=(name, current_date))
                t.start()

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
        # Return the most recent detection as JSON
        return jsonify(last_detection)
    except Exception:
        return jsonify({'name': None, 'reg_id': None, 'timestamp': None})


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
        reg_id = request.form['reg_id']
        password = request.form['password']
        role = request.form['role']
        hashed_pass = bcrypt.generate_password_hash(password).decode('utf-8')
        # Check if username or reg_id already exists
        existing_user = Users.query.filter_by(username=username).first()
        existing_reg_id = Users.query.filter_by(reg_id=reg_id).first()

        if existing_user:
            error = 'Username already exists!'
            print('Username already exists!')
        elif existing_reg_id:
            error = 'Registration ID already exists!'
            print('Registration ID already exists!')
        # Password strength validation disabled
        else:
            # Verify face: we expect a captured live image (base64) submitted from the page.
            try:
                email = request.form.get('email')
                captured_dataurl = request.form.get('captured_image')

                if not email:
                    error = 'Email is required.'
                    return render_template('register.html', error=error)

                if not captured_dataurl:
                    error = 'Please capture a live face for verification using the webcam.'
                    return render_template('register.html', error=error)

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
                                 password=hashed_pass, role=role, email=email)
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
                return render_template('login.html', error=error)
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
        username = request.form.get('username')
        password = request.form.get('password')

        try:
            user = Users.query.filter(Users.username == username).first()

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
                    return render_template('data.html', error=error_message)
                elif user.role == 'teacher':
                    flash(error_message, 'success')
                    return render_template('results.html', error=error_message)
                elif user.role == 'student':
                    flash(error_message, 'success')
                    return render_template('display_data.html', error=error_message)
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
    return render_template('faces.html', faces=face_list)


@app.route('/faces/<int:face_id>/delete', methods=['POST'])
@login_required
def delete_face(face_id):
    if current_user.role != 'admin':
        return 'UnAuthorized access'
    try:
        face = Face.query.get(face_id)
        if not face:
            flash('Face record not found', 'warning')
            return redirect(url_for('faces'))
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
    return redirect(url_for('faces'))


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if current_user.role == 'student':
        name = session['username']
        data = Attendance.query.filter_by(name=name).all()
        no_of_attendance = len(data)
        return render_template('profile.html', data=data, no_of_attendance=no_of_attendance)


# Route to the index page where the camera feed is displayed
@app.route('/')
def index():
    start_camera()
    try:
        users = Users.query.all()
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
