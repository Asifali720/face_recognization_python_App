# 🪟 Windows Setup Guide - Face Recognition Attendance System

اگر آپ Windows پر یہ پروجیکٹ چلانا چاہتے ہیں تو یہ گائیڈ آپ کو مدد دے گی۔ Linux میں سیٹ اپ آسان ہے لیکن Windows میں `dlib` جیسے packages کی وجہ سے مسائل ہوتے ہیں۔

## ⚠️ Windows پر مسائل

1. **dlib installation failure** - Compiler issues
2. **face_recognition** - dlib پر منحصر ہے
3. **OpenCV اور MySQL drivers** - OS-specific dependencies

---

## ✅ حل 1: Pre-compiled Wheels استعمال کریں (آسان ترین)

### Step 1: Python 3.9 یا 3.10 انسٹال کریں
```bash
# ڈاؤن لوڈ کریں: https://www.python.org/downloads/
# IMPORTANT: "Add Python to PATH" چیک کریں install کرتے وقت
```

### Step 2: Visual C++ Build Tools انسٹال کریں
```bash
# ڈاؤن لوڈ کریں: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Install کریں اور "Desktop development with C++" select کریں
```

### Step 3: Project Directory میں جائیں
```bash
cd C:\path\to\Flask_Face_Recognition--multiple_Cameras
```

### Step 4: Virtual Environment بنائیں
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 5: Pre-compiled dlib انسٹال کریں
```bash
# Python 3.9 کے لیے
pip install dlib-19.24.4-cp39-cp39-win_amd64.whl

# Python 3.10 کے لیے
pip install dlib-19.24.4-cp310-cp310-win_amd64.whl

# Python 3.11 کے لیے
pip install dlib-19.24.4-cp311-cp311-win_amd64.whl
```

**Pre-compiled wheels ڈاؤن لوڈ کریں:**
```
https://github.com/davisking/dlib/releases/
یا
https://pypi.org/project/dlib-binary/
```

### Step 6: باقی Dependencies انسٹال کریں
```bash
pip install face-recognition face-recognition-models opencv-python Flask Flask-SQLAlchemy Flask-Login mysqlclient
```

### Step 7: MySQL Server انسٹال کریں
```
https://dev.mysql.com/downloads/mysql/
یا
MariaDB استعمال کریں: https://mariadb.org/download/
```

### Step 8: Database Configure کریں
```bash
# config.json میں اپنے MySQL credentials ڈالیں
{
  "params": {
    "sql_url": "mysql+pymysql://username:password@localhost/database_name"
  }
}
```

### Step 9: Application چلائیں
```bash
# venv activate ہے تو:
python app.py

# یا
python run.py
```

---

## ✅ حل 2: Anaconda استعمال کریں (بہت اچھا)

### Step 1: Anaconda انسٹال کریں
```
https://www.anaconda.com/download/
```

### Step 2: Anaconda Prompt کھولیں اور جائیں
```bash
cd C:\path\to\Flask_Face_Recognition--multiple_Cameras
```

### Step 3: Conda Environment بنائیں
```bash
conda create -n face_rec python=3.9
conda activate face_rec
```

### Step 4: dlib انسٹال کریں (Conda سے)
```bash
conda install -c conda-forge dlib
```

### Step 5: باقی Dependencies انسٹال کریں
```bash
conda install -c conda-forge opencv flask sqlalchemy
pip install face-recognition face-recognition-models Flask-SQLAlchemy Flask-Login Flask-Bcrypt mysqlclient
```

### Step 6: Application چلائیں
```bash
python app.py
```

---

## ✅ حل 3: Docker استعمال کریں (سب سے بہتر)

اگر packages install نہیں ہو رہے تو Docker استعمال کریں:

### Step 1: Docker انسٹال کریں
```
https://www.docker.com/products/docker-desktop
```

### Step 2: Dockerfile بنائیں
```dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install dlib && \
    pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

### Step 3: Docker بنائیں اور چلائیں
```bash
docker build -t face-recognition .
docker run -p 5000:5000 face-recognition
```

---

## 🔧 Troubleshooting - عام مسائل

### مسئلہ 1: dlib install نہیں ہو رہا
```bash
# حل: Pre-compiled wheel استعمال کریں
pip install dlib-binary
```

### مسئلہ 2: mysqlclient error
```bash
# Python 3.11+ میں PyMySQL استعمال کریں
pip uninstall mysqlclient
pip install PyMySQL

# config.json میں:
"sql_url": "mysql+pymysql://user:password@localhost/dbname"
```

### مسئلہ 3: OpenCV camera نہیں کھل رہا
```bash
# Webcam permission دیں یا
# USB camera استعمال کریں
# requirements.txt میں opencv-python-headless ہٹائیں
```

### مسئلہ 4: Face Recognition models download نہیں ہو رہے
```bash
# Manually ڈاؤن لوڈ کریں:
https://github.com/ageitgey/face_recognition_models/releases/

# پھر Resources folder میں رکھیں
```

---

## 📋 Windows Setup Checklist

- [ ] Python 3.9/3.10 انسٹال کریں
- [ ] Visual C++ Build Tools انسٹال کریں
- [ ] Virtual Environment بنائیں
- [ ] dlib انسٹال کریں (wheel سے)
- [ ] باقی dependencies انسٹال کریں
- [ ] MySQL/MariaDB سیٹ اپ کریں
- [ ] config.json میں database credentials ڈالیں
- [ ] `python app.py` چلائیں

---

## 🚀 Quick Start Command (Windows)

### Method 1: سادہ طریقہ
```batch
# 1. Project میں جائیں
cd C:\path\to\project

# 2. Virtual Environment activate کریں
venv\Scripts\activate

# 3. Application چلائیں
python app.py

# 4. Browser میں کھولیں
# http://localhost:5000
```

### Method 2: DEV Mode کے ساتھ (Linux جیسے - Recommended)

**Linux میں:**
```bash
DEV=1 python app.py
```

**Windows Command Prompt میں:**
```batch
set DEV=1
python app.py
```

**Windows PowerShell میں:**
```powershell
$env:DEV=1
python app.py
```

### Method 3: `.bat` Script بنائیں (سب سے آسان)

`run_dev.bat` بنائیں اور یہ کوڈ ڈالیں:
```batch
@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
set DEV=1
python app.py
pause
```

پھر صرف اس فائل کو double-click کریں۔

### Method 4: `.ps1` PowerShell Script (Modern Windows)

`run_dev.ps1` بنائیں:
```powershell
$env:DEV = "1"
& .\venv\Scripts\Activate.ps1
python app.py
```

پھر PowerShell میں چلائیں:
```powershell
.\run_dev.ps1
```

---

## 🔑 Environment Variables کیا کرتے ہیں؟

`DEV=1` سیٹ کرنے سے:
- ✅ Flask Debug Mode ON ہوتا ہے
- ✅ Auto-reload ہوتا ہے (code change پر server restart)
- ✅ Detailed error messages دکھتے ہیں
- ✅ Development کے لیے بہترین ہے

**Production میں:**
```batch
# DEV نہ سیٹ کریں یا
set DEV=0
python app.py
```

---

## 💡 Tips

- **Python 3.9 بہترین ہے** Windows کے لیے (3.11+ میں dlib issues ہیں)
- **Anaconda استعمال کریں** اگر pip سے مسائل ہوں
- **Docker سب سے محفوظ حل ہے** اگر بہت مسائل ہوں
- **MySQL کی بجائے SQLite** استعمال کر سکتے ہیں testing کے لیے

---

## مزید مدد

اگر مسائل ہوں تو:
1. Error message کو Google میں سرچ کریں
2. Stack Overflow دیکھیں
3. GitHub Issues چیک کریں
4. Docker solution استعمال کریں (سب سے محفوظ)

**Happy Coding! 🎉**
