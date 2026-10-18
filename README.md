# Driver Fatigue & Distraction Safety Monitor 🚗

A real-time **Computer Vision and Machine Learning** system for detecting driver drowsiness, fatigue, microsleep, and distraction using an in-cabin camera.

> **Project Status: In Development (Initial Architecture Setup)**  
> Baseline FastAPI server, configuration, and dataset validation utilities are implemented. Computer vision telemetry pipelines, deep learning model training, and the web frontend are currently in development.

---

## 📌 Project Overview

The system monitors driver alertness and cabin behaviors in real time:
* **Facial Landmark Telemetry (Planned):** Analyzes facial features using MediaPipe Face Mesh. Computes **Eye Aspect Ratio (EAR)** for drowsiness and microsleep, **Mouth Aspect Ratio (MAR)** for yawning, and **head pose** (pitch, yaw, roll via `SolvePnP`) for visual distraction.
* **Driver Distraction Classification (Planned):** Categorizes driver postures into the 10 State Farm distraction categories (`c0`–`c9`).
* **Handheld Phone Detection (Planned):** Identifies mobile phone usage and hand-to-ear proximity.
* **Real-Time Alerts (Planned):** Multi-level visual warnings and audio alerts triggered when dangerous behavior is sustained.

---

## 🎯 Project Goal

To develop a lightweight, real-time driver monitoring system that helps reduce roadway accidents caused by **drowsiness, fatigue, and driver distraction**.

---

## 🛠️ Technology Stack

* **Backend & API:** Python, FastAPI, Uvicorn, WebSockets
* **Computer Vision & Telemetry:** OpenCV, MediaPipe
* **Machine Learning & Data:** NumPy, Pandas, Scikit-learn
* **Frontend Dashboard (Planned):** Dark-themed web dashboard
* **Documentation & Planning:** Microsoft Word (`.docx`), Markdown

---

## 👥 Team & Roles

**Project 119 — Driver Fatigue & Distraction Safety Monitor**  
**Domain:** Computer Vision | Machine Learning | Automotive Safety | Telematics

* **Yash** — *Backend, Frontend & Integration Lead:*  
  FastAPI, WebSocket streaming, dashboard, module integration, live status/metrics, alerts, and end-to-end testing.
* **Divyanshi** — *Computer Vision Lead:*  
  Facial landmarks, EAR, MAR, head-pose features, and optional phone-use detection.
* **Ananya** — *Dataset & Machine Learning Lead:*  
  State Farm dataset validation, c0–c9 class handling, model training/evaluation, metrics, and model-weight export.

---

## 📁 Repository Structure

```text
Mini project/
├── Driver_Fatigue_Distraction_Safety_Monitor_Project_Synopsis.docx  # College project synopsis
├── .gitignore                         # Excludes datasets, model weights, venvs, and cache
├── README.md                          # Primary project documentation
├── README.local.md                    # Preserved local architecture draft
│
├── backend/                           # FastAPI server & vision API
│   ├── main.py                        # Server entry point (GET /, GET /health)
│   └── requirements.txt               # Backend dependencies
│
├── frontend/                          # Dark-themed web dashboard (in development)
│
├── training/                          # Dataset validation and training utilities
│   ├── config.py                      # State Farm dataset settings & c0-c9 definitions
│   └── validate_and_prep.py           # Dataset directory validation tool
│
├── datasets/                          # Local dataset storage (Git-ignored)
└── models/                            # Trained model weights & checkpoints (Git-ignored)
```

---

## 🏷️ State Farm Distracted Driver Classes (c0–c9)

Configured in `training/config.py`:
* **`c0`**: Safe driving
* **`c1`**: Texting - right hand
* **`c2`**: Talking on the phone - right ear
* **`c3`**: Texting - left hand
* **`c4`**: Talking on the phone - left ear
* **`c5`**: Operating the radio
* **`c6`**: Drinking
* **`c7`**: Reaching behind
* **`c8`**: Hair and makeup
* **`c9`**: Talking to passenger

---

## 💾 Datasets & Model Weights Storage

> **Important Policy:** Datasets (raw/processed images, CSVs, archives) and trained model weights (`.h5`, `.pt`, `.onnx`, `.bin`) are stored **locally only** and are strictly excluded from Git via `.gitignore`. Never commit large dataset or binary weight files to the repository.

---

## 🚀 Running the Project

### 1. Backend Server (Current Baseline)
The baseline FastAPI application is located in `backend/main.py` and provides `GET /` (service status) and `GET /health` (health check).

Start the development server:
```powershell
python -m uvicorn backend.main:app --reload
```

*Note: Live video streaming and vision inference endpoints will be added in upcoming phases.*

### 2. State Farm Dataset Validation
To verify local dataset folder structure and sample counts across classes `c0` through `c9`:
```powershell
python training/validate_and_prep.py
```
*(Expects dataset images placed under `datasets/state_farm/train/`)*
