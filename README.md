<p align="center">
  <img src="assets/banner.png" alt="Indian Traffic Sign Detection Pipeline" width="100%"/>
</p>

<h1 align="center">🚦 Indian Traffic Sign Detection Pipeline</h1>

<p align="center">
  <strong>Automated dataset creation from YouTube dashcam footage using YOLOv8 &amp; PostgreSQL</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/YOLOv8-ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=white" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/PostgreSQL-14+-336791?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL 14+"/>
  <img src="https://img.shields.io/badge/pytubefix-InnerTube_API-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="pytubefix"/>
  <img src="https://img.shields.io/badge/ffmpeg-007808?style=for-the-badge&logo=ffmpeg&logoColor=white" alt="FFmpeg"/>
</p>

---

An end-to-end Python pipeline that **discovers** Indian dashcam videos on YouTube via the InnerTube API, **downloads** them, **extracts frames**, runs **YOLOv8 traffic sign detection**, and stores structured results (bounding boxes, confidence scores, timestamps) in a **PostgreSQL** database. Built for researchers building Indian traffic sign datasets at scale.

## ✨ Key Features

- 🔍 **Automated Video Discovery** — Scrapes YouTube using 2,000+ combinatorial search keywords (city × vehicle × modifier)
- 📥 **Smart Downloading** — Progressive or adaptive streams via pytubefix, with automatic ffmpeg muxing for 1080p
- 🎞️ **Frame Extraction** — Configurable FPS and frame stride to control storage usage
- 🧠 **YOLOv8 Detection** — Batch inference with configurable confidence/IoU thresholds, optional annotated frame output
- 🗃️ **PostgreSQL Storage** — Normalised schema with videos → frames → detections, all parameterised queries
- 🔀 **Producer-Consumer Architecture** — Decoupled scraper (producer) and N worker threads (consumers) process videos in parallel
- 🔁 **Crash-Safe Resumability** — `discovered_videos` table with status tracking; stale `processing` rows auto-reset on restart
- 🧹 **Auto-Cleanup** — Videos and frames are deleted after processing to conserve disk space
- 📊 **Appearance Query** — Standalone script to find continuous appearance time ranges per sign class

---

## 📐 Architecture

```
┌──────────────┐     ┌─────────────────────────┐     ┌─────────────────┐
│   Scraper    │────▸│    discovered_videos     │────▸│  Worker Thread  │
│  (Producer)  │     │    (PostgreSQL queue)    │     │  (Consumer ×N)  │
│  1 thread    │     │                          │     │                 │
│              │     │  status: pending →       │     │  download →     │
│  saves each  │     │    processing →          │     │  extract →      │
│  video to DB │     │    completed / failed    │     │  detect →       │
│  immediately │     │                          │     │  store →        │
│              │     │  indexed on youtube_id   │     │  cleanup        │
│              │     │  indexed on status       │     │                 │
└──────────────┘     └─────────────────────────┘     └─────────────────┘
```

### Pipeline Flow

```
Producer Thread                              Worker Threads (×N)
───────────────                              ───────────────────
Keywords ──→ YouTube Search                  claim_next_video()
    │            │                               │
    │     For each video:                   Download video
    │       save to discovered_videos       Extract frames
    │       (status = 'pending')            YOLOv8 detect
    │       push signal to queue ──────▸    Store in DB
    │            │                           Mark 'completed'
    └── next keyword                         Delete video
                                             └── next video
```

---

## 📁 Project Structure

```
TrafficSIgns/
├── config/
│   ├── __init__.py
│   └── settings.py              # Dataclass config — all env-var overridable
├── src/
│   ├── __init__.py
│   ├── scraper.py               # InnerTube search via pytubefix (producer)
│   ├── downloader.py            # Video download (progressive + adaptive)
│   ├── extractor.py             # Frame extraction via ffmpeg subprocess
│   ├── detector.py              # YOLOv8 batch inference
│   └── db.py                    # PostgreSQL CRUD + discovered_videos queue
├── scripts/
│   └── query_appearances.py     # Standalone appearance duration query
├── pipeline.py                  # Producer-consumer CLI orchestrator
├── keyword_generator.py         # Generates combinatorial_keywords.txt
├── best.pt                      # Trained YOLOv8 weights (not in repo)
├── combinatorial_keywords.txt   # 2,047 search queries
├── urls.txt                     # Manual URL list (alternative to search)
├── requirements.txt
├── .env.example                 # Environment variable template
├── .gitignore
├── TRAINING.md                  # Step-by-step YOLOv8 training guide
└── output/                      # Auto-created at runtime
    ├── videos/                  #   Downloaded videos (cleaned up after use)
    ├── frames/<video_id>/       #   Extracted frames (cleaned up after use)
    └── detections/<video_id>/   #   Annotated frames with bounding boxes
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **FFmpeg** installed and on `PATH` (`brew install ffmpeg` on macOS)
- **PostgreSQL 14+** installed locally (`brew install postgresql` on macOS)
- **Trained YOLOv8 weights** — see [TRAINING.md](TRAINING.md) or place your `best.pt` in the project root

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/TrafficSIgns.git
cd TrafficSIgns

python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set Up PostgreSQL

Ensure PostgreSQL is running locally (`brew services start postgresql` on macOS), then create the database and user:

```bash
psql postgres -c "CREATE USER traffic WITH PASSWORD 'traffic';"
psql postgres -c "CREATE DATABASE indian_traffic_signs OWNER traffic;"
psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE indian_traffic_signs TO traffic;"
psql indian_traffic_signs -c "GRANT ALL ON SCHEMA public TO traffic;"
```

Verify the connection:

```bash
PGPASSWORD=traffic psql -h localhost -U traffic -d indian_traffic_signs -c "SELECT current_database(), current_user;"
```

| Parameter | Value |
|---|---|
| Database | `indian_traffic_signs` |
| User | `traffic` |
| Password | `traffic` |
| Port | `5432` |

### 3. Configure (Optional)

```bash
cp .env.example .env
# Edit .env to override any defaults
```

### 4. Generate Search Keywords

```bash
python keyword_generator.py
# → Saves 2,047 queries to combinatorial_keywords.txt
```

### 5. Run the Pipeline

```bash
# 🔍 Search mode — discover & process videos in parallel
python pipeline.py --search

# 🔍 With custom settings
python pipeline.py --search --keywords my_keywords.txt --max-results 10 --workers 4

# 🔄 Resume — process remaining pending videos (no new search)
python pipeline.py --workers-only

# 🎯 Process a single video (inline, no threads)
python pipeline.py --url "https://www.youtube.com/watch?v=XXXXXXXXXXX"

# 📄 Add URLs from a file to the queue and process
python pipeline.py --urls urls.txt
```

---

## ⚙️ Configuration

All settings are defined in `config/settings.py` as a Python dataclass. Every value can be overridden via environment variables:

| Environment Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://traffic:traffic@localhost:5432/indian_traffic_signs` | PostgreSQL connection string |
| `YOLO_WEIGHTS` | `best.pt` | Path to trained YOLOv8 weights |
| `YOLO_DEVICE` | `cpu` | Inference device (`cpu`, `0` for GPU) |
| `YOLO_CONF_THRESHOLD` | `0.35` | Minimum detection confidence |
| `YOLO_IOU_THRESHOLD` | `0.45` | NMS IoU threshold |
| `YOLO_IMG_SIZE` | `640` | Inference image size |
| `YOLO_BATCH_SIZE` | `16` | Frames per inference batch |
| `FPS` | `30` | Frame extraction rate |
| `FRAME_STRIDE` | `1` | Keep every Nth frame (1 = all) |
| `KEYWORDS_FILE` | `combinatorial_keywords.txt` | Search queries file |
| `MAX_RESULTS_PER_KEYWORD` | `5` | Videos per search query |
| `SEARCH_DELAY` | `1.0` | Seconds between search queries |
| `SAVE_ANNOTATED_FRAMES` | `true` | Save frames with drawn bounding boxes |
| `DELETE_VIDEO_AFTER_PROCESSING` | `true` | Delete video files after processing |
| `NUM_WORKERS` | `2` | Number of consumer threads |

---

## 🗃️ Database Schema

Four tables are auto-created on startup:

```sql
-- Metadata for each YouTube video
videos (id, url, youtube_id, title, uploader, upload_date, duration_sec, created_at)

-- Frames that contain at least one detected traffic sign
detected_frames (id, video_id→videos, frame_number, timestamp_sec, frame_path, created_at)

-- Individual bounding-box detections
detections (id, frame_id→detected_frames, class_id, class_name, confidence,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2)

-- DB-backed work queue with status tracking (indexed on youtube_id and status)
discovered_videos (id, youtube_id, url, title, channel, length_sec, keyword,
                   status, error_message, discovered_at, processed_at)
```

### Entity Relationship

```
videos ──< detected_frames ──< detections
discovered_videos (independent — acts as persistent work queue)
```

### Status Lifecycle

```
pending ──→ processing ──→ completed
                      └──→ failed
```

On startup, any rows stuck in `processing` from a previous crash are automatically reset to `pending`.

---

## 📊 Querying Results

### Queue Status

```sql
-- Check the state of the pipeline queue
SELECT status, COUNT(*) FROM discovered_videos GROUP BY status ORDER BY status;
```

### Appearance Duration Report

Find continuous time ranges where each sign class appears in each video:

```bash
python scripts/query_appearances.py
```

**Output:**

```
------------------------------------------------------------
Class Name                      Video URL                        Start (s)    End (s)   Duration (s)
------------------------------------------------------------
speed_limit_40                  https://youtube.com/watch?v=...      12.50      18.30           5.80
stop                            https://youtube.com/watch?v=...      45.00      47.50           2.50
no_entry                        https://youtube.com/watch?v=...     102.30     105.60           3.30
------------------------------------------------------------
```

Results are also exported to `output/appearances.csv`.

### Direct SQL Queries

```sql
-- Top 10 most frequently detected sign classes
SELECT class_name, COUNT(*) as detections
FROM detections
GROUP BY class_name
ORDER BY detections DESC
LIMIT 10;

-- Videos with the most traffic signs
SELECT v.title, v.youtube_id, COUNT(DISTINCT df.id) as frames_with_signs
FROM videos v
JOIN detected_frames df ON df.video_id = v.id
GROUP BY v.id
ORDER BY frames_with_signs DESC;

-- High-confidence detections only
SELECT d.class_name, d.confidence, df.timestamp_sec, v.url
FROM detections d
JOIN detected_frames df ON d.frame_id = df.id
JOIN videos v ON df.video_id = v.id
WHERE d.confidence > 0.80
ORDER BY d.confidence DESC;

-- Failed videos (for debugging)
SELECT youtube_id, url, error_message, discovered_at
FROM discovered_videos
WHERE status = 'failed'
ORDER BY discovered_at DESC;
```

---

## 🧠 Model Training

The pipeline expects a trained YOLOv8 `.pt` weights file. See **[TRAINING.md](TRAINING.md)** for a complete guide covering:

1. Recommended Indian traffic sign datasets on Kaggle
2. Converting annotations to YOLOv8 format
3. Training commands and hyperparameters
4. Evaluation metrics to aim for
5. Placing the trained `best.pt` in the project root

**Quick start:**

```bash
# Download dataset
kaggle datasets download -d pkdarabi/indian-traffic-sign-dataset

# Train
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640

# Deploy weights
cp runs/detect/train/weights/best.pt ./best.pt
```

---

## 🔍 How Search Works

The `keyword_generator.py` script creates combinatorial search queries by combining:

| Component | Examples | Count |
|---|---|---|
| **Vehicle/Format** | bike ride, dashcam, motovlog, car vlog | 9 |
| **Location** | Mumbai, Delhi, NH48, Yamuna Expressway | 23 |
| **Modifier** | POV, 4K, highway, monsoon, night | 10 |

This produces **2,047** unique queries like:
- `bike ride Mumbai POV`
- `dashcam Delhi highway`
- `motovlog Ladakh 4K`
- `car drive Mumbai-Pune Expressway monsoon`

The scraper (`src/scraper.py`) feeds each keyword to YouTube's InnerTube API via `pytubefix.Search`, collects up to 5 unique videos per keyword (deduplication by video ID), and immediately saves each video to the `discovered_videos` database table. Rate-limiting between queries avoids throttling.

---

## 🛡️ Resumability & Crash Safety

The pipeline is designed for long-running, interruptible operation:

| Mechanism | How it works |
|---|---|
| **`discovered_videos` table** | Acts as a persistent work queue. Status = `pending` → `processing` → `completed` / `failed`. |
| **Atomic claiming** | Workers use `FOR UPDATE SKIP LOCKED` — multiple threads can claim different videos without contention. |
| **Crash recovery** | On startup, any rows stuck in `processing` are automatically reset to `pending`. |
| **`--workers-only` flag** | Re-process pending videos without re-running the search phase. |
| **Per-video error isolation** | If one video fails, the worker catches the error, marks it `failed`, and moves to the next. |
| **Graceful shutdown** | `Ctrl+C` sets a stop event — workers finish their current video and exit cleanly. |

---

## 📂 Output Structure

```
output/
├── videos/                      # Temporary — deleted after processing
│   └── <video_id>.mp4
├── frames/                      # Temporary — deleted after processing
│   └── <video_id>/
│       ├── frame_000001.jpg
│       ├── frame_000002.jpg
│       └── ...
├── detections/                  # Persistent — annotated frames kept
│   └── <video_id>/
│       ├── frame_000042.jpg     # Has bounding boxes drawn on it
│       └── ...
└── appearances.csv              # Generated by query_appearances.py
```

> **Note:** Videos and raw frames are automatically deleted after processing to conserve storage. Only annotated detection frames are preserved. Set `DELETE_VIDEO_AFTER_PROCESSING=false` to keep everything.

---

## 🔧 Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Video discovery | `pytubefix` (InnerTube API) | YouTube search & metadata |
| Video download | `pytubefix` + `ffmpeg` | Stream download & adaptive muxing |
| Frame extraction | `ffmpeg` (subprocess) | Fixed-FPS frame capture |
| Object detection | `ultralytics` (YOLOv8) | Traffic sign detection |
| Database | `PostgreSQL` (local) + `psycopg2` | Structured result storage & work queue |
| Concurrency | `threading` + `queue.Queue` | Producer-consumer parallelism |
| Image processing | `OpenCV` + `Pillow` | Annotated frame rendering |
| Configuration | Python `dataclass` | Env-var overridable settings |

---

## 📝 Logging

All pipeline activity is logged to both **stdout** and **`pipeline.log`**. Log lines include the thread name for debugging concurrent operations:

```
2026-06-19 08:30:00  INFO  pipeline  [MainThread]   Started 2 worker threads
2026-06-19 08:30:00  INFO  pipeline  [Producer]     Started producer thread — searching 2047 keywords
2026-06-19 08:30:08  INFO  src.scraper [Producer]   [1/2047] Searching: bike ride Mumbai vlog
2026-06-19 08:30:15  INFO  src.scraper [Producer]     Found 10 new videos (total unique: 10)
2026-06-19 08:30:16  INFO  pipeline  [Worker-0]     Processing abc123 — Highway Ride Mumbai
2026-06-19 08:31:00  INFO  src.detector [Worker-0]  Detection complete — 42 / 18000 frames
2026-06-19 08:31:01  INFO  pipeline  [Worker-0]     ✓ Finished abc123 — 42 frames with detections
2026-06-19 08:31:02  INFO  pipeline  [MainThread]   Queue status — pending: 5, processing: 1, completed: 3, failed: 0
```

---

## 📜 License

This project is for research and educational purposes. Please respect YouTube's Terms of Service when downloading videos. The trained model weights and datasets may have their own licensing terms — check the respective Kaggle dataset pages.
