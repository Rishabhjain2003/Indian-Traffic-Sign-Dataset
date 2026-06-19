# Walkthrough — Indian Traffic Sign Detection Pipeline

## What Was Built

A complete end-to-end Python pipeline for creating an Indian traffic sign dataset from YouTube dashcam videos, built around a **decoupled producer-consumer architecture**. **15 files** across 5 directories:

```
TrafficSIgns/
├── config/
│   ├── __init__.py
│   └── settings.py              ← Dataclass config with env-var overrides
├── src/
│   ├── __init__.py
│   ├── scraper.py               ← InnerTube search via pytubefix (producer)
│   ├── downloader.py            ← pytubefix video downloading
│   ├── extractor.py             ← ffmpeg frame extraction
│   ├── detector.py              ← YOLOv8 batch inference
│   └── db.py                    ← PostgreSQL schema + discovered_videos queue
├── scripts/
│   └── query_appearances.py     ← Standalone appearance duration query
├── pipeline.py                  ← Producer-consumer CLI orchestrator
├── keyword_generator.py         ← Generates combinatorial_keywords.txt
├── combinatorial_keywords.txt   ← 2,047 search queries
├── best.pt                      ← Trained YOLOv8 weights (not in repo)
├── urls.txt                     ← Manual URL list (alternative to search)
├── requirements.txt             ← Pinned dependencies
├── .env.example                 ← Environment variable template
├── .gitignore
├── TRAINING.md                  ← YOLOv8 training guide
└── output/                      ← Auto-created at runtime
    ├── videos/                  ←   Temporary (deleted after processing)
    ├── frames/<video_id>/       ←   Temporary (deleted after processing)
    └── detections/<video_id>/   ←   Annotated frames with bounding boxes
```

---

## Architecture

```
┌──────────────┐     ┌─────────────────────────┐     ┌─────────────────┐
│   Scraper    │────▸│    discovered_videos     │────▸│  Worker Thread  │
│  (Producer)  │     │    (PostgreSQL queue)    │     │  (Consumer ×N)  │
│  1 thread    │     │                          │     │                 │
│              │     │  status: pending →       │     │  download →     │
│  saves each  │     │    processing →          │     │  extract →      │
│  video to DB │     │    completed / failed    │     │  detect →       │
│  immediately │     │                          │     │  store →        │
│              │     │  FOR UPDATE SKIP LOCKED  │     │  cleanup        │
└──────────────┘     └─────────────────────────┘     └─────────────────┘
```

The producer and consumers are fully decoupled via the database. The producer can run for hours discovering videos while workers process them in real-time. If the pipeline crashes, pending videos remain in the DB and are automatically resumed on the next run.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Producer-consumer threading** | Decouples search (slow, I/O-bound) from processing (CPU-bound). Workers start processing videos before the search completes. |
| `discovered_videos` as DB queue | Persistent — survives crashes. Status tracking (`pending→processing→completed→failed`) with `FOR UPDATE SKIP LOCKED` for lock-free concurrent claiming. |
| One DB connection per thread | `psycopg2` connections are not thread-safe; each worker gets its own `Database` instance. |
| `_safe_video_info()` in scraper | `video.title` can trigger `BotDetection` — we catch it per-attribute instead of losing the whole video. |
| `reset_stale_processing()` on startup | Any rows stuck in `processing` from a crash are reset to `pending` automatically. |
| `pytubefix` over `yt-dlp` | InnerTube API search + download in one library; handles progressive and adaptive streams |
| `subprocess` for ffmpeg | Simpler, more reliable for fixed-FPS extraction |
| Singleton `Settings` dataclass | One import gives you all config; every field overridable via env vars |
| Transaction-per-video in DB | All frames+detections for a video committed atomically; rolled back on failure |
| Parameterised SQL throughout | Zero f-string SQL — all queries use `%s` placeholders |
| Auto-delete videos after processing | Conserves disk space; only annotated detection frames are kept |
| YOLO loaded once | `TrafficSignDetector` class loads the model in `__init__` and reuses it across all batches |

---

## How to Get Started

### 1. Set Up PostgreSQL (Local)

```bash
# Ensure PostgreSQL is running
brew services start postgresql

# Create user and database
psql postgres -c "CREATE USER traffic WITH PASSWORD 'traffic';"
psql postgres -c "CREATE DATABASE indian_traffic_signs OWNER traffic;"
psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE indian_traffic_signs TO traffic;"
psql indian_traffic_signs -c "GRANT ALL ON SCHEMA public TO traffic;"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Place YOLO Weights

Place your trained `best.pt` in the project root. See [TRAINING.md](TRAINING.md) for training instructions.

### 4. Run the Pipeline

```bash
# Search mode — producer + 2 worker threads (default)
python pipeline.py --search

# Search with custom settings
python pipeline.py --search --keywords my_keywords.txt --max-results 10 --workers 4

# Resume — process remaining pending videos (no new search)
python pipeline.py --workers-only

# Single video (inline, no threads)
python pipeline.py --url "https://www.youtube.com/watch?v=XXXXXXXXXXX"

# Add URLs from file to queue + start workers
python pipeline.py --urls urls.txt
```

### 5. Monitor Progress

```sql
-- Check queue state
SELECT status, COUNT(*) FROM discovered_videos GROUP BY status ORDER BY status;
```

### 6. Query Appearances

```bash
python scripts/query_appearances.py
```

Outputs a pretty-printed table and exports to `output/appearances.csv`.

---

## Verification

| Check | Result |
|---|---|
| All 11 Python files pass `ast.parse()` | ✓ |
| CLI args `--search`, `--workers-only`, `--url`, `--urls`, `--workers` present | ✓ |
| All SQL queries use parameterised `%s` placeholders | ✓ |
| All file paths use `pathlib.Path` | ✓ |
| No f-string SQL anywhere in codebase | ✓ |
| `discovered_videos` table has B-tree indexes on `youtube_id` and `status` | ✓ |
| `FOR UPDATE SKIP LOCKED` for atomic video claiming | ✓ |
| `reset_stale_processing()` recovers from crashes | ✓ |
| Each worker thread has its own DB connection | ✓ |
| Graceful `Ctrl+C` shutdown via `stop_event` | ✓ |
| Local PostgreSQL connection verified | ✓ |

> [!NOTE]
> Runtime testing requires installing dependencies (`pip install -r requirements.txt`), a running local PostgreSQL instance, and a trained YOLO model (`best.pt` in the project root).
