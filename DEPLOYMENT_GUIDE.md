# Crypto Filter Shiny - Docker Deployment Guide

## 🚀 Overview
Docker container for the **Crypto Filter Shiny Application**. The container packages both the internal quantitative analytics backend API and the interactive Shiny dashboard into a single, self-contained service.

---

## 🐳 Quick Start with Docker

### Method 1: Docker Compose (Recommended)
Launch the Shiny application:
```bash
docker compose up -d
```

- **Shiny Dashboard URL**: [http://localhost:8000](http://localhost:8000)

To stop the container:
```bash
docker compose down
```

To rebuild after code changes:
```bash
docker compose up --build -d
```

---

### Method 2: Standalone Docker Command

#### 1. Build Image:
```bash
docker build -t crypto-filter-shiny .
```

#### 2. Run Container:
```bash
docker run -d \
  --name crypto-filter-shiny \
  -p 8000:8000 \
  -v $(pwd)/data_cache:/app/data_cache \
  crypto-filter-shiny
```

---

## ⚙️ Architecture & Port Configuration

The container manages two internal components via [`entrypoint.sh`](entrypoint.sh):
1. **Backend API**: Runs internally on `127.0.0.1:3000` (isolated inside the container, no host port collision).
2. **Shiny App**: Listens on `0.0.0.0:8000` (exposed to your host browser).

### Custom Port:
If port `8000` is already in use on your host machine, you can customize it:

- **Via Docker Compose**:
  Create or edit `.env`:
  ```bash
  SHINY_PORT=8080
  ```
  Then run `docker compose up -d`. Access at `http://localhost:8080`.

- **Via Docker CLI**:
  ```bash
  docker run -d -p 8080:8000 -v $(pwd)/data_cache:/app/data_cache crypto-filter-shiny
  ```

---

## 💾 Data Persistence
Historical parquet files cached in [`data_cache/`](data_cache/) are mounted as a volume (`./data_cache:/app/data_cache`), ensuring market data persists across container restarts and rebuilds.
