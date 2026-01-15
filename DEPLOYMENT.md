# AION Analytics Deployment Guide

Complete guide for deploying AION Analytics to production environments.

## Table of Contents

- [Pre-Deployment Checklist](#pre-deployment-checklist)
- [Systemd Service (Linux)](#systemd-service-linux)
- [Docker Compose](#docker-compose)
- [Kubernetes](#kubernetes)
- [Environment Configuration](#environment-configuration)
- [Monitoring Setup](#monitoring-setup)
- [Backup and Restore](#backup-and-restore)
- [Troubleshooting](#troubleshooting)

## Pre-Deployment Checklist

### 1. System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 200GB SSD for data and logs
- **OS**: Ubuntu 20.04+, Debian 11+, or similar

### 2. Network Requirements

- **Outbound**: Access to Alpaca API, Polygon, Alpha Vantage
- **Inbound**: Ports 8000, 8001, 8010 (can be configured)

### 3. API Keys Required

- Alpaca Markets API key and secret
- (Optional) Polygon API key
- (Optional) Alpha Vantage API key

### 4. Security Checklist

- [ ] API keys stored in `.env` (not committed)
- [ ] `.env` file has restrictive permissions (600)
- [ ] Firewall configured
- [ ] SSH key-based authentication enabled
- [ ] Regular security updates enabled

## Systemd Service (Linux)

### 1. Create Service User

```bash
sudo useradd -r -s /bin/bash -d /opt/aion aion
sudo mkdir -p /opt/aion
sudo chown aion:aion /opt/aion
```

### 2. Install Application

```bash
sudo su - aion
cd /opt/aion
git clone https://github.com/stockanalyzerpro/Aion_Analytics.git
cd Aion_Analytics

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with production values
nano .env

# Set permissions
chmod 600 .env
```

### 4. Create Systemd Service File

```bash
sudo nano /etc/systemd/system/aion-analytics.service
```

Add the following content:

```ini
[Unit]
Description=AION Analytics Trading Platform
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=aion
Group=aion
WorkingDirectory=/opt/aion/Aion_Analytics
Environment="PATH=/opt/aion/Aion_Analytics/venv/bin"
ExecStart=/opt/aion/Aion_Analytics/venv/bin/python run_backend.py
Restart=always
RestartSec=10
StandardOutput=append:/opt/aion/Aion_Analytics/logs/aion.log
StandardError=append:/opt/aion/Aion_Analytics/logs/aion-error.log

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/aion/Aion_Analytics

[Install]
WantedBy=multi-user.target
```

### 5. Enable and Start Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable aion-analytics
sudo systemctl start aion-analytics

# Check status
sudo systemctl status aion-analytics

# View logs
sudo journalctl -u aion-analytics -f
```

### 6. Log Rotation

```bash
sudo nano /etc/logrotate.d/aion-analytics
```

```
/opt/aion/Aion_Analytics/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 aion aion
}
```

## Docker Compose

### 1. Create docker-compose.yml

```yaml
version: '3.8'

services:
  aion-backend:
    image: python:3.11-slim
    container_name: aion-backend
    working_dir: /app
    volumes:
      - ./:/app
      - ./data:/app/data
      - ./logs:/app/logs
      - ./da_brains:/app/da_brains
    environment:
      - PYTHONUNBUFFERED=1
    env_file:
      - .env
    command: python run_backend.py
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8010:8010"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8010/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    container_name: aion-prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: aion-grafana
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana-dashboard.json:/etc/grafana/provisioning/dashboards/aion.json
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=your_secure_password
      - GF_USERS_ALLOW_SIGN_UP=false
    ports:
      - "3000:3000"
    restart: unless-stopped

volumes:
  prometheus-data:
  grafana-data:
```

### 2. Deploy with Docker Compose

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f aion-backend

# Stop services
docker-compose down

# Restart after changes
docker-compose restart aion-backend
```

## Kubernetes

### 1. Create Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: aion-analytics
```

### 2. Create ConfigMap for Environment

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aion-config
  namespace: aion-analytics
data:
  DT_DRY_RUN: "0"
  DT_ENABLE_LIVE_TRADING: "1"
  DT_MAX_POSITIONS: "3"
  # Add other non-secret config here
```

### 3. Create Secret for API Keys

```bash
kubectl create secret generic aion-secrets \
  --from-literal=alpaca-key=YOUR_KEY \
  --from-literal=alpaca-secret=YOUR_SECRET \
  -n aion-analytics
```

### 4. Create Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aion-backend
  namespace: aion-analytics
spec:
  replicas: 1
  selector:
    matchLabels:
      app: aion-backend
  template:
    metadata:
      labels:
        app: aion-backend
    spec:
      containers:
      - name: aion
        image: python:3.11-slim
        command: ["python", "run_backend.py"]
        workingDir: /app
        envFrom:
        - configMapRef:
            name: aion-config
        env:
        - name: ALPACA_API_KEY_ID
          valueFrom:
            secretKeyRef:
              name: aion-secrets
              key: alpaca-key
        - name: ALPACA_API_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: aion-secrets
              key: alpaca-secret
        ports:
        - containerPort: 8010
          name: dt-api
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8010
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8010
          initialDelaySeconds: 10
          periodSeconds: 5
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

### 5. Deploy to Kubernetes

```bash
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml

# Check status
kubectl get pods -n aion-analytics
kubectl logs -f deployment/aion-backend -n aion-analytics
```

## Environment Configuration

### Production Environment Variables

```bash
# Trading Controls
DT_DRY_RUN=0                      # 0 for live trading
DT_ENABLE_LIVE_TRADING=1          # Enable live trading
DT_EMERGENCY_STOP_FILE=/var/run/aion/emergency_stop

# Risk Management
DT_MAX_POSITIONS=3
DT_DAILY_LOSS_LIMIT_USD=300.0
DT_MAX_WEEKLY_DRAWDOWN_PCT=8.0
DT_MAX_MONTHLY_DRAWDOWN_PCT=15.0
DT_VIX_SPIKE_THRESHOLD=35.0

# API Keys (use secrets management)
ALPACA_API_KEY_ID=your_key_here
ALPACA_API_SECRET_KEY=your_secret_here

# Monitoring
DT_APP_PORT=8010
APP_PORT=8000
```

## Monitoring Setup

### 1. Install Prometheus

```bash
# Download and install Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvfz prometheus-*.tar.gz
cd prometheus-*

# Copy config
cp /opt/aion/Aion_Analytics/monitoring/prometheus.yml prometheus.yml

# Start Prometheus
./prometheus --config.file=prometheus.yml
```

### 2. Install Grafana

```bash
# Add Grafana repository
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -

# Install
sudo apt-get update
sudo apt-get install grafana

# Start service
sudo systemctl start grafana-server
sudo systemctl enable grafana-server
```

### 3. Configure Grafana

1. Access Grafana at `http://localhost:3000`
2. Login with admin/admin
3. Add Prometheus data source:
   - URL: `http://localhost:9090`
4. Import dashboard from `monitoring/grafana-dashboard.json`

## Backup and Restore

### What to Backup

- Configuration files (`.env`, `dt_knobs.env`)
- Trade logs (`da_brains/intraday/dt_trades.jsonl`)
- State files (`da_brains/intraday/dt_state.json`)
- ML models (`ml_data/`, `ml_data_dt/`)

### Backup Script

```bash
#!/bin/bash
# backup-aion.sh

BACKUP_DIR="/backup/aion/$(date +%Y%m%d)"
APP_DIR="/opt/aion/Aion_Analytics"

mkdir -p "$BACKUP_DIR"

# Backup configuration
cp "$APP_DIR/.env" "$BACKUP_DIR/"
cp "$APP_DIR/dt_knobs.env" "$BACKUP_DIR/"

# Backup trade logs
tar -czf "$BACKUP_DIR/da_brains.tar.gz" "$APP_DIR/da_brains/"

# Backup ML models
tar -czf "$BACKUP_DIR/ml_data.tar.gz" "$APP_DIR/ml_data/" "$APP_DIR/ml_data_dt/"

# Remove backups older than 30 days
find /backup/aion -type d -mtime +30 -exec rm -rf {} +
```

### Restore Procedure

```bash
# Stop service
sudo systemctl stop aion-analytics

# Restore files
cd /opt/aion/Aion_Analytics
cp /backup/aion/20240115/.env .
cp /backup/aion/20240115/dt_knobs.env .
tar -xzf /backup/aion/20240115/da_brains.tar.gz
tar -xzf /backup/aion/20240115/ml_data.tar.gz

# Restart service
sudo systemctl start aion-analytics
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
sudo journalctl -u aion-analytics -n 100 --no-pager

# Check configuration
python -c "from dt_backend.core.knob_validator_dt import validate_and_warn; validate_and_warn()"

# Check permissions
ls -la /opt/aion/Aion_Analytics/.env
```

### Trading Halted

```bash
# Check emergency stop
curl http://localhost:8010/emergency/status

# Check risk rails
curl http://localhost:8010/health

# Review logs
tail -f /opt/aion/Aion_Analytics/logs/dt_backend_*.log | grep stand_down
```

### High Memory Usage

```bash
# Check process
ps aux | grep python

# Reduce universe size
# Edit dt_knobs.env
DT_UNIVERSE_SIZE=50  # Down from 150
```

### No Trades Being Placed

1. Check dry run mode: `grep DT_DRY_RUN .env`
2. Check live trading enabled: `grep DT_ENABLE_LIVE_TRADING .env`
3. Check emergency stop: `curl http://localhost:8010/emergency/status`
4. Check risk rails: Review logs for "stand_down"

## Rolling Updates

### Zero-Downtime Deployment

```bash
# Pull latest changes
cd /opt/aion/Aion_Analytics
git pull origin main

# Install dependencies
source venv/bin/activate
pip install -r requirements.txt

# Run migrations (if any)
# python scripts/migrate.py

# Restart service
sudo systemctl restart aion-analytics

# Verify
curl http://localhost:8010/health
```

## Security Hardening

### 1. Firewall Configuration

```bash
# Allow SSH
sudo ufw allow 22/tcp

# Allow API ports (restrict to trusted IPs)
sudo ufw allow from TRUSTED_IP to any port 8010

# Enable firewall
sudo ufw enable
```

### 2. SSL/TLS Setup (with nginx)

```nginx
server {
    listen 443 ssl http2;
    server_name aion.yourdomain.com;

    ssl_certificate /etc/ssl/certs/aion.crt;
    ssl_certificate_key /etc/ssl/private/aion.key;

    location / {
        proxy_pass http://localhost:8010;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Rate Limiting

Edit dt_knobs.env:
```bash
DT_MIN_TRADE_GAP_MINUTES=15
DT_MAX_TRADES_PER_SYMBOL_PER_DAY=2
```

---

For additional support, refer to README.md and ARCHITECTURE.md
