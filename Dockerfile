# ── Stage 1: Build React Frontend ────────────
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/frontend/package*.json ./
RUN npm install
COPY frontend/frontend/ ./
RUN npm run build

# ── Stage 2: Python Backend ───────────────────
FROM python:3.11-slim

WORKDIR /app/backend

# Install Python dependencies
COPY requirements.txt ../
RUN pip install --no-cache-dir -r ../requirements.txt

# Copy backend code
COPY backend/ ./

# Copy model
COPY model/ ../model/

# Copy React build from Stage 1
COPY --from=frontend-build /app/frontend/build ../frontend/frontend/build

# Expose port
EXPOSE 8080

# Start Flask directly — no cd needed
ENV PORT=8080
CMD ["python", "app.py"]