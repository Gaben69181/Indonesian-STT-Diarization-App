# Stage 1: Build React frontend
FROM node:18-alpine AS build-frontend

WORKDIR /app/frontend

# Copy package.json and package-lock.json (or yarn.lock)
COPY frontend/package.json frontend/package-lock.json* ./
# If you use yarn, copy yarn.lock instead and use yarn install/build

# Install dependencies
RUN npm install

# Copy the rest of frontend source code
COPY frontend/ ./

# Build the frontend
RUN npm run build

# Stage 2: Setup Python backend and serve frontend
FROM python:3.9-slim

WORKDIR /app

# Set environment variables (optional, can be set in Hugging Face Spaces secrets)
# ENV HF_TOKEN="your_hugging_face_token_here" # Better to set this in HF Spaces secrets

# Install system dependencies that might be needed by PyTorch or other libraries
# For example, libsndfile1 for soundfile, ffmpeg for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install Python dependencies
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend application code
COPY backend/ ./backend/

# Copy built frontend static files from the build-frontend stage
COPY --from=build-frontend /app/frontend/dist ./frontend_dist

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application
# The backend will need to be configured to serve static files from ./frontend_dist
# Or, for Hugging Face Spaces, you might just run the backend and HF handles the static part if configured.
# For now, we assume the backend will serve the frontend.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]