# ClaimSnap Backend - AI Claim Processing

This repository contains the Python/Flask backend for **ClaimSnap**, an automated insurance claim processing system.

## Core Features
* **YOLOv8 Computer Vision**: Automatically analyzes video/image uploads to detect asset damage.
* **Automated Decision Engine**: Routes claims to Approved, Rejected, or Needs Review based on AI findings and past claim history.
* **Human-in-the-Loop Override**: API endpoints for adjusters to override AI decisions, logging calibration data for future model improvements.
* **PostgreSQL Memory Layer**: Stores claim history and calibration logs.

## Tech Stack
* Flask / Python
* Ultralytics YOLOv8 (Computer Vision)
* SQLAlchemy
* Gunicorn (Production Server)
