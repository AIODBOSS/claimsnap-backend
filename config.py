import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    PORT = int(os.getenv('PORT', 5000))
    
    # Database Configuration (PostgreSQL with local SQLite fallback)
    db_url = os.getenv('DATABASE_URL', 'sqlite:///claimsnap.db')
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
        
    SQLALCHEMY_DATABASE_URI = db_url
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Decision thresholds
    AUTO_APPROVE_CONFIDENCE = 85
    AUTO_REJECT_CONFIDENCE  = 20

    # File constraints
    MAX_VIDEO_SIZE_MB = 100
    MAX_VIDEO_DURATION_SECONDS = 20
    FRAMES_TO_EXTRACT = 5

config = Config()
