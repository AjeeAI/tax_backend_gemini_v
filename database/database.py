# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker, declarative_base
# from dotenv import load_dotenv
# import os
# import logging


# load_dotenv()

# DB_USER = os.getenv("DB_USER")
# DB_PASSWORD = os.getenv("DB_PASSWORD")
# DB_HOST = os.getenv("DB_HOST")
# DB_PORT = os.getenv("DB_PORT")
# DB_NAME = os.getenv("DB_NAME")


# DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # 2. Build the full path to the certificate
# ssl_cert_path = os.path.join(BASE_DIR, "isrgrootx1.pem")


# engine = create_engine(
#     DATABASE_URL,
#     connect_args={
#         "ssl": {
#             "ca": ssl_cert_path
#         }
#     }
# )
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# # Dependency to get DB session
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # Auto-create tables on import
# def init_db():
#     """Create database tables if they don't exist."""
#     try:
#         # Import models here to avoid circular imports
#         from database.models import User
        
#         # Create all tables
#         Base.metadata.create_all(bind=engine)
#         logging.info("Database tables created/verified")
        
#     except Exception as e:
#         logging.error(f"Failed to create database tables: {e}")
#         raise

# # Call init_db when this module is imported
# init_db()




from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os
import logging

load_dotenv()

# 1. Load variables
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# 2. Construct the connection URL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Build the full path to the certificate
ssl_cert_path = os.path.join(BASE_DIR, "isrgrootx1.pem")


engine = create_engine(
    DATABASE_URL,
    connect_args={
        "ssl": {
            "ca": ssl_cert_path
        }
    }
)



SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Create database tables if they don't exist."""
    try:
        # Import models here to avoid circular imports
        from database.models import User
        
        Base.metadata.create_all(bind=engine)
        logging.info("Database tables created/verified")
        
    except Exception as e:
        logging.error(f"Failed to create database tables: {e}")
        raise

# It is generally better to call this from main.py, 
# but keeping it here matches your previous setup.
init_db()