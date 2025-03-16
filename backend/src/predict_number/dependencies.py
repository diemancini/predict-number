from predict_number.database import SessionLocal, engine
from predict_number import models

# Create database tables
models.Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
