import os

from sqlalchemy import create_engine, URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
app_env = os.getenv("APP_ENV")

SQLALCHEMY_DATABASE_URL = URL.create(
    "postgresql+psycopg2",
    username=f"{db_user}",
    password=f"{db_password}",  # plain (unescaped) text
    host=f"{db_host}",
    database=f"{db_name}",
    port=db_port,
)

# Create the database engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create a configured Session class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()
print(Base)
