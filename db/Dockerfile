# Use the official PostgreSQL image as the base image
FROM postgres:latest

# Set environment variables for PostgreSQL
ENV POSTGRES_USER="username1"
ENV POSTGRES_PASSWORD="super_secrect_password_1"
ENV POSTGRES_DB="machine_learning_log_database"

# Copy initialization scripts (optional)
#COPY init.sql /docker-entrypoint-initdb.d/

# Install additional dependencies (if needed)
RUN apt-get update && apt-get install -y \
  && rm -rf /var/lib/apt/lists/*

# Expose the PostgreSQL port
EXPOSE 5432

# Start PostgreSQL
CMD ["postgres"]