# app/Dockerfile

FROM python:3.9-slim

WORKDIR /web

# Copy the requirements file into the container
COPY requirements.txt .

RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  software-properties-common \
  git \
  && rm -rf /var/lib/apt/lists/*

#RUN git clone https://github.com/streamlit/streamlit-example.git .

RUN pip3 install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]