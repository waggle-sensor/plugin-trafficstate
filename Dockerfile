FROM waggle/plugin-base:1.1.1-ml-torch1.9.0

RUN apt-get update \
  && apt-get install -y \
  ffmpeg \
  && rm -rf /var/lib/apt/lists/*
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install --upgrade pip
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt

COPY utils/ /app/utils
COPY models/ /app/models
COPY app.py app_utils.py sort.py coco.names /app/

ADD https://web.lcrc.anl.gov/public/waggle/models/vehicletracking/yolov7.pt /app/model.pt

WORKDIR /app
ENTRYPOINT ["python3", "-u", "/app/app.py"]
