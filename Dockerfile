FROM waggle/plugin-base:1.1.1-ml-cuda10.2-arm64

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY deep_sort /app/deep_sort
COPY detection /app/detection
COPY tool /app/tool
COPY app.py deepsort.py siamese_net.py yolov4.py /app/

ARG SAGE_STORE_URL="https://osn.sagecontinuum.org"
ARG BUCKET_ID_MODEL="cafb2b6a-8e1d-47c0-841f-3cad27737698"

ENV SAGE_STORE_URL=${SAGE_STORE_URL} \
    BUCKET_ID_MODEL=${BUCKET_ID_MODEL}

RUN sage-cli.py storage files download ${BUCKET_ID_MODEL} model640.pt --target /app/model640.pt \
  && sage-cli.py storage files download ${BUCKET_ID_MODEL} yolov4.cfg --target /app/yolov4.cfg \
  && sage-cli.py storage files download ${BUCKET_ID_MODEL} yolov4.weights --target /app/yolov4.weights

WORKDIR /app
ENTRYPOINT ["python3", "-u", "/app/app.py"]
