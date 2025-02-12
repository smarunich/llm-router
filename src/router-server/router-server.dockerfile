FROM nvcr.io/nvidia/tritonserver:24.10-py3

# RUN git clone https://github.com/triton-inference-server/python_backend -b r24.10
COPY src/router-server/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt
# ENV NVIDIA_API_KEY=nvapi-YOUR-KEY-HERE