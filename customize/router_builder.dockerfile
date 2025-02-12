FROM nvcr.io/nvidia/pytorch:24.10-py3

RUN pip install transformers 
RUN pip install tritonclient[all]
RUN pip install --upgrade ipywidgets
RUN pip install sentencepiece