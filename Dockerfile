FROM nvcr.io/nvidia/pytorch:22.01-py3

RUN apt-get update

RUN pip install torch==1.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchaudio==0.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install pip --upgrade
RUN pip install wandb

COPY ./ /app/
WORKDIR /app

#CMD ["wandb", "login", "7c6c025691da1f01124a2b61a50c7c2932f0fb85"]
CMD ["python", "main.py"]