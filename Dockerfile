FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

RUN apt update
RUN apt install libgl1-mesa-glx -y
# add requirements
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r /app/requirements.txt

# Define environment variable

ENV MODEL "/app/models/model_v3.hdf5"
ENV CATEGORIES "Vestidos,Jeans,Poleras/Camisas"
ENV MIN_CONFIDENCE "60"

COPY ./models/* /app/models/
COPY ./main.py /app/main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

