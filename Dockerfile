FROM neosapience/pytorch-cpu:1.5.0-v1

WORKDIR /app

COPY . .

RUN pip install -r /app/requirements.txt

ENV PYTHONPATH=/app

CMD python3 service.py api
