FROM neosapience/pytorch-cpu:1.5.0-v1

RUN mkdir /app
COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

WORKDIR /app

COPY . .
ENV PYTHONPATH=/app

RUN chmod 777 service.py
CMD python3 service.py api