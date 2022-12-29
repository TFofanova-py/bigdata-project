FROM python:latest

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY app.py app.py
COPY stemmed_writers_rus.json stemmed_writers_rus.json
COPY templates/view_response.html templates/view_response.html
COPY artifacts/model_34_baseline:v1/ artifacts/model_34_baseline:v1/

EXPOSE 5000

CMD ["flask", "run"]
