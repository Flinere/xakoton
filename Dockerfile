FROM python:3.12-alpine

WORKDIR /app

COPY . /app

RUN pip install ultralytics Flask

EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0"]
