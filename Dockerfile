FROM python:3.9-slim

RUN pip3 install --upgrade pip
RUN apt-get update && apt-get install -y unzip

WORKDIR /brat
COPY app/. ./
RUN pip3 install -r requirements.txt

ADD https://github.com/nlplab/brat/archive/refs/heads/master.zip /tmp/brat-latest.zip
RUN unzip /tmp/brat-latest.zip -d /tmp/
RUN cp /tmp/brat-master/* . -r
RUN echo "application/xhtml+xml xhtml" > /etc/mime.types

ADD app/config/* ./
RUN mkdir work

EXPOSE 8001
CMD ["python", "start.py", "--label", "drug", "--label_list", "O", "B-drug", "I-drug"]
