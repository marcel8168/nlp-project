FROM python:3.9-slim

RUN pip3 install --upgrade pip
RUN apt-get update && apt-get install -y unzip

WORKDIR /brat
COPY app/. ./
RUN pip3 install -r requirements.txt

# forked from https://github.com/dtoddenroth/annotationimage
# download brat
ADD https://github.com/nlplab/brat/archive/refs/heads/master.zip /tmp/brat-latest.zip
RUN unzip /tmp/brat-latest.zip -d /tmp/
RUN cp /tmp/brat-master/* . -r
RUN echo "application/xhtml+xml xhtml" > /etc/mime.types

# apply configuration
ADD app/config/* ./
RUN mkdir work

EXPOSE 8001

# start script and pass command line arguments
CMD ["python", "start.py", "--label", "drug", "--label_list", "O", "B-drug", "I-drug"]
