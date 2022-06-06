FROM baiduz/tracknetv2

RUN apt-get install -y git
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install pyqt5
RUN pip3 install pandas
RUN pip3 install PyMySQL
RUN pip3 install opencv-python
RUN pip3 install imutils
RUN pip3 install Pillow
RUN pip3 install piexif
RUN pip3 install -U scikit-learn
RUN pip3 install keras

COPY ./TrackNetv2 TrackNetv2/
