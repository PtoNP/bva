FROM tensorflow/tensorflow:latest-gpu

 #set environment variables
 ENV LC_ALL C.UTF-8
 ENV LANG C.UTF-8

 # get ubuntu key
 RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
 
 # install ?
 RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
 RUN apt-get install unzip
 RUN apt-get install -y ffmpeg x264 libx264-dev
 
 # install python 3.7
 RUN apt update
 RUN apt install -y software-properties-common
 RUN add-apt-repository ppa:deadsnakes/ppa   
 RUN apt-get update   
 RUN apt install -y python3.7-distutils
 RUN apt -y install python3.7
 RUN apt-get -y install python3-pip
 RUN python3.7 -m pip install --upgrade pip
 
 # set python 3.7 as default python
 RUN alias python3="/usr/bin/python3.7"
 RUN echo 'alias python3="/usr/bin/python3.7"' >> ~/.bashrc
 
 # copy Tracknet + modify predict file
 COPY tracknetv2/TrackNetv2 /TrackNetv2
 RUN rm -rf /TrackNetv2/predict3.py
 COPY tracknetv2/predict3.py /TrackNetv2/predict3.py 
 
 # copy bva files
 COPY bva /bva
 
 # recreate input folder
 RUN mkdir /input_data
 
 COPY requirements.txt /bva/requirements.txt
 
 RUN python3.7 -m pip install -r /bva/requirements.txt
 
 RUN apt update
 RUN apt install libgl1-mesa-glx -y
 
 RUN pip3 install --upgrade requests
 
 RUN pip3 install tensorflow==2.6.0
 
 # RUN python3.7 /bva/hitnet_model.py
 
 EXPOSE 8080
 CMD streamlit run --server.port 8080 --browser.serverAddress 0.0.0.0 --server.enableCORS False --server.enableXsrfProtection False bva/gui_app.py
 
 # docker run --gpus all -p80:8501
 
 
 # DEPLOY TO gcloud 
 # docker tag bva eu.gcr.io/lewagondata864/bva
 # docker push eu.gcr.io/lewagondata864/bva
 # gcloud container images list-tags eu.gcr.io/lewagondata864/bva


 # VM
 # cos-extensions install gpu
 # docker-credential-gcr configure-docker
 
 # sudo mount --bind /var/lib/nvidia /var/lib/nvidia
 # sudo mount -o remount,exec /var/lib/nvidia
 # /var/lib/nvidia/bin/nvidia-smi
 
 # docker run \
 #	--volume /usr/lib/nvidia/lib64:/usr/local/nvidia/lib64 \
 #	--volume /usr/lib/nvidia/bin:/usr/local/nvidia/bin \
  #  --device /dev/nvidia0:/dev/nvidia0 \
   # --device /dev/nvidia-uvm:/dev/nvidia-uvm \
    #--device /dev/nvidiactl:/dev/nvidiactl \
    # --gpus all -p 80:8080 eu.gcr.io/lewagondata864/bva
	
	
  # import tensorflow as tf
  # tf.config.list_physical_devices('GPU')