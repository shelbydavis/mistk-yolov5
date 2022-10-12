# this model uses a generic python v3 base image
FROM python:3.6

# install requirement packages using pip
RUN mkdir -p /usr/src/models/yolov5-model
COPY requirements.txt /usr/src/models/yolov5-model
RUN pip3 install --no-cache-dir -r /usr/src/models/yolov5-model/requirements.txt \
    --trusted-host pypi.python.org --extra-index-url https://download.pytorch.org/whl/cpu

# these lines pull the MISTK base infrastructure into this model
ARG mistk_url
ARG mistk_version

RUN echo "Downloading mistk release from $mistk_url"
# This downloads the latest MISTK wheel file from the Github releases page
ADD $mistk_url /tmp/mistk-$mistk_version-py3-none-any.whl

RUN ls /tmp

# Install it via pip
RUN pip install /tmp/mistk-$mistk_version-py3-none-any.whl


# install the python code for our Template model

COPY . /usr/src/models/yolov5-model
RUN cd /usr/src/models/yolov5-model && python setup.py easy_install -Z .

# these lines set up and run this model using the MISTK infrastructure
EXPOSE 8080
ENTRYPOINT ["python3"]
CMD ["-m", "mistk", "yolov5_model.yolov5", "YOLOv5Model"]
