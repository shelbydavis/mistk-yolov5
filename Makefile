MODEL_NAME:=yolov5-model

default_mistk_version:=1.0.0
url_base:=https://github.com/mistkml/mistk/releases
mistk_version:=$(shell curl -s ${url_base}/latest | grep -o -E "href=[\"'](.*)[\"']" | sed 's:.*/::' | tr -d '"')
mistk_version:=$(if $(mistk_version),$(mistk_version),$(default_mistk_version))
mistk_url:=$(url_base)/download/$(mistk_version)/mistk-$(mistk_version)-py3-none-any.whl

version:=1.0.0

docker-image:
	docker build -t sml-models/$(MODEL_NAME) \
    	--build-arg "mistk_version=$(mistk_version)" \
		--build-arg "mistk_url=$(mistk_url)" \
		--network=host \
    	.

docker-test: 
	docker run -it --volume /tmp:/tmp --publish 8080:8080 sml-models/$(MODEL_NAME)
