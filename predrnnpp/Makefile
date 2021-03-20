include make_docker

# Set default parameters
PROJECT ?= masterproef
COMPONENT ?= 
VERSION ?= latest

NS = docker.io

IMAGE_NAME = $(PROJECT)_$(COMPONENT)
CONTAINER_NAME = $(strip $(USER))_$(COMPONENT)
CONTAINER_INSTANCE ?= default

ARM_PREFIX = rpi
JETSON_PREFIX = j4t

.PHONY: build build-arm build-j4t \
    push push-arm push-j4t \
    shell shell-arm shell-j4t \
    run run-arm run-j4t \
    start start-arm start-j4t \
    stop stop-arm stop-j4t \
    rm rm-arm rm-j4t \
    release release-arm release-j4t

build: Dockerfile
	docker build -t $(NS)/$(IMAGE_NAME):$(VERSION) --build-arg ARCH=amd64/ -f Dockerfile .

build-arm: Dockerfile.arm
	docker build -t $(NS)/$(ARM_PREFIX)_$(IMAGE_NAME):$(VERSION) --build-arg ARCH=arm64/v8 -f Dockerfile.arm .

build-j4t: Dockerfile.j4t
	docker build -t $(NS)/$(JETSON_PREFIX)_$(IMAGE_NAME):$(VERSION) --build-arg ARCH=arm64/j4t -f Dockerfile.j4t .

push:
	docker push $(NS)/$(IMAGE_NAME):$(VERSION)

push-arm:
	docker push $(NS)/$(ARM_PREFIX)_$(IMAGE_NAME):$(VERSION)

push-j4t:
	docker push $(NS)/$(JETSON_PREFIX)_$(IMAGE_NAME):$(VERSION)

shell:
	xhost + && docker run --rm --name $(CONTAINER_NAME)-$(CONTAINER_INSTANCE) -i -t $(PORTS) $(VOLUMES) -v $(WORKSPACE):/workspace $(ENV) $(NS)/$(IMAGE_NAME):$(VERSION) /bin/bash

shell-arm:
	xhost + && docker run --rm --name $(ARM_PREFIX)-$(CONTAINER_NAME)-$(CONTAINER_INSTANCE) -i -t $(PORTS) $(VOLUMES) -v $(WORKSPACE):/workspace $(ENV) $(NS)/$(ARM_PREFIX)_$(IMAGE_NAME):$(VERSION) /bin/bash

shell-j4t:
	xhost + && docker run --rm --name $(JETSON_PREFIX)-$(CONTAINER_NAME)-$(CONTAINER_INSTANCE) -i -t $(PORTS) $(VOLUMES) -v $(WORKSPACE):/workspace $(ENV) $(NS)/$(JETSON_PREFIX)_$(IMAGE_NAME):$(VERSION) /bin/bash

run:
	xhost + && docker run --rm --user=$(shell id -u):$(shell id -g) --name $(CONTAINER_NAME)-$(CONTAINER_INSTANCE) $(PORTS) $(VOLUMES) $(ENV) $(NS)/$(IMAGE_NAME):$(VERSION)

run-arm:
	xhost + && docker run --rm --name $(ARM_PREFIX)-$(CONTAINER_NAME)-$(CONTAINER_INSTANCE) $(PORTS) $(VOLUMES) $(ENV) $(NS)/$(ARM_PREFIX)_$(IMAGE_NAME):$(VERSION)

run-j4t:
	xhost + && docker run --rm --name $(JETSON_PREFIX)-$(CONTAINER_NAME)-$(CONTAINER_INSTANCE) $(PORTS) $(VOLUMES) $(ENV) $(NS)/$(JETSON_PREFIX)_$(IMAGE_NAME):$(VERSION)

start:
	xhost + && docker run -d --name $(CONTAINER_NAME)-$(CONTAINER_INSTANCE) $(PORTS) $(VOLUMES) $(ENV) $(NS)/$(IMAGE_NAME):$(VERSION)

start-arm:
	xhost + && docker run -d --name $(ARM_PREFIX)-$(CONTAINER_NAME)-$(CONTAINER_INSTANCE) $(PORTS) $(VOLUMES) $(ENV) $(NS)/$(ARM_PREFIX)_$(IMAGE_NAME):$(VERSION)

start-j4t:
	xhost + && docker run -d --name $(JETSON_PREFIX)-$(CONTAINER_NAME)-$(CONTAINER_INSTANCE) $(PORTS) $(VOLUMES) $(ENV) $(NS)/$(JETSON_PREFIX)_$(IMAGE_NAME):$(VERSION)

stop:
	docker stop $(CONTAINER_NAME)-$(CONTAINER_INSTANCE)

stop-arm:
	docker stop $(ARM_PREFIX)_$(CONTAINER_NAME)-$(CONTAINER_INSTANCE)

stop-j4t:
	docker stop $(JETSON_PREFIX)_$(CONTAINER_NAME)-$(CONTAINER_INSTANCE)

rm:
	docker rm $(CONTAINER_NAME)-$(CONTAINER_INSTANCE)

rm-arm:
	docker rm $(ARM_PREFIX)-$(CONTAINER_NAME)-$(CONTAINER_INSTANCE)

rm-j4t:
	docker rm $(JETSON_PREFIX)-$(CONTAINER_NAME)-$(CONTAINER_INSTANCE)

release: build
	make push -e VERSION=$(VERSION)

release-arm: build-arm
	make push-arm -e VERSION=$(VERSION)

release-j4t: build-j4t
	make push-j4t -e VERSION=$(VERSION)

default: build
