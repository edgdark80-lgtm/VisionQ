IMAGE ?= visionq:2.0.0
PORT ?= 8000

.PHONY: install train test lint build run stop smoke load compose evaluate

install:            ## Entorno de desarrollo local
	pip install -r requirements-dev.txt

train:              ## Reentrena ambos modelos (requiere data/)
	python training/train_failure.py
	python training/train_vision.py

test:               ## Pruebas automatizadas (pytest)
	pytest -q

lint:
	ruff check app tests scripts training

build:              ## Construye la imagen Docker
	docker build -t $(IMAGE) .

run:                ## Ejecuta el contenedor en segundo plano
	docker run -d --name visionq -p $(PORT):8000 --read-only --tmpfs /tmp $(IMAGE)

stop:
	docker rm -f visionq

smoke:              ## Prueba de humo E2E contra el contenedor
	bash scripts/smoke_test.sh http://localhost:$(PORT)

load:               ## Prueba de carga
	python scripts/load_test.py --base http://localhost:$(PORT)

evaluate:           ## Evaluación de robustez de los modelos
	python scripts/evaluate_models.py

compose:            ## Alternativa con docker compose
	docker compose up --build -d
