.PHONY: help install backend frontend demo test eval docker-up clean

help:
	@echo "SARang commands"
	@echo "make install       Install backend dependencies"
	@echo "make backend       Run FastAPI backend"
	@echo "make frontend      Run React frontend"
	@echo "make demo          Generate demo data"
	@echo "make test          Run backend tests"
	@echo "make eval          Run evaluation script"
	@echo "make docker-up     Run using Docker Compose"
	@echo "make clean         Remove caches"

install:
	cd backend && pip install -r requirements.txt

backend:
	cd backend && uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

demo:
	python scripts/make_demo_data.py

test:
	cd backend && pytest -v

eval:
	python evaluation/evaluate.py --help

docker-up:
	docker compose up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache
