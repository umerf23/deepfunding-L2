.PHONY: install test run serve clean
install:
	pip install -r requirements.txt
test:
	python -m pytest tests/ -v
run:
	python -m src.pipeline.run_pipeline
serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
clean:
	rm -rf data/processed/cache artifacts/*.csv artifacts/*.gexf .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
