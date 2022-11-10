lint:
	isort .
	black .

ut:
	pytest test
