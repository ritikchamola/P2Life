# Makefile

.PHONY: install run test clean

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r numpy matplotlib

# Run the main training script
run:
	@echo "Running training script..."
	python3 main.py

# Run the test script
test:
	@echo "Running test script..."
	python3 test.py

# Clean up (optional)
clean:
	@echo "Cleaning up..."
	rm -f *.pyc
	rm -rf __pycache__
