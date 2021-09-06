package:
	python setup.py install
	pip install -e .

test:
	./tests/test-import.py

test-gpu:
	queue.pl -l q_gpu -V test_gpu.log tests/test-gpu.py

clean:
	rm -rf build dist *.egg-info *.egg-info/ *pkwrap*.so

cleanly: clean package

all: clean package
