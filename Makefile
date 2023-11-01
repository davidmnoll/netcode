

.PHONY: rundemo
rundemo:
	cd demo && make run

.PHONY: test
test:
	python -m unittest discover -s tests -p '*_test.py'


# why is this broken? 
run-fe:
	cd demo/frontend && npm run dev
	
run-be:
	cd demo/backend && python ./app.y

build_cuda:
	cd cuda_test && python setup.py build_ext --inplace && pip install .

