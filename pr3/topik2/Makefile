.PHONY: all

bin/tiling: tiling.cu
	nvcc tiling.cu -o $@

bin/tiling-coalesce: tiling-coalesce.cu
	nvcc tiling-coalesce.cu -o $@

bin/tiling-avoiding: tiling-avoiding.cu
	nvcc tiling-avoiding.cu -o $@

bin/naive: naive.cu
	nvcc naive.cu -o $@

bin/naive-2: naive-2.cu
	nvcc naive-2.cu -o $@

bin/cublas-tensor: cublas_tensor.cu
	nvcc cublas_tensor.cu -lcublas -o $@

bin/cublas: cublas.cu
	nvcc cublas.cu -lcublas -o $@

all: bin/tiling bin/tiling-coalesce bin/naive bin/naive-2 bin/tiling-avoiding bin/cublas bin/cublas-tensor



