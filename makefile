all: filtry_gpu

FLAGS=#-Wall
OCV_FLAGS=-lopencv_highgui -lopencv_core -L/usr/local/lib
CUDA:=$(shell nvcc --version 2> /dev/null)
FILTERS=tiltshift.o blur.o sharpen.o resize.o mask.o filter.o

CPU_DEPENDENCIES=utils.o $(FILTERS)

ifdef CUDA
DEPENDENCIES=$(CPU_DEPENDENCIES) gpu.o
CUDA_FLAGS=-L/usr/local/cuda/lib* -lcudart -lcuda
DIR=gpu
EXT=cu
COMPILER=nvcc
CUDA=-DCUDA=1
else
DEPENDENCIES=$(CPU_DEPENDENCIES)
DIR=cpu
EXT=cpp
COMPILER=g++
CUDA=-DCUDA=0
endif

filtry_gpu: src/main.cpp $(DEPENDENCIES)
	g++ src/main.cpp $(DEPENDENCIES) $(FLAGS) $(OCV_FLAGS) $(CUDA_FLAGS) -o filtry_gpu

utils.o: src/utils.cpp
	$(COMPILER) src/utils.cpp -c -o utils.o

filter.o: src/filter.cpp
	$(COMPILER) src/filter.cpp $(CUDA) -c -o filter.o

gpu.o: $(DIR)/gpu.$(EXT)
	$(COMPILER) $(DIR)/gpu.$(EXT) $(CUDA) -c -o gpu.o

tiltshift.o: $(DIR)/tiltshift.$(EXT)
	$(COMPILER) $(DIR)/tiltshift.$(EXT) $(CUDA) -c -o tiltshift.o
resize.o: $(DIR)/resize.$(EXT)
	$(COMPILER) $(DIR)/resize.$(EXT) $(CUDA) -c -o resize.o
mask.o: $(DIR)/mask.$(EXT)
	$(COMPILER) $(DIR)/mask.$(EXT) $(CUDA) -c -o mask.o
sharpen.o: src/sharpen.cpp
	$(COMPILER) src/sharpen.cpp $(CUDA) -c -o sharpen.o
blur.o: src/blur.cpp
	$(COMPILER) src/blur.cpp $(CUDA) -c -o blur.o

clean:
	-rm -rf *.o
	-rm -f filtry_gpu

rebuild: clean all
