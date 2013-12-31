FLAGS=-Wall
OCV_FLAGS=-lopencv_highgui -lopencv_core -L/usr/local/lib

FILTERS=tiltshift.o blur.o sharpen.o resize.o mask.o filter.o # @TODO: add missing

all: filtry_gpu

filtry_gpu: src/main.cpp utils.o $(FILTERS)
	g++ src/main.cpp utils.o $(FILTERS) $(FLAGS) $(OCV_FLAGS) -o filtry_gpu

utils.o: src/utils.cpp
	gcc src/utils.cpp $(OCV_FLAGS) -c -o utils.o

filter.o: src/filter.cpp
	gcc src/filter.cpp -c -o filter.o

# filters

ifdef CUDA
DIR=gpu
EXT=cu
COMPILER=nvcc
else
DIR=cpu
EXT=cpp
COMPILER=gcc
endif
tiltshift.o: $(DIR)/tiltshift.$(EXT)
	$(COMPILER) $(DIR)/tiltshift.$(EXT) -c -o tiltshift.o
resize.o: $(DIR)/resize.$(EXT)
	$(COMPILER) $(DIR)/resize.$(EXT) -c -o resize.o
mask.o: $(DIR)/mask.$(EXT)
	$(COMPILER) $(DIR)/mask.$(EXT) -c -o mask.o
sharpen.o: src/sharpen.cpp
	gcc src/sharpen.cpp -c -o sharpen.o
blur.o: src/blur.cpp
	gcc src/blur.cpp -c -o blur.o

clean:
	-rm -rf *.o
	-rm -f filtry_gpu
