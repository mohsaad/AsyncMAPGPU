CC = g++
CFLAGS = -Wall -W -fopenmp -O3 -pedantic -std=c++0x
LFLAGS =
LLIBS =

NVCC = nvcc
NVFLAGS = -use-fast-math

ifdef FUNC
	OPTS = -DWHICH_FUNC=$(FUNC)
else
	OPTS = -DWHICH_FUNC=1
endif

all:
	make prog_TestAsyncRMP
	make prog_TestRMP

clean:
	rm -f *.o
	rm -f prog_TestAsyncRMP
	rm -f prog_StereoMRF_*
	rm -f prog_TestRMP


prog_TestAsyncRMP: testAsyncRMP/TestAsyncRMP.cpp libAsyncRMP/Region.h
	$(CC) testAsyncRMP/TestAsyncRMP.cpp -o bin/prog_TestAsyncRMP $(CFLAGS) $(LFLAGS) $(LLIBS) $(OPTS)

prog_TestRMP: testRMP/TestRMP.cpp libAsyncRMP/Region.h
	$(CC) testRMP/TestRMP.cpp -o bin/prog_TestRMP $(CFLAGS) $(LFLAGS) $(LLIBS) $(OPTS)

# gpu_TestAsyncRMP: gpu/testGPU.cpp gpu/Region.cu
# 	$(NVCC) testGPU.cpp -o testGPUAsync
