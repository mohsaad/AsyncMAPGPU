CC = g++
LIBFLAGS = -c
CFLAGS = -Wall -W -fopenmp -O3 -pedantic -std=c++0x -g
LFLAGS =
LLIBS =

NVCC = nvcc
NVLIBFLAGS = -c -std=c++11 -rdc=true -arch=sm_30 --expt-relaxed-constexpr
NVFLAGS = -Wno-deprecated-gpu-targets -O0
NVDEBUG = -g -G

CUDAFLAGS= -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart
ifdef FUNC
	OPTS = -DWHICH_FUNC=$(FUNC)
else
	OPTS = -DWHICH_FUNC=2
endif

all:
	make gpuTestAsyncRMP
	make testAsyncRMP

clean:
	rm -f bin/*
	rm -f *.o
	rm -f prog_TestAsyncRMP
	rm -f prog_StereoMRF_*
	rm -f prog_TestRMP



testAsyncRMP: testAsyncRMP/TestAsyncRMP.cpp libAsyncRMP/Region.h
	$(CC) testAsyncRMP/TestAsyncRMP.cpp libAsyncRMP/Region.cpp -o bin/testAsyncRMP $(CFLAGS) -g $(LFLAGS) $(LLIBS) $(OPTS)

testRMP: testRMP/TestRMP.cpp libAsyncRMP/Region.h
	$(CC) testRMP/TestRMP.cpp libAsyncRMP/Region.cpp -o bin/testRMP $(CFLAGS) $(LFLAGS) $(LLIBS) $(OPTS)

gpuTestAsyncRMP: testAsyncRMP/testGPUAsyncRMP.cpp gpu/Region.h
	$(NVCC) $(NVLIBFLAGS) $(NVDEBUG) gpu/Region.cu -o bin/cudaRegion.o $(NVFLAGS)
	$(NVCC) $(NVLIBFLAGS) $(NVDEBUG) gpu/RegionImpl.cu -o bin/Region.o $(NVFLAGS)
	$(NVCC) $(NVLIBFLAGS) $(NVDEBUG) testAsyncRMP/testGPUAsyncRMP.cu -o bin/gpuTestAsync.o $(NVFLAGS)
	$(NVCC) $(NVDEBUG) bin/gpuTestAsync.o bin/cudaRegion.o bin/Region.o -o bin/gpuTestAsyncRMP $(NVFLAGS)
	# $(CC) $(CUDAFLAGS) testAsyncRMP/testGPUAsyncRMP.cpp  gpu/Region.cpp bin/cudaRegion.o -o bin/gpuTestAsyncRMP $(CFLAGS) $(LFLAGS) $(LLIBS) $(OPTS)
