CC = g++
LIBFLAGS = -c
CFLAGS = -Wall -W -fopenmp -O3 -pedantic -std=c++0x -g
LFLAGS =
LLIBS =

NVCC = nvcc
NVLIBFLAGS = -c -std=c++11 -rdc=true -arch=sm_61 --expt-relaxed-constexpr
NVFLAGS = -Wno-deprecated-gpu-targets -O0
NVDEBUG = -g -G

CUDAFLAGS= -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart
ifdef FUNC
	OPTS = -DWHICH_FUNC=$(FUNC)
else
	OPTS = -DWHICH_FUNC=1
endif

all:
	#mkdir output
	make gpuTestStereoMRF
clean:
	rm -f bin/*
	rm -f *.o
	rm -f prog_TestAsyncRMP
	rm -f prog_StereoMRF_*
	rm -f prog_TestRMP
	rm -f output/*


testAsyncRMP: testAsyncRMP/TestAsyncRMP.cpp libAsyncRMP/Region.h
	$(CC) testAsyncRMP/TestAsyncRMP.cpp libAsyncRMP/Region.cpp -o bin/testAsyncRMP $(CFLAGS) -g $(LFLAGS) $(LLIBS) $(OPTS)

testRMP: testRMP/TestRMP.cpp libAsyncRMP/Region.h
	$(CC) testRMP/TestRMP.cpp libAsyncRMP/Region.cpp -o bin/testRMP $(CFLAGS) $(LFLAGS) $(LLIBS) $(OPTS)


prog_StereoMRF_$(FUNC): StereoMRF/StereoMRF.cpp libAsyncRMP/Region.h
	$(CC) StereoMRF/StereoMRF.cpp -o prog_StereoMRF_$(FUNC) $(CFLAGS) $(LFLAGS) $(LLIBS) $(OPTS)

prog_StereoMRF: prog_StereoMRF_$(FUNC)

gpuTestAsyncRMP: testAsyncRMP/testGPUAsyncRMP.cpp gpu/Region.h
	$(NVCC) $(NVLIBFLAGS) $(NVDEBUG) gpu/Region.cu -o output/cudaRegion.o $(NVFLAGS)
	$(NVCC) $(NVLIBFLAGS) $(NVDEBUG) gpu/RegionImpl.cu -o output/Region.o $(NVFLAGS)
	$(NVCC) $(NVLIBFLAGS) $(NVDEBUG) testAsyncRMP/testGPUAsyncRMP.cu -o output/gpuTestAsync.o $(NVFLAGS)
	$(NVCC) $(NVDEBUG) output/gpuTestAsync.o output/cudaRegion.o output/Region.o -o bin/gpuTestAsyncRMP $(NVFLAGS)

gpuTestStereoMRF: StereoMRF/StereoMRF.cu gpu/Region.h
	$(NVCC) $(NVLIBFLAGS) $(NVDEBUG) gpu/Region.cu -o output/cudaRegion.o $(NVFLAGS)
	$(NVCC) $(NVLIBFLAGS) $(NVDEBUG) gpu/RegionImpl.cu -o output/Region.o $(NVFLAGS)
	$(NVCC) $(NVLIBFLAGS) $(NVDEBUG) StereoMRF/StereoMRF.cu -o output/stereoMRF.o $(NVFLAGS)
	$(NVCC) $(NVDEBUG) output/stereoMRF.o output/cudaRegion.o output/Region.o -o bin/gpuStereoMRF $(NVFLAGS)
