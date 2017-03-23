CC = g++
LIBFLAGS = -c
CFLAGS = -Wall -W -fopenmp -O3 -pedantic -std=c++0x
LFLAGS =
LLIBS =

NVCC = nvcc
NVFLAGS = -Wno-deprecated-gpu-targets

ifdef FUNC
	OPTS = -DWHICH_FUNC=$(FUNC)
else
	OPTS = -DWHICH_FUNC=2
endif

all:
	make testAsyncRMP
	make testRMP

clean:
	rm -f bin/*
	rm -f *.o
	rm -f prog_TestAsyncRMP
	rm -f prog_StereoMRF_*
	rm -f prog_TestRMP



testAsyncRMP: testAsyncRMP/TestAsyncRMP.cpp libAsyncRMP/Region.h
	$(CC) testAsyncRMP/TestAsyncRMP.cpp libAsyncRMP/Region.cpp -o bin/testAsyncRMP $(CFLAGS) $(LFLAGS) $(LLIBS) $(OPTS)

testRMP: testRMP/TestRMP.cpp libAsyncRMP/Region.h
	$(CC) testRMP/TestRMP.cpp libAsyncRMP/Region.cpp -o bin/testRMP $(CFLAGS) $(LFLAGS) $(LLIBS) $(OPTS)
