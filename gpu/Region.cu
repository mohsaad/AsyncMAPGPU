#include "Region.h"
#include <cuda.h>

// our kernel for edge updates
// parameters:
// g: graph
// epsilon: epsilon
// numThreadUpdates: number of updates in each thread
// lambdaGlobal: global lambda array
// runFlag: a flag that controls when we want to terminate the array
// rangeRandNums: random numbers (defined by the graph)
template<typename T, typename S>
__global__ void EdgeUpdateKernel(MPGraph<T, S>* g, T epsilon, size_t* numThreadUpdates, T* lambdaGlobal, volatile int* runFlag, int numThreads)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;

    if(tx < numThreads)
    {
        int uid;
        curandState_t state;
        curand_init(clock64(),tx,0,&state);

        // allocate space for edge workspace
        typename MPGraph<T, S>::REdgeWorkspaceID rew;
        rew = g->AllocateReparameterizeEdgeWorkspaceMem(epsilon);

        // allocate an array that will act as our base
        size_t msgSize = g->GetLambdaSize();
        T* devLambdaBase = (T*)malloc(msgSize * sizeof(T));
        //memset(devLambdaBase, T(0), sizeof(T) * msgSize);

        int rangeRandNums = g->NumberOfEdges();


        uid = floorf(curand_uniform(&state) * rangeRandNums);
	    g->CopyMessagesForEdge(lambdaGlobal, devLambdaBase, uid);
	    g->ReparameterizeEdge(devLambdaBase, uid, epsilon, false, rew);
	    g->UpdateEdge(devLambdaBase, lambdaGlobal, uid, false);

        g->DeAllocateReparameterizeEdgeWorkspaceMem(rew);
        free(devLambdaBase);
    }
}

__device__ bool checkFlag(volatile bool* flag)
{
    return *flag;
}


template<typename T, typename S>
__global__ void RegionUpdateKernel(MPGraph<T, S>* g, T epsilon, size_t* numThreadUpdates, T* lambdaGlobal, T* lambdaBase, volatile bool* runFlag, int numThreads)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;

    if(tx < numThreads)
    {
        int uid;
        curandState_t state;
        curand_init(clock64(),tx,0,&state);

        // allocate space for edge workspace
        typename MPGraph<T, S>::RRegionWorkspaceID rew;
        rew = g->AllocateReparameterizeRegionWorkspaceMem(epsilon);

        // allocate an array that will act as our base
        size_t msgSize = g->GetLambdaSize();
	    T* devLambdaBase = (T*)malloc(msgSize * sizeof(T));
	    memset(devLambdaBase, T(0), sizeof(T) * msgSize);

        int rangeRandNums = g->NumberOfRegionsWithParents();

        uid = floorf(curand_uniform(&state) * rangeRandNums);
	    g->CopyMessagesForStar(lambdaGlobal, devLambdaBase, uid);
	    g->ReparameterizeRegion(devLambdaBase, uid, epsilon, false, rew);
	    g->UpdateRegion(devLambdaBase, lambdaGlobal, uid, false);

    
        // free device pointers
        g->DeAllocateReparameterizeRegionWorkspaceMem(rew);
        free(devLambdaBase);


     }
}


template<typename T, typename S>
int CudaAsyncRMPThread<T,S>::CudaRunMP(MPGraph<T, S>& g, T epsilon, int numIterations, int numThreads, int WaitTimeInMS) {

    size_t msgSize = g.HostGetLambdaSize();

    std::cout << "Num threads " << numThreads << std::endl;

    // handle this case later.i
    if (msgSize == 0) {
        typename MPGraph<T, S>::DualWorkspaceID dw = g.HostAllocateDualWorkspaceMem(epsilon);
        std::cout << "0: " << g.HostComputeDual(NULL, epsilon, dw) << std::endl;
        g.HostDeAllocateDualWorkspaceMem(dw);
        return 0;
    }
    std::cout << std::setprecision(15);

    // allocate device pointers for lambda global
    T* devLambdaGlobal = NULL;
    gpuErrchk(cudaMalloc((void**)&devLambdaGlobal, sizeof(T) * msgSize));
    gpuErrchk(cudaMemset((void*)devLambdaGlobal, T(0), sizeof(T)*msgSize));


    // allocate on host memory for cuda streaming
    T* lambdaGlob = NULL;
    gpuErrchk(cudaMallocHost((void**)&lambdaGlob, sizeof(T)*msgSize));
    gpuErrchk(cudaMemset((void*)lambdaGlob, T(0), sizeof(T)*msgSize));




    // allocate space and copy graph to GPU
    MPGraph<T,S>* gPtr = NULL;
    gpuErrchk(cudaMalloc((void**)&gPtr, sizeof(g)));
    gpuErrchk(cudaMemcpy(gPtr, &g, sizeof(g), cudaMemcpyHostToDevice));

    // initialize the number of region updates
    size_t* numThreadUpdates = NULL;
    size_t* hostThreadUpdates = new size_t[numThreads];
    gpuErrchk(cudaMalloc((void**)&numThreadUpdates, numThreads * sizeof(size_t)));
    gpuErrchk(cudaMemset((void*)numThreadUpdates, 0, numThreads * sizeof(size_t)));


    // allocate all the base lambdas
    T* indivLambda;
    gpuErrchk(cudaMalloc((void**)&indivLambda, sizeof(T)*msgSize*numThreads));
    gpuErrchk(cudaMemset((void*)indivLambda, 0, sizeof(T)*msgSize*numThreads)); 

    // allocate run flag
    bool* devRunFlag = NULL;
    bool tmpTest = true;
    gpuErrchk(cudaMalloc((void**)&devRunFlag, sizeof(bool)));
    gpuErrchk(cudaMemcpy(devRunFlag, &tmpTest, sizeof(bool), cudaMemcpyHostToDevice));

    // create an asynchronous cuda stream
    // we only have two streams, the main (CPU) stream, and the GPU one
    // CPU stream only copies back every so often (or writes to the GPU)
    // GPU is executing
    cudaStream_t streamCopy, streamExec;
    gpuErrchk(cudaStreamCreate(&streamCopy));
    gpuErrchk(cudaStreamCreate(&streamExec));


    // create a ThreadSync object (not necessary at all, but hey, I wanna
    // make sure this actually works)
    ThreadSync<T, S> sy(numThreads, lambdaGlob, epsilon, &g);

    // grid/block dimensions
    dim3 DimGrid(ceil(numThreads * 1.0 / BLOCK_SIZE),1,1);
    dim3 DimBlock(BLOCK_SIZE,1,1);
    bool stopFlag = false;

    std::cout << "Executing kernel..." << std::endl;



    RegionUpdateKernel<<<DimGrid, DimBlock>>>(gPtr, epsilon, numThreadUpdates, devLambdaGlobal, indivLambda, devRunFlag, numThreads);

    gpuErrchk(cudaMemcpy(hostThreadUpdates, numThreadUpdates, sizeof(size_t)*numThreads, cudaMemcpyDeviceToHost));


    cudaMemcpy(lambdaGlob, devLambdaGlobal, sizeof(T)*msgSize, cudaMemcpyDeviceToHost);
    sy.ComputeDualNoSync();
    std::cout << "Kernel Terminated" << std::endl;

    size_t regionUpdates = 0;


    //cudaFree(gPtr);
    cudaFreeHost(lambdaGlob);
    //cudaFree(devRunFlag);
    //cudaFree(indivLambda);
    //cudaFree(devLambdaGlobal);
    //cudaFreeHost(lambdaGlob);
    delete [] hostThreadUpdates;
    cudaStreamDestroy(streamCopy);
    cudaStreamDestroy(streamExec);

    cudaDeviceReset();

    std::cout << "Region updates: " << regionUpdates << std::endl;
    std::cout << "Total regions:  " << g.HostNumberOfRegionsWithParents() << std::endl;

//    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::cout << "Terminating program." << std::endl;
    return 0;
}
