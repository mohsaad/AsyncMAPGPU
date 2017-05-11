
#ifndef __REGION_H_
#define __REGION_H_

#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <thread>
#include <memory>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>

// vectors for lambdaGlobal
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// curand
#include <curand_kernel.h>

#include "CPrecisionTimer.h"


#ifdef _MSC_VER
#define likely(x)		(x)
#define unlikely(x)		(x)
#else
#define likely(x)		__builtin_expect(!!(x), 1)
#define unlikely(x)		__builtin_expect(!!(x), 0)
#endif

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#define BLOCK_SIZE 32
/*
    The MPGraph class. Defines a graph.

*/

template <typename T, typename S>
class MPGraph
{
    public:
        struct PotentialID {
    		S PotID;
    	};
    	struct RegionID {
    		S RegID;
    	};
    	struct DualWorkspaceID {
    		T* DualWorkspace;
    	};
    	struct RRegionWorkspaceID {
    		T* RRegionMem;
    		T* MuMem;
    		S* MuIXMem;
    		S* IXMem;
    	};
    	struct REdgeWorkspaceID {
    		T* MuMem;
    		S* MuIXMem;
    		S* IXMem;
    	};
    	struct GEdgeWorkspaceID {
    		T* mem1;
    		T* mem2;
    	};
    	struct FunctionUpdateWorkspaceID {
    		T* mem;
    	};
    	struct PotentialVector {
    		T* data;
    		S size;
    		PotentialVector(T* d, S s) : data(d), size(s) {};
    	};

    public:
        class Region
        {
            public:
                T c_r;
        		T sum_c_r_c_p;
        		T* pot;
        		S potSize;
        		void* tmp;
        		std::vector<S> varIX;

                Region(T c_r, T* pot, S potSize, const std::vector<S>& varIX) : c_r(c_r), pot(pot), potSize(potSize), tmp(NULL), varIX(varIX)
                {

                };


                virtual ~Region()
                {

                };

                S GetPotentialSize() {
                    return potSize;
                };
        };

        struct MPNode;
    	struct EdgeID;

        struct MsgContainer {
    		size_t lambda;
    		struct MPNode* node;
    		struct EdgeID* edge;
    		std::vector<S> Translator;
    		MsgContainer(size_t l, MPNode* n, EdgeID* e, const std::vector<S>& Trans) : lambda(l), node(n), edge(e), Translator(Trans) {};
    	};

    	struct MPNode : public Region {
    		MPNode(T c_r, const std::vector<S>& varIX, T* pot, S potSize) : Region(c_r, pot, potSize, varIX) {};
    		std::vector<MsgContainer> Parents;
    		std::vector<MsgContainer> Children;
    	};

    	std::vector<S> Cardinalities;
    	std::vector<MPNode*> Graph;
    	std::vector<size_t> ValidRegionMapping;
    	std::vector<PotentialVector> Potentials;
    	size_t LambdaSize;

    	struct EdgeID {
    		MsgContainer* parentPtr;
    		MsgContainer* childPtr;
    		std::vector<S> rStateMultipliers;//cumulative size of parent region variables that overlap with child region (r)
    		std::vector<S> newVarStateMultipliers;//cumulative size of parent region variables that are unique to parent region
    		//std::vector<S> newVarCumSize;//cumulative size of new variables
    		std::vector<S> newVarIX;//variable indices of new variables
    		S newVarSize;
    	};
    	std::vector<EdgeID*> Edges;

	// GPU versions
	class GpuRegion
	{
		public:
			T c_r;
            T sum_c_r_c_p;
			T* pot;
			S potSize;
			void* tmp;
			S* varIX;
            size_t varIXsize;


			GpuRegion(T c_r, T* pot, S potSize, S* varIX, size_t varIXsize) : c_r(c_r), pot(pot), potSize(potSize), tmp(NULL), varIX(varIX), varIXsize(varIXsize)
			{

			};

			virtual ~GpuRegion()
			{

			};

			CUDA_HOSTDEV S GetPotentialSize()
			{
				return potSize;
			};
	};

        //
    struct GpuMPNode;
    struct GpuEdgeID;

    struct GpuMsgContainer {
        size_t lambda;
    	struct GpuMPNode* node;
    	struct GpuEdgeID* edge;
    	S* Translator;
    	GpuMsgContainer(size_t l, GpuMPNode* n, GpuEdgeID* e, S* Trans) : lambda(l), node(n), edge(e), Translator(Trans) {};
    };

    struct GpuMPNode : public GpuRegion {
    		GpuMPNode(T c_r, S* varIX, T* pot, S potSize, size_t varIXsize) : GpuRegion(c_r, pot, potSize, varIX, varIXsize) {};

            GpuMsgContainer* GpuParents;
            GpuMsgContainer* GpuChildren;
            size_t numParents;
            size_t numChildren;


    	};

        // hack to get around the whole "no declared vectors in derive"
        //std::map<GpuMPNode*, std::vector<GpuMsgContainer*>> nodeParents;
        //std::map<GpuMPNode*, std::vector<GpuMsgContainer*>> nodeChildren;

        thrust::host_vector<S> GpuCardinalities;
        thrust::host_vector<GpuMPNode*> GpuGraph;
	    thrust::device_vector<GpuMPNode*> deviceGpuGraph;
        thrust::host_vector<size_t> GpuValidRegionMapping;
        thrust::host_vector<PotentialVector> GpuPotentials;

        // all device pointers
        S* deviceCardinalities;
        size_t numCards;

        GpuMPNode** deviceGraph;
        size_t deviceNodes;

        size_t* deviceValidRegionMapping;
        size_t numValidRegions;

        PotentialVector* devicePotentials;
        size_t numPotentials;

    	struct GpuEdgeID {
    		GpuMsgContainer* parentPtr;
    		GpuMsgContainer* childPtr;
    		S* rStateMultipliers;//cumulative size of parent region variables that overlap with child region (r)
    		size_t rStateMultipliersSize;
		S* newVarStateMultipliers;//cumulative size of parent region variables that are unique to parent region
    		//std::vector<S> newVarCumSize;//cumulative size of new variables
    		S* newVarIX;//variable indices of new variables
            size_t newVarIXsize;
    		S newVarSize;
    	};
    	thrust::host_vector<GpuEdgeID*> GpuEdges;
	    thrust::device_vector<GpuEdgeID*> deviceGpuEdges;
        GpuEdgeID** deviceEdges;
	    size_t numEdges;

        std::map<MPNode*, GpuMPNode*> CpuGpuMap;
        std::map<EdgeID*, GpuEdgeID*> CpuGpuEdgeMap;
        std::map<MsgContainer*, GpuMsgContainer*> CpuGpuMsgMap;


 



        std::map<EdgeID*, size_t> EdgeIDMap;


    public:
        MPGraph();

        virtual ~MPGraph();

        int AddVariables(const std::vector<S>& card);

        const PotentialID AddPotential(const PotentialVector& potVals);

        const RegionID AddRegion(T c_r, const std::vector<S>& varIX, const PotentialID& p);

        int AddConnection(const RegionID& child, const RegionID& parent);

        int AllocateMessageMemory();

        __device__ const DualWorkspaceID AllocateDualWorkspaceMem(T epsilon) const;

        const DualWorkspaceID HostAllocateDualWorkspaceMem(T epsilon) const;

        void HostDeAllocateDualWorkspaceMem(DualWorkspaceID& dw) const;

    	__device__ void DeAllocateDualWorkspaceMem(DualWorkspaceID& dw) const;

    	__device__ T* GetMaxMemComputeMu(T epsilon) const;

        __device__ T* CudaGetMaxMemComputeMu(T epsilon) const;

        __device__ S* GetMaxMemComputeMuIXVar() const;

        __device__ S* CudaGetMaxMemComputeMuIXVar() const;

        __device__ const RRegionWorkspaceID AllocateReparameterizeRegionWorkspaceMem(T epsilon) const;

        __device__ void DeAllocateReparameterizeRegionWorkspaceMem(RRegionWorkspaceID& w) const;

    	__device__ const REdgeWorkspaceID AllocateReparameterizeEdgeWorkspaceMem(T epsilon) const;

        __device__ const REdgeWorkspaceID CudaAllocateReparameterizeEdgeWorkspaceMem(T epsilon) const;

    	__device__ void DeAllocateReparameterizeEdgeWorkspaceMem(REdgeWorkspaceID& w) const;

        void CudaDeAllocateReparameterizeEdgeWorkspaceMem(REdgeWorkspaceID& w) const;

        __device__ const GEdgeWorkspaceID AllocateGradientEdgeWorkspaceMem() const;

    	__device__ void DeAllocateGradientEdgeWorkspaceMem(GEdgeWorkspaceID& w) const;

        __device__ const FunctionUpdateWorkspaceID AllocateFunctionUpdateWorkspaceMem() const;

        __device__ void DeAllocateFunctionUpdateWorkspaceID(FunctionUpdateWorkspaceID& w) const;

        int FillEdge();

    	void ComputeCumulativeSize(MPNode* r_ptr, std::vector<S>& cumVarR);

        int FillTranslator();

        T HostComputeReparameterizationPotential(T* lambdaBase, const MPNode* const r_ptr, const S s_r) const;

    	__device__ size_t NumberOfRegionsTotal() const;

    	__device__ size_t NumberOfRegionsWithParents() const;

        size_t HostNumberOfRegionsWithParents() const;

        __device__ size_t NumberOfEdges() const;

        __device__ void UpdateEdge(T* lambdaBase, T* lambdaGlobal, int e, bool additiveUpdate) const;

        __device__ void CopyLambda(T* lambdaSrc, T* lambdaDst, size_t s_r_e) const;

    	__device__ void CopyMessagesForLocalFunction(T* lambdaSrc, T* lambdaDst, int r) const;

        __device__ void ComputeLocalFunctionUpdate(T* lambdaBase, int r, T epsilon, T multiplier, bool additiveUpdate, FunctionUpdateWorkspaceID& w);

    	__device__ void UpdateLocalFunction(T* lambdaBase, T* lambdaGlobal, int r, bool additiveUpdate);

    	__device__ void CopyMessagesForEdge(T* lambdaSrc, T* lambdaDst, int e) const;

        //void CudaCopyMessagesForEdge(T* lambdaSrc, T* lambdaDst, int e) const ;

        __device__ void CopyMessagesForStar(T* lambdaSrc, T* lambdaDst, int r) const;

        __device__ void ReparameterizeEdge(T* lambdaBase, int e, T epsilon, bool additiveUpdate, REdgeWorkspaceID& wspace) const;

        __device__ T ComputeMu(T* lambdaBase, GpuEdgeID* edge, S* indivVarStates, size_t numVarsOverlap, T epsilon, T* workspaceMem, S* MuIXMem) const;

        __device__ void UpdateRegion(T* lambdaBase, T* lambdaGlobal, int r, bool additiveUpdate);

        __device__ void ReparameterizeRegion(T* lambdaBase, int r, T epsilon, bool additiveUpdate, RRegionWorkspaceID& wspace);

        __device__ T ComputeReparameterizationPotential(T* lambdaBase, const GpuMPNode* const r_ptr, const S s_r) const;

        __device__ T ComputeDual(T* lambdaBase, T epsilon, DualWorkspaceID& dw) const;

        T HostComputeDual(T* lambdaBase, T epsilon, DualWorkspaceID& dw) const;

        __device__ size_t GetLambdaSize() const;

        size_t HostGetLambdaSize() const;

        __device__ void GradientUpdateEdge(T* lambdaBase, int e, T epsilon, T stepSize, bool additiveUpdate, GEdgeWorkspaceID& gew);

        __device__ void ComputeBeliefForRegion(GpuMPNode* r_ptr, T* lambdaBase, T epsilon, T* mem, size_t s_r_e);

        __device__ size_t ComputeBeliefs(T* lambdaBase, T epsilon, T** belPtr, bool OnlyUnaries);

        __device__ void Marginalize(T* curBel, T* oldBel, GpuEdgeID* edge, const S* indivVarStates, size_t indivVarStatesLength, T& marg_new, T& marg_old);

        __device__ T ComputeImprovement(T* curBel, T* oldBel);

    	__device__ void DeleteBeliefs();

        int CopyMessageMemory();
	

        int ResetMessageMemory();

        bool DeallocateGpuGraph();

    private:






        void AllocateNewGPUNode(MPNode* cpuNode, T c_r, const std::vector<S>& varIX, T* pot, S potSize);

        bool DeallocateGpuNode(GpuMPNode* node);



        bool DeallocateGpuEdge(GpuEdgeID* edge);
    

	bool DeallocateGpuContainer(GpuMsgContainer* container);

        void setupDeviceVariables();

        void AllocateMsgContainers();

        void AddGpuConnection(const MPNode* child, const MPNode* parent);




};

template class MPGraph<float, int>;
template class MPGraph<double, int>;

template <class T, class S>
class ThreadSync {
	enum STATE : unsigned char { NONE, INTR, TERM, INIT};
	STATE state;
	int numThreads;
	T* lambdaGlobal;
	T epsilon;
	MPGraph<T, S>* g;
	int currentlyStoppedThreads;
	std::mutex mtx;
	std::condition_variable cond_;
	typename MPGraph<T, S>::DualWorkspaceID dw;
	CPrecisionTimer CTmr, CTmr1;
	T prevDual;
	std::vector<T> LambdaForNoSync;

    public:
        ThreadSync(int nT, T* lambdaGlobal, T epsilon, MPGraph<T, S>* g);

        virtual ~ThreadSync();

        bool checkSync();

        void startTimer();

        double stopTimer();

        bool cudaCheckSync();

        void interruptFunc();

        void terminateFunc();

        void ComputeDualNoSync();

        void CudaComputeDualNoSync();

        bool startFunc();

};

template class ThreadSync<float, int>;
template class ThreadSync<double, int>;

template <class T, class S>
class ThreadWorker {
	size_t cnt;
	ThreadSync<T, S>* ts;
	std::thread* thread_;
	MPGraph<T, S>* g;
	T epsilon;
	int randomSeed;
	T* lambdaGlobal;
	T* lambdaBase;
	size_t msgSize;
	std::vector<T> lambdaLocal;

    #if WHICH_FUNC==1
    	typename MPGraph<T, S>::RRegionWorkspaceID rrw;
    #elif WHICH_FUNC==2
    	typename MPGraph<T, S>::REdgeWorkspaceID rew;
    #elif WHICH_FUNC==3
    	typename MPGraph<T, S>::FunctionUpdateWorkspaceID fw;
    #endif

    std::uniform_int_distribution<int>* uid;
    std::mt19937 eng;
    T* stepsize;

    public:
        ThreadWorker(ThreadSync<T, S>* ts, MPGraph<T, S>* g, T epsilon, int randomSeed, T* lambdaGlobal, T* stepsize);

        ~ThreadWorker();

        void start();

        size_t GetCount();

    private:
        void run();

};

template class ThreadWorker<float, int>;
template class ThreadWorker<double, int>;

template <typename T, typename S>
class AsyncRMPThread {
	std::vector<T> lambdaGlobal;
public:
	int RunMP(MPGraph<T, S>& g, T epsilon, int numIterations, int numThreads, int WaitTimeInMS);

    size_t GetBeliefs(MPGraph<T, S>& g, T epsilon, T** belPtr, bool OnlyUnaries);
};

template class AsyncRMPThread<float, int>;
template class AsyncRMPThread<double, int>;

template <typename T, typename S>
class CudaAsyncRMPThread {
	thrust::host_vector<T> hostLambdaGlobal;
public:

    int CudaRunMP(MPGraph<T, S>& g, T epsilon, int numIterations, int numThreads, int WaitTimeInMS);


    size_t GetBeliefs(MPGraph<T, S>& g, T epsilon, T** belPtr, bool OnlyUnaries);
};

template class CudaAsyncRMPThread<float, int>;
template class CudaAsyncRMPThread<double, int>;

template <typename T, typename S>
class RMP
{
    public:
        RMP();

        virtual ~RMP();

        int RunMP(MPGraph<T, S>& g, T epsilon);
};

template class RMP<float, int>;
template class RMP<double, int>;

__device__ bool checkFlag(volatile bool* flag)
{
    return *flag;
}


template<typename T, typename S>
__global__ void RegionUpdateKernel(MPGraph<T, S>* g, T epsilon, size_t* numThreadUpdates, T* lambdaGlobal, T* lambdaBase, volatile bool* runFlag, int numThreads, size_t* numElementedUpdated)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    //__shared__ float lambdaBase[BLOCK_SIZE + 20];


    if(tx < numThreads)
    {
        int uid;
        curandState_t state;
       // curand_init(clock64(),tx,0,&state);

        // allocate space for edge workspace
        typename MPGraph<T, S>::RRegionWorkspaceID rew;
        rew = g->AllocateReparameterizeRegionWorkspaceMem(epsilon);

        // allocate an array that will act as our base
        size_t msgSize = g->GetLambdaSize();
	    T* devLambdaBase = &lambdaBase[tx*msgSize];
        //T* devLambdaBase = (T*)malloc(msgSize * sizeof(T));
	    //memset(devLambdaBase, T(0), sizeof(T) * msgSize);

        size_t numUpdates = 0;
        int rangeRandNums = g->NumberOfRegionsWithParents();
        while(checkFlag(runFlag))
        {
            for(int i = 0; i < msgSize - 32; i += 32)
            {
                if(!checkFlag(runFlag))
                    break;

	            g->CopyMessagesForStar(lambdaGlobal, devLambdaBase, i + tx);
	            g->ReparameterizeRegion(devLambdaBase, i + tx, epsilon, false, rew);
	            g->UpdateRegion(devLambdaBase, lambdaGlobal, i + tx, false);
                numUpdates++;
                __syncthreads();
            }
            
        }
        numThreadUpdates[tx] = numUpdates;
    
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


    size_t* numElementsUsed;
    gpuErrchk(cudaMalloc((void**)&numElementsUsed, sizeof(size_t)));

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



    RegionUpdateKernel<<<DimGrid, DimBlock, 0, streamExec>>>(gPtr, epsilon, numThreadUpdates, devLambdaGlobal, indivLambda, devRunFlag, numThreads, numElementsUsed);


    for(int i = 0; i < numIterations; i++)
    {
        std::cout << "Iteration: " << i << std::endl;
        gpuErrchk(cudaMemcpyAsync(lambdaGlob, devLambdaGlobal, sizeof(T)*msgSize, cudaMemcpyDeviceToHost, streamCopy));
        cudaStreamSynchronize(streamCopy);
        sy.ComputeDualNoSync();
        std::this_thread::sleep_for(std::chrono::milliseconds(WaitTimeInMS));
    }
    
    gpuErrchk(cudaMemcpyAsync(devRunFlag, &stopFlag, sizeof(bool), cudaMemcpyHostToDevice, streamCopy));

    cudaStreamSynchronize(streamExec);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(hostThreadUpdates, numThreadUpdates, sizeof(size_t)*numThreads, cudaMemcpyDeviceToHost));


    cudaMemcpy(lambdaGlob, devLambdaGlobal, sizeof(T)*msgSize, cudaMemcpyDeviceToHost);
    sy.ComputeDualNoSync();
    std::cout << "Kernel Terminated" << std::endl;

    size_t numElementsUpdated = 0;
    gpuErrchk(cudaMemcpy(&numElementsUpdated, numElementsUsed, sizeof(size_t), cudaMemcpyDeviceToHost));


    size_t regionUpdates = 0;

    for(size_t i = 0; i < numThreads; i++)
    {
        regionUpdates += hostThreadUpdates[i];
    }

    std::cout << "Number of elements used w/ 32 threads " << numElementsUpdated << "out of " << msgSize << std::endl;
    


    //cudaFree(gPtr);
    cudaFreeHost(lambdaGlob);
    //cudaFree(devRunFlag);
    //cudaFree(indivLambda);
    //cudaFree(devLambdaGlobal);
    //cudaFreeHost(lambdaGlob);
    delete [] hostThreadUpdates;
    cudaStreamDestroy(streamCopy);
    cudaStreamDestroy(streamExec);

//    cudaDeviceReset();

    std::cout << "Region updates: " << regionUpdates << std::endl;
    std::cout << "Total regions:  " << g.HostNumberOfRegionsWithParents() << std::endl;


    std::cout << "Terminating program." << std::endl;
    return 0;
}



template<typename T, typename S>
MPGraph<T,S>::MPGraph() : LambdaSize(0)
{

}

template<typename T, typename S>
MPGraph<T,S>::~MPGraph() {
    for (typename std::vector<MPNode*>::iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
        MPNode* r_ptr = *r;
        delete r_ptr;
    }

    for (typename std::vector<EdgeID*>::iterator e = Edges.begin(), e_e = Edges.end(); e != e_e; ++e) {
        EdgeID* e_ptr = *e;
        delete e_ptr;
    }


}


// kind of a function to replace the allocated memories for MPNode
template<typename T, typename S>
void MPGraph<T,S>::AllocateNewGPUNode(MPNode* cpuNode, T c_r, const std::vector<S>& varIX, T* pot, S potSize)
{
    // allocate the node
    GpuMPNode* node;
    gpuErrchk(cudaMalloc((void**)&node, sizeof(GpuMPNode)));

    // malloc a device array of size T*
    T* devicePot;
    gpuErrchk(cudaMalloc((void**)&devicePot, sizeof(T)*potSize));
    gpuErrchk(cudaMemcpy(devicePot, pot, sizeof(T)*potSize, cudaMemcpyHostToDevice));

    // copy over length of varIX
    size_t varIXsize = varIX.size();

    // now copy each individual S value from the vector
    // first malloc an array for it
    S* varIXarr;
    gpuErrchk(cudaMalloc((void**)&varIXarr, sizeof(S) * varIXsize));

    // copy
    gpuErrchk(cudaMemcpy(varIXarr, &(varIX[0]), sizeof(S)*varIXsize, cudaMemcpyHostToDevice));


    // now copy the entire host pointer to device
    GpuMPNode test(c_r, varIXarr, devicePot, potSize, varIXsize);
    gpuErrchk(cudaMemcpy(node, &test, sizeof(test), cudaMemcpyHostToDevice));

    // add to total vector
    GpuGraph.push_back(node);


    CpuGpuMap.insert(std::pair<MPNode*, GpuMPNode*>(cpuNode, node));

    // also add into parents
//nodeParents.insert(std::pair<GpuMPNode*, std::vector*<GpuMsgContainer*>>(node, new std::vector<GpuMsgContainer*>()));
    //nodeChildren.insert(std::pair<GpuMPNode*, std::vector*<GpuMsgContainer*>>(node, new std::vector<GpuMsgContainer*>()));

}

template<typename T, typename S>
bool MPGraph<T,S>::DeallocateGpuGraph()
{
    for(size_t i = 0; i < deviceNodes; i++)
    {
        DeallocateGpuNode(GpuGraph[i]);
    }

    for(size_t i = 0; i < numEdges;i++)
    {
        DeallocateGpuEdge(GpuEdges[i]);
    }

    gpuErrchk(cudaFree(deviceGraph));
    gpuErrchk(cudaFree(deviceEdges));

    gpuErrchk(cudaFree(deviceCardinalities));
    gpuErrchk(cudaFree(deviceValidRegionMapping));

    return true;
}




template<typename T, typename S>
bool MPGraph<T,S>::DeallocateGpuNode(GpuMPNode* node)
{
    GpuMPNode hostNode(0, NULL, NULL, 0, 0);

    gpuErrchk(cudaMemcpy(&hostNode, node, sizeof(hostNode), cudaMemcpyDeviceToHost));
    GpuMsgContainer hostContainer(0, NULL, NULL, NULL);
    for(int i = 0; i < hostNode.numParents; i++)
    {
        gpuErrchk(cudaMemcpy(&hostContainer, hostNode.GpuParents + i, sizeof(hostContainer), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(hostContainer.Translator));

    }
    for(size_t i = 0; i < hostNode.numChildren; i++)
    {
        gpuErrchk(cudaMemcpy(&hostContainer, hostNode.GpuChildren, sizeof(hostContainer), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(hostContainer.Translator));
    }
    gpuErrchk(cudaFree(hostNode.GpuParents));
    gpuErrchk(cudaFree(hostNode.GpuChildren));


    gpuErrchk(cudaFree(hostNode.varIX));
    gpuErrchk(cudaFree(hostNode.pot));
    gpuErrchk(cudaFree(hostNode.tmp));

    gpuErrchk(cudaFree(node));
    return true;
}

template<typename T, typename S>
bool MPGraph<T,S>::DeallocateGpuEdge(GpuEdgeID* edge)
{
    GpuEdgeID hostEdge {NULL, NULL, NULL,0, NULL, NULL, 0};
    gpuErrchk(cudaMemcpy(&hostEdge, edge, sizeof(hostEdge), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(hostEdge.rStateMultipliers));
    gpuErrchk(cudaFree(hostEdge.newVarStateMultipliers));
    gpuErrchk(cudaFree(hostEdge.newVarIX));

    gpuErrchk(cudaFree(edge));
    return true;

}

template<typename T, typename S>
bool MPGraph<T,S>::DeallocateGpuContainer(GpuMsgContainer* container)
{
	return false;
}

template<typename T, typename S>
int MPGraph<T,S>::AddVariables(const std::vector<S>& card) {
    Cardinalities = card;


    return 0;
}

template<typename T, typename S>
const typename MPGraph<T,S>::PotentialID MPGraph<T,S>::AddPotential(const typename MPGraph<T,S>::PotentialVector& potVals) {
    S PotID = S(Potentials.size());
    Potentials.push_back(potVals);

    // also push back on GPU
   // GpuPotentials.push_back(potVals);

    return typename MPGraph<T,S>::PotentialID{ PotID };
}

template<typename T, typename S>
const typename MPGraph<T,S>::RegionID MPGraph<T,S>::AddRegion(T c_r, const std::vector<S>& varIX, const typename MPGraph<T,S>::PotentialID& p) {
    S RegID = S(Graph.size());
    Graph.push_back(new MPNode(c_r, varIX, Potentials[p.PotID].data, Potentials[p.PotID].size));
    //parentChildMap.insert(std::pair<MPNode*, size_t>(Graph[Graph.size() - 1], Graph.size() - 1));


    AllocateNewGPUNode(Graph.back(), c_r, varIX, Potentials[p.PotID].data, Potentials[p.PotID].size);


    // // create a GPU node
    // GpuMPNode* test;
    // gpuAssert(cudaMalloc((void**)&test, sizeof(GpuMPNode));
    // // copy c_r
    // gpuAssert(cudaMemcpy(test->c_r, c_r, sizeof(T), cudaMemcpyHostToDevice));

    // copy varIx


    return typename MPGraph<T,S>::RegionID{ RegID };
}

// template<typename T, typename S>
// void MPGraph<T,S>:AddGpuConnection(const MPNode* child, const MPNode* parent)
// {
//
//
// }

template<typename T, typename S>
int MPGraph<T,S>::AddConnection(const RegionID& child, const RegionID& parent) {
    MPNode* c = Graph[child.RegID];
    MPNode* p = Graph[parent.RegID];

    LambdaSize += c->GetPotentialSize();

    // GpuMsgContainer* cMsgGpu;
    // GpuMsgContainer* pMsgGpu;
    // gpuErrchk(cudaMalloc((void**)&cMsgGpu, sizeof(GpuMsgContainer));
    // gpuErrchk(cudaMalloc((void**)&pMsgGpu, sizeof(GpuMsgContainer));
    //
    // //c->Parents.push_back(MsgContainer{ 0, p, NULL, std::vector<S>() });
    // //p->Children.push_back(MsgContainer{ 0, c, NULL, std::vector<S>() });

    // gpuErrchk(cudaMemcpy(&(cMsgGpu->node), GpuGraph, sizeof)
    //
    //
    // nodeChildren[cGpu]->push_back()
    //
    c->Parents.push_back(MsgContainer(0, p, NULL, std::vector<S>()));
    p->Children.push_back(MsgContainer(0, c, NULL, std::vector<S>()));

    //AddGpuConnection(c, p);

    //GpuMPNode* gpuChild = CpuGpuMap[c];
    //GpuMPNode* gpuParent = CpuGpuMap[p];

    //GpuMsgContainer cCopy(0, gpuParent, NULL, NULL);
    //GpuMsgContainer pCopy(0, gpuChild, NULL, NULL);


   // GpuMsgContainer* CMsg;
    //GpuMsgContainer* PMsg;
    //gpuErrchk(cudaMalloc((void**)&CMsg, sizeof(GpuMsgContainer)));
    //gpuErrchk(cudaMalloc((void**)&PMsg, sizeof(GpuMsgContainer)));


    // copy over stuff to device
   // gpuErrchk(cudaMemcpy(CMsg, &cCopy, sizeof(cCopy), cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpy(PMsg, &pCopy, sizeof(pCopy), cudaMemcpyHostToDevice));

    //nodeParents[gpuChild].push_back(CMsg);
   // nodeChildren[gpuParent].push_back(PMsg);

    // insert a copy here
   // CpuGpuMsgMap.insert(std::pair<MsgContainer*, GpuMsgContainer*>(&(c->Parents.back()), nodeParents[CpuGpuMap[c]].back()));
   // CpuGpuMsgMap.insert(std::pair<MsgContainer*, GpuMsgContainer*>(&(p->Children.back()), nodeChildren[CpuGpuMap[p]].back()));

    return 0;
}


template<typename T, typename S>
int MPGraph<T,S>::ResetMessageMemory()
{
    // first copy over nodes
    // specifically we need to copy over sum_c_r_c_p
    MsgContainer* origContainer;
    GpuMPNode* gpuNode;
    GpuMPNode hostNode(0,NULL,NULL,0,0);

    GpuMsgContainer hostContainer(0, NULL, NULL, NULL);


    for(int i = 0; i < Graph.size(); i++)
    {

        gpuNode = CpuGpuMap[Graph[i]];


        // all we need to do for now is copy over sum_c_r_c_p
        // later, I think we'll allocate everything here
        gpuErrchk(cudaMemcpy(&hostNode, gpuNode, sizeof(hostNode), cudaMemcpyDeviceToHost));





        // parents
        for(int j = 0; j < Graph[i]->Parents.size(); j++)
        {
	    gpuErrchk(cudaMemcpy(&hostContainer, CpuGpuMsgMap[&(Graph[i]->Parents[j])], sizeof(hostContainer), cudaMemcpyDeviceToHost));
	    Graph[i]->Parents[j].lambda = hostContainer.lambda;
        }

        // children
        for(int j = 0; j < Graph[i]->Children.size(); j++)
        {

	    gpuErrchk(cudaMemcpy(&hostContainer, CpuGpuMsgMap[&(Graph[i]->Children[j])], sizeof(hostContainer), cudaMemcpyDeviceToHost));
	    Graph[i]->Children[j].lambda = hostContainer.lambda;
	}


	gpuErrchk(cudaMemcpy(gpuNode, &hostNode, sizeof(hostNode), cudaMemcpyHostToDevice));
    }
/*
    // copy over edges
    GpuEdgeID* gpuEdge;
    GpuEdgeID hostEdge {NULL, NULL, NULL,0, NULL, NULL, 0};
    for(int i = 0; i < Edges.size(); i++)
    {
        gpuEdge = CpuGpuEdgeMap[Edges[i]];
        gpuErrchk(cudaMemcpy(&hostEdge, gpuEdge, sizeof(hostEdge), cudaMemcpyDeviceToHost));

        hostEdge.newVarSize = Edges[i]->newVarSize;
        hostEdge.newVarIXsize = Edges[i]->newVarIX.size();
	hostEdge.childPtr = CpuGpuMsgMap[Edges[i]->childPtr];
	hostEdge.parentPtr = CpuGpuMsgMap[Edges[i]->parentPtr];

        // now, allocate space for the three vectors
        gpuErrchk(cudaMalloc((void**)&(hostEdge.rStateMultipliers), sizeof(S)*Edges[i]->rStateMultipliers.size()));
        gpuErrchk(cudaMemcpy(hostEdge.rStateMultipliers, &(Edges[i]->rStateMultipliers[0]),sizeof(S)*Edges[i]->rStateMultipliers.size(), cudaMemcpyHostToDevice));

	hostEdge.rStateMultipliersSize = Edges[i]->rStateMultipliers.size();

        gpuErrchk(cudaMalloc((void**)&hostEdge.newVarStateMultipliers, sizeof(S)*Edges[i]->newVarStateMultipliers.size()));
        gpuErrchk(cudaMemcpy(hostEdge.newVarStateMultipliers, &(Edges[i]->newVarStateMultipliers[0]),sizeof(S)*Edges[i]->newVarStateMultipliers.size(), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc((void**)&hostEdge.newVarIX, sizeof(S)*Edges[i]->newVarIX.size()));
        gpuErrchk(cudaMemcpy(hostEdge.newVarIX, &(Edges[i]->newVarIX[0]),sizeof(S)*Edges[i]->newVarIX.size(), cudaMemcpyHostToDevice));

        // copy over to GPU
        gpuErrchk(cudaMemcpy(gpuEdge, &hostEdge, sizeof(hostEdge), cudaMemcpyHostToDevice));

    }

    deviceGpuGraph = GpuGraph;
    deviceNodes = GpuGraph.size();
    deviceGpuEdges = GpuEdges;
    numEdges = GpuEdges.size();
    deviceGraph = thrust::raw_pointer_cast(&deviceGpuGraph[0]);
    deviceEdges = thrust::raw_pointer_cast(&deviceGpuEdges[0]);


    // cardinalities
    gpuErrchk(cudaMalloc((void**)&deviceCardinalities, sizeof(S)*Cardinalities.size()));
    gpuErrchk(cudaMemcpy(deviceCardinalities, &Cardinalities[0], Cardinalities.size() * sizeof(S), cudaMemcpyHostToDevice));
    numCards = Cardinalities.size();

    // Potentials
    PotentialVector tmpPot {NULL, 0};
    T* data;
    for(int i = 0; i < Potentials.size(); i++)
    {
        gpuErrchk(cudaMalloc((void**)&data, sizeof(T) * Potentials[i].size));
        gpuErrchk(cudaMemcpy(data, Potentials[i].data, sizeof(T)*Potentials[i].size, cudaMemcpyHostToDevice));
    }

    // Valid region mapping
    gpuErrchk(cudaMalloc((void**)&deviceValidRegionMapping, sizeof(size_t)*GpuValidRegionMapping.size()));
    gpuErrchk(cudaMemcpy(deviceValidRegionMapping, &(GpuValidRegionMapping[0]), sizeof(size_t)*GpuValidRegionMapping.size(), cudaMemcpyHostToDevice));
    numValidRegions = GpuValidRegionMapping.size();

*/
    return 0;

}


template<typename T, typename S>
int MPGraph<T,S>::CopyMessageMemory()
{
    // first copy over nodes
    // specifically we need to copy over sum_c_r_c_p
    MsgContainer* origContainer;
    GpuMPNode* gpuNode;
    GpuMPNode hostNode(0,NULL,NULL,0,0);

    GpuMsgContainer hostContainer(0, NULL, NULL, NULL);


    for(int i = 0; i < Graph.size(); i++)
    {

        gpuNode = CpuGpuMap[Graph[i]];


        // all we need to do for now is copy over sum_c_r_c_p
        // later, I think we'll allocate everything here
        gpuErrchk(cudaMemcpy(&hostNode, gpuNode, sizeof(hostNode), cudaMemcpyDeviceToHost));
        hostNode.sum_c_r_c_p = Graph[i]->sum_c_r_c_p;


	gpuErrchk(cudaMalloc((void**)&(hostNode.GpuParents), sizeof(hostContainer)*Graph[i]->Parents.size()));
	gpuErrchk(cudaMalloc((void**)&(hostNode.GpuChildren), sizeof(hostContainer)*Graph[i]->Children.size()));


        hostNode.numParents = Graph[i]->Parents.size();
        hostNode.numChildren = Graph[i]->Children.size();

        // parents
        for(int j = 0; j < Graph[i]->Parents.size(); j++)
        {
            origContainer = &(Graph[i]->Parents[j]);
	        hostContainer.lambda = origContainer->lambda;
            hostContainer.node = CpuGpuMap[Graph[i]->Parents[j].node];
            hostContainer.edge = CpuGpuEdgeMap[Graph[i]->Parents[j].edge];


            // fill out translator array
            gpuErrchk(cudaMalloc((void**)&(hostContainer.Translator), sizeof(S)* Graph[i]->Parents[j].Translator.size()));
            gpuErrchk(cudaMemcpy(hostContainer.Translator,&(Graph[i]->Parents[j].Translator[0]), sizeof(S)* Graph[i]->Parents[j].Translator.size(), cudaMemcpyHostToDevice));

            // malloc and copy
            gpuErrchk(cudaMemcpy(&(hostNode.GpuParents[j]), &hostContainer, sizeof(hostContainer), cudaMemcpyHostToDevice));
	    //CpuGpuMsgMap.insert(std::pair<MsgContainer*, GpuMsgContainer*>(&(Graph[i]->Parents[j]), &(hostNode.GpuParents[j])));
	    CpuGpuMsgMap[&(Graph[i]->Parents[j])] = &(hostNode.GpuParents[j]);

        }

        // children
        for(int j = 0; j < Graph[i]->Children.size(); j++)
        {
            origContainer = &(Graph[i]->Children[j]);
	    hostContainer.lambda = origContainer->lambda;
	    hostContainer.node = CpuGpuMap[Graph[i]->Children[j].node];
            hostContainer.edge = CpuGpuEdgeMap[Graph[i]->Children[j].edge];

            // fill out translator array
            gpuErrchk(cudaMalloc((void**)&(hostContainer.Translator), sizeof(S)* Graph[i]->Children[j].Translator.size()));
            gpuErrchk(cudaMemcpy(hostContainer.Translator,&(Graph[i]->Children[j].Translator[0]), sizeof(S)* Graph[i]->Children[j].Translator.size(), cudaMemcpyHostToDevice));

            // malloc and copy
            gpuErrchk(cudaMemcpy(&hostNode.GpuChildren[j], &hostContainer, sizeof(hostContainer), cudaMemcpyHostToDevice));


	    CpuGpuMsgMap[&(Graph[i]->Children[j])] = &(hostNode.GpuChildren[j]);
	    //CpuGpuMsgMap.insert(std::pair<MsgContainer*, GpuMsgContainer*>(&(Graph[i]->Children[j]), &(hostNode.GpuChildren[j])));
	}


	gpuErrchk(cudaMemcpy(gpuNode, &hostNode, sizeof(hostNode), cudaMemcpyHostToDevice));
    }

    // copy over edges
    GpuEdgeID* gpuEdge;
    GpuEdgeID hostEdge {NULL, NULL, NULL,0, NULL, NULL, 0};
    for(int i = 0; i < Edges.size(); i++)
    {
        gpuEdge = CpuGpuEdgeMap[Edges[i]];
        gpuErrchk(cudaMemcpy(&hostEdge, gpuEdge, sizeof(hostEdge), cudaMemcpyDeviceToHost));

        hostEdge.newVarSize = Edges[i]->newVarSize;
        hostEdge.newVarIXsize = Edges[i]->newVarIX.size();
	hostEdge.childPtr = CpuGpuMsgMap[Edges[i]->childPtr];
	hostEdge.parentPtr = CpuGpuMsgMap[Edges[i]->parentPtr];

        // now, allocate space for the three vectors
        gpuErrchk(cudaMalloc((void**)&(hostEdge.rStateMultipliers), sizeof(S)*Edges[i]->rStateMultipliers.size()));
        gpuErrchk(cudaMemcpy(hostEdge.rStateMultipliers, &(Edges[i]->rStateMultipliers[0]),sizeof(S)*Edges[i]->rStateMultipliers.size(), cudaMemcpyHostToDevice));

	hostEdge.rStateMultipliersSize = Edges[i]->rStateMultipliers.size();

        gpuErrchk(cudaMalloc((void**)&hostEdge.newVarStateMultipliers, sizeof(S)*Edges[i]->newVarStateMultipliers.size()));
        gpuErrchk(cudaMemcpy(hostEdge.newVarStateMultipliers, &(Edges[i]->newVarStateMultipliers[0]),sizeof(S)*Edges[i]->newVarStateMultipliers.size(), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc((void**)&hostEdge.newVarIX, sizeof(S)*Edges[i]->newVarIX.size()));
        gpuErrchk(cudaMemcpy(hostEdge.newVarIX, &(Edges[i]->newVarIX[0]),sizeof(S)*Edges[i]->newVarIX.size(), cudaMemcpyHostToDevice));

        // copy over to GPU
        gpuErrchk(cudaMemcpy(gpuEdge, &hostEdge, sizeof(hostEdge), cudaMemcpyHostToDevice));

    }

    deviceGpuGraph = GpuGraph;
    deviceNodes = GpuGraph.size();
    deviceGpuEdges = GpuEdges;
    numEdges = GpuEdges.size();
    deviceGraph = thrust::raw_pointer_cast(&deviceGpuGraph[0]);
    deviceEdges = thrust::raw_pointer_cast(&deviceGpuEdges[0]);


    // cardinalities
    gpuErrchk(cudaMalloc((void**)&deviceCardinalities, sizeof(S)*Cardinalities.size()));
    gpuErrchk(cudaMemcpy(deviceCardinalities, &Cardinalities[0], Cardinalities.size() * sizeof(S), cudaMemcpyHostToDevice));
    numCards = Cardinalities.size();

    // Potentials
    PotentialVector tmpPot {NULL, 0};
    T* data;
    for(int i = 0; i < Potentials.size(); i++)
    {
        gpuErrchk(cudaMalloc((void**)&data, sizeof(T) * Potentials[i].size));
        gpuErrchk(cudaMemcpy(data, Potentials[i].data, sizeof(T)*Potentials[i].size, cudaMemcpyHostToDevice));
    }

    // Valid region mapping
    gpuErrchk(cudaMalloc((void**)&deviceValidRegionMapping, sizeof(size_t)*GpuValidRegionMapping.size()));
    gpuErrchk(cudaMemcpy(deviceValidRegionMapping, &(GpuValidRegionMapping[0]), sizeof(size_t)*GpuValidRegionMapping.size(), cudaMemcpyHostToDevice));
    numValidRegions = GpuValidRegionMapping.size();


    return 0;

}

template<typename T, typename S>
int MPGraph<T,S>::AllocateMessageMemory() {

    size_t lambdaOffset = 0;

    GpuEdgeID* gpuID = NULL;
    GpuEdgeID testID{NULL, NULL, NULL, 0, NULL, NULL, 0};

    for (typename std::vector<MPNode*>::iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
        MPNode* r_ptr = *r;

        size_t numEl = r_ptr->GetPotentialSize();
        if (r_ptr->Parents.size() > 0) {
            ValidRegionMapping.push_back(r - Graph.begin());
            GpuValidRegionMapping.push_back(r - Graph.begin());
        }

        for (typename std::vector<MsgContainer>::iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn) {
            MPNode* pn_ptr = pn->node;

            pn->lambda = lambdaOffset;
            for (typename std::vector<MsgContainer>::iterator cn = pn_ptr->Children.begin(), cn_e = pn_ptr->Children.end(); cn != cn_e; ++cn) {
                if (cn->node == r_ptr) {
                    cn->lambda = lambdaOffset;
                    Edges.push_back(new EdgeID{ &(*pn), &(*cn), std::vector<S>(), std::vector<S>(), std::vector<S>(), 0 });
                    pn->edge = cn->edge = Edges.back();

                    gpuErrchk(cudaMalloc((void**)&gpuID, sizeof(testID)));
                    //testID.parentPtr = CpuGpuMsgMap[&(*cn)];
                    //testID.childPtr = CpuGpuMsgMap[&(*pn)];

                    gpuErrchk(cudaMemcpy(gpuID, &testID, sizeof(testID), cudaMemcpyHostToDevice));

                    CpuGpuEdgeMap.insert(std::pair<EdgeID*, GpuEdgeID*>(Edges.back(), gpuID));
                    GpuEdges.push_back(gpuID);
		    gpuID = NULL;

                    break;
                }
            }
            lambdaOffset += numEl;
        }
    }
    FillTranslator();
    for (typename std::vector<MPNode*>::iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
        MPNode* r_ptr = *r;
        T sum_c_p = r_ptr->c_r;
        for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
            sum_c_p += p->node->c_r;
        }
        r_ptr->sum_c_r_c_p = sum_c_p;
    }
    return 0;
}

template<typename T, typename S>
int MPGraph<T,S>::FillEdge() {
    for (typename std::vector<EdgeID*>::iterator e = Edges.begin(), e_e = Edges.end(); e != e_e; ++e) {
        MPNode* r_ptr = (*e)->childPtr->node;
        MPNode* p_ptr = (*e)->parentPtr->node;

        (*e)->newVarSize = 1;
        (*e)->rStateMultipliers.assign(r_ptr->varIX.size(), 1);
        for (typename std::vector<S>::iterator vp = p_ptr->varIX.begin(), vp_e = p_ptr->varIX.end(); vp != vp_e; ++vp) {
            typename std::vector<S>::iterator vr = std::find(r_ptr->varIX.begin(), r_ptr->varIX.end(), *vp);
            size_t posP = vp - p_ptr->varIX.begin();
            if (vr == r_ptr->varIX.end()) {
                (*e)->newVarIX.push_back(*vp);

    (*e)->newVarStateMultipliers.push_back(((std::vector<S>*)p_ptr->tmp)->at(posP));
                (*e)->newVarSize *= Cardinalities[*vp];
            } else {
                size_t posR = vr - r_ptr->varIX.begin();
                (*e)->rStateMultipliers[posR] = ((std::vector<S>*)p_ptr->tmp)->at(posP);
            }
        }

    }
    return 0;
}

template<typename T, typename S>
void MPGraph<T,S>::ComputeCumulativeSize(MPNode* r_ptr, std::vector<S>& cumVarR) {
    size_t numVars = r_ptr->varIX.size();
    cumVarR.assign(numVars, 1);
    for (size_t v = 1; v < numVars; ++v) {
        cumVarR[v] = cumVarR[v - 1] * Cardinalities[r_ptr->varIX[v]];
    }
}

template<typename T, typename S>
int MPGraph<T,S>::FillTranslator() {
    for (typename std::vector<MPNode*>::iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
        MPNode* r_ptr = *r;
        std::vector<S>* cumVarR = new std::vector<S>;
        ComputeCumulativeSize(r_ptr, *cumVarR);
        r_ptr->tmp = (void*)cumVarR;
    }
    FillEdge();

    for (typename std::vector<MPNode*>::iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
        MPNode* r_ptr = *r;
        size_t numEl = r_ptr->GetPotentialSize();

        for (typename std::vector<MsgContainer>::iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn) {
            MPNode* c_ptr = cn->node;
            size_t numVarsC = c_ptr->varIX.size();

            cn->Translator.assign(numEl, S(0));

            for (size_t s_r = 0; s_r < numEl; ++s_r) {
                S val = 0;
                for (size_t cIX = 0; cIX < numVarsC; ++cIX) {
                    S curVar = c_ptr->varIX[cIX];
                    typename std::vector<S>::iterator iter = std::find(r_ptr->varIX.begin(), r_ptr->varIX.end(), curVar);
                    size_t pos = iter - r_ptr->varIX.begin();
                    assert(iter != r_ptr->varIX.end());
                    S curState = (s_r / (((std::vector<S>*)r_ptr->tmp)->at(pos))) % Cardinalities[r_ptr->varIX[pos]];
                    val += curState*((std::vector<S>*)c_ptr->tmp)->at(cIX);
                }

                cn->Translator[s_r] = val;
            }
        }
    }

    for (typename std::vector<MPNode*>::iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
        MPNode* r_ptr = *r;
        std::vector<S>* cumVarR = (std::vector<S>*)r_ptr->tmp;
        delete cumVarR;
        r_ptr->tmp = NULL;
    }
    return 0;
}


template<typename T, typename S>
__device__ const typename MPGraph<T,S>::DualWorkspaceID MPGraph<T,S>::AllocateDualWorkspaceMem(T epsilon) const {
    size_t maxMem = 0;
    // for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
    for(int i = 0; i < deviceNodes; i++)
    {
        GpuMPNode* r_ptr = deviceGraph[i];
        T ecr = epsilon*(r_ptr->c_r);
        size_t s_r_e = r_ptr->GetPotentialSize();

        if (ecr != T(0)) {
            maxMem = (s_r_e > maxMem) ? s_r_e : maxMem;
        }
    }

    return DualWorkspaceID{((maxMem > 0) ? new T[maxMem] : NULL)};
}

template<typename T, typename S>
const typename MPGraph<T,S>::DualWorkspaceID MPGraph<T,S>::HostAllocateDualWorkspaceMem(T epsilon) const {
    size_t maxMem = 0;
    for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
        MPNode* r_ptr = *r;
        T ecr = epsilon*r_ptr->c_r;
        size_t s_r_e = r_ptr->GetPotentialSize();

        if (ecr != T(0)) {
            maxMem = (s_r_e > maxMem) ? s_r_e : maxMem;
        }
    }

    return DualWorkspaceID{ ((maxMem > 0) ? new T[maxMem] : NULL) };
}

template<typename T, typename S>
void MPGraph<T,S>::HostDeAllocateDualWorkspaceMem(DualWorkspaceID& dw) const {
    delete[] dw.DualWorkspace;
}

template<typename T, typename S>
size_t MPGraph<T,S>::HostNumberOfRegionsWithParents() const {
    return ValidRegionMapping.size();
}

template<typename T, typename S>
T MPGraph<T,S>::HostComputeDual(T* lambdaBase, T epsilon, DualWorkspaceID& dw) const {
    T dual = T(0);

    for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
        MPNode* r_ptr = *r;
        T ecr = epsilon*r_ptr->c_r;
        size_t s_r_e = r_ptr->GetPotentialSize();

        T* mem = dw.DualWorkspace;

        T maxVal = -std::numeric_limits<T>::max();
        for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
            T potVal = HostComputeReparameterizationPotential(lambdaBase, r_ptr, s_r);

            if (ecr != T(0)) {
                potVal /= ecr;
                mem[s_r] = potVal;
            }

            maxVal = ((potVal > maxVal) ? potVal : maxVal);
        }

        if (ecr != T(0)) {
            T sum = std::exp(mem[0] - maxVal);
            for (size_t s_r = 1; s_r != s_r_e; ++s_r) {
                sum += std::exp(mem[s_r] - maxVal);
            }
            dual += ecr*(maxVal + std::log(sum + T(1e-20)));
        } else {
            dual += maxVal;
        }
    }
    return dual;
}


template<typename T, typename S>
__device__ void MPGraph<T,S>::DeAllocateDualWorkspaceMem(DualWorkspaceID& dw) const {
    delete[] dw.DualWorkspace;
}

template<typename T, typename S>
__device__ T* MPGraph<T,S>::GetMaxMemComputeMu(T epsilon) const {
    size_t maxMem = 0;
    // for (typename std::vector<EdgeID*>::const_iterator e = Edges.begin(), e_e = Edges.end(); e != e_e; ++e) {
    for(int i = 0; i < numEdges; i++)
    {
        GpuEdgeID* edge = deviceEdges[i];
        size_t s_p_e = edge->newVarSize;
        GpuMPNode* p_ptr = edge->parentPtr->node;

        T ecp = epsilon*p_ptr->c_r;
        if (ecp != T(0)) {
            maxMem = (s_p_e > maxMem) ? s_p_e : maxMem;
        }
    }
    return ((maxMem > 0) ? new T[maxMem] : NULL);
}


template<typename T, typename S>
__device__ S* MPGraph<T,S>::GetMaxMemComputeMuIXVar() const {
    size_t maxMem = 0;
    GpuMsgContainer* p;
    for (size_t r = 0; r < numValidRegions; ++r) {
        GpuMPNode* r_ptr = deviceGraph[deviceValidRegionMapping[r]];
        // for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p)
        for(size_t i = 0; i < r_ptr->numParents; i++)
        {
            p = &(r_ptr->GpuParents[i]);
            size_t tmp = p->edge->newVarIXsize;
            maxMem = (tmp>maxMem) ? tmp : maxMem;
        }
    }

    return ((maxMem > 0) ? new S[maxMem] : NULL);
}




template<typename T, typename S>
__device__ const typename MPGraph<T,S>::RRegionWorkspaceID MPGraph<T,S>::AllocateReparameterizeRegionWorkspaceMem(T epsilon) const {
    size_t maxMem = 0;
    size_t maxIXMem = 0;
    for (size_t r = 0; r < numValidRegions; ++r) {
        GpuMPNode* r_ptr = deviceGraph[deviceValidRegionMapping[r]];
        size_t psz = r_ptr->numParents;
        maxMem = (psz>maxMem) ? psz : maxMem;
        size_t rvIX = r_ptr->varIXsize;
        maxIXMem = (rvIX>maxIXMem) ? rvIX : maxIXMem;
    }

    return RRegionWorkspaceID{ ((maxMem > 0) ? new T[maxMem] : NULL), GetMaxMemComputeMu(epsilon), GetMaxMemComputeMuIXVar(), ((maxIXMem > 0) ? new S[maxIXMem] : NULL) };
}



template<typename T, typename S>
__device__ void MPGraph<T,S>::DeAllocateReparameterizeRegionWorkspaceMem(RRegionWorkspaceID& w) const {
    delete[] w.RRegionMem;
    delete[] w.MuMem;
    delete[] w.MuIXMem;
    delete[] w.IXMem;
    w.RRegionMem = NULL;
    w.MuMem = NULL;
    w.MuIXMem = NULL;
    w.IXMem = NULL;
}

template<typename T, typename S>
__device__ const typename MPGraph<T,S>::REdgeWorkspaceID MPGraph<T,S>::AllocateReparameterizeEdgeWorkspaceMem(T epsilon) const {
    size_t maxIXMem = 0;
    // for(typename std::vector<EdgeID*>::const_iterator eb=Edges.begin(), eb_e=Edges.end();eb!=eb_e;++eb)
    for(int i = 0; i < numEdges; i++)
    {
        GpuMPNode* r_ptr = deviceEdges[i]->childPtr->node;
        size_t rNumVar = r_ptr->varIXsize;
        maxIXMem = (rNumVar>maxIXMem) ? rNumVar : maxIXMem;
 }
    return REdgeWorkspaceID{ GetMaxMemComputeMu(epsilon), GetMaxMemComputeMuIXVar(), ((maxIXMem > 0) ? new S[maxIXMem] : NULL) };
}



template<typename T, typename S>
__device__ void MPGraph<T,S>::DeAllocateReparameterizeEdgeWorkspaceMem(REdgeWorkspaceID& w) const {
    delete[] w.MuMem;
    delete[] w.MuIXMem;
    delete[] w.IXMem;
    w.MuMem = NULL;
    w.MuIXMem = NULL;
    w.IXMem = NULL;
}



template<typename T, typename S>
__device__ const typename MPGraph<T,S>::GEdgeWorkspaceID MPGraph<T,S>::AllocateGradientEdgeWorkspaceMem() const {
    size_t memSZ = 0;
    // for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r)
    for(size_t i = 0; i < deviceNodes; i++)
    {
        GpuMPNode* r_ptr = deviceGraph[i];
        size_t s_r_e = r_ptr->GetPotentialSize();
        memSZ = (s_r_e > memSZ) ? s_r_e : memSZ;
    }
    return GEdgeWorkspaceID{ ((memSZ>0) ? new T[memSZ] : NULL), ((memSZ>0) ? new T[memSZ] : NULL) };
}

template<typename T, typename S>
__device__ void MPGraph<T,S>::DeAllocateGradientEdgeWorkspaceMem(GEdgeWorkspaceID& w) const {
    delete[] w.mem1;
    delete[] w.mem2;
}

template<typename T, typename S>
__device__ const typename MPGraph<T,S>::FunctionUpdateWorkspaceID MPGraph<T,S>::AllocateFunctionUpdateWorkspaceMem() const {
    size_t memSZ = 0;
    // for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r)
    for(int i = 0; i < deviceNodes; i++)
    {
        GpuMPNode* r_ptr = deviceGraph[i];
        size_t s_r_e = r_ptr->GetPotentialSize();
        memSZ = (s_r_e > memSZ) ? s_r_e : memSZ;
    }
    return FunctionUpdateWorkspaceID{ ((memSZ>0) ? new T[memSZ] : NULL)};
}

template<typename T, typename S>
__device__ void MPGraph<T,S>::DeAllocateFunctionUpdateWorkspaceID(FunctionUpdateWorkspaceID& w) const {
    delete[] w.mem;
}







template<typename T, typename S>
__device__ size_t MPGraph<T,S>::NumberOfRegionsTotal() const {
    return deviceNodes;
}

template<typename T, typename S>
__device__ size_t MPGraph<T,S>::NumberOfRegionsWithParents() const {
    return numValidRegions;
}

template<typename T, typename S>
__device__ size_t MPGraph<T,S>::NumberOfEdges() const {
    return numEdges;
}

template<typename T, typename S>
__device__ void MPGraph<T,S>::UpdateEdge(T* lambdaBase, T* lambdaGlobal, int e, bool additiveUpdate) const {
    if (lambdaBase == lambdaGlobal) {
        assert(additiveUpdate == false);//change ReparameterizationEdge function to directly perform update and don't call UpdateEdge
        return;
    }

    GpuEdgeID* edge = deviceEdges[e];
    GpuMPNode* r_ptr = edge->childPtr->node;
    size_t s_r_e = r_ptr->GetPotentialSize();

    for (size_t s_r = 0; s_r < s_r_e; ++s_r) {
        if (additiveUpdate) {
            lambdaGlobal[edge->childPtr->lambda + s_r] += lambdaBase[edge->childPtr->lambda + s_r];
        } else {
            lambdaGlobal[edge->childPtr->lambda + s_r] = lambdaBase[edge->childPtr->lambda + s_r];
        }
    }
}

template<typename T, typename S>
__device__ void MPGraph<T,S>::CopyLambda(T* lambdaSrc, T* lambdaDst, size_t s_r_e) const {
    //std::copy(lambdaSrc, lambdaSrc + s_r_e, lambdaDst);
    //memcpy((void*)(lambdaDst), (void*)(lambdaSrc), s_r_e*sizeof(T));
    for (T* ptr_e = lambdaSrc + s_r_e; ptr_e != lambdaSrc;) {
        *lambdaDst++ = *lambdaSrc++;
    }
}



template<typename T, typename S>
__device__ void MPGraph<T,S>::CopyMessagesForLocalFunction(T* lambdaSrc, T* lambdaDst, int r) const {
    GpuMPNode* r_ptr = deviceGraph[r];
    size_t s_r_e = r_ptr->GetPotentialSize();

    GpuMsgContainer* cn;
    GpuMsgContainer* pn;

    // for (typename std::vector<MsgContainer>::const_iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn) {
    for(size_t i = 0; i < r_ptr->numParents; i++)
    {
        pn = &(r_ptr->GpuParents[i]);
        CopyLambda(lambdaSrc + pn->lambda, lambdaDst + pn->lambda, s_r_e);
    }

    // for (typename std::vector<MsgContainer>::const_iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn) {
    for(size_t i = 0; i < r_ptr->numChildren; i++)
    {
        cn = &(r_ptr->GpuChildren[i]);
        size_t s_r_c = cn->node->GetPotentialSize();
        CopyLambda(lambdaSrc + cn->lambda, lambdaDst + cn->lambda, s_r_c);
    }
}
template<typename T, typename S>
__device__ void MPGraph<T,S>::ComputeLocalFunctionUpdate(T* lambdaBase, int r, T epsilon, T multiplier, bool additiveUpdate, FunctionUpdateWorkspaceID& w) {
    assert(additiveUpdate==true);
    GpuMPNode* r_ptr = deviceGraph[r];
    size_t s_r_e = r_ptr->GetPotentialSize();

    T c_r = r_ptr->c_r;
    T frac = multiplier;
    if(epsilon>0) {
            frac /= (epsilon*c_r);
    }

    T* mem = w.mem;
    ComputeBeliefForRegion(r_ptr, lambdaBase, epsilon, mem, s_r_e);
    for(size_t s_r=0;s_r<s_r_e;++s_r) {
        mem[s_r] *= frac;
    }

    // for (typename std::vector<MsgContainer>::iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
    for(size_t i = 0; i < r_ptr->numParents; i++)
    {
        for (size_t s_r = 0; s_r < s_r_e; ++s_r) {
            lambdaBase[r_ptr->GpuParents[i].lambda+s_r] = -mem[s_r];
        }
    }

    // for (typename std::vector<MsgContainer>::const_iterator c = r_ptr->Children.begin(), c_e = r_ptr->Children.end(); c != c_e; ++c)
    GpuMsgContainer* c;
    for(size_t i = 0; i < r_ptr->numChildren; i++)
    {
        c = &(r_ptr->GpuChildren[i]);
        size_t s_r_c = c->node->GetPotentialSize();
        for(size_t i = 0; i < s_r_c; i++)
        {
            lambdaBase[c->lambda + i] = T(0);
        }
        // std::fill_n(lambdaBase+c->lambda, s_r_c, T(0));
        for(size_t s_r = 0;s_r < s_r_e;++s_r) {
            lambdaBase[c->lambda+c->Translator[s_r]] += mem[s_r];
        }
    }
}

template<typename T, typename S>
__device__ void MPGraph<T,S>::UpdateLocalFunction(T* lambdaBase, T* lambdaGlobal, int r, bool additiveUpdate) {
    if (lambdaBase == lambdaGlobal) {
        return;
    }
    GpuMPNode* r_ptr = deviceGraph[r];
    size_t s_r_e = r_ptr->GetPotentialSize();

    GpuMsgContainer* p;
    GpuMsgContainer* cn;
    if (additiveUpdate) {
        // for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p)
        for(size_t i = 0; i < r_ptr->numParents; i++)
        {
            p = &(r_ptr->GpuParents[i]);
            for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
                lambdaGlobal[p->lambda + s_r] += lambdaBase[p->lambda + s_r];
            }
        }

        // for (typename std::vector<MsgContainer>::const_iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn)
        for(size_t i = 0; i < r_ptr->numChildren; i++)
        {
            cn = &(r_ptr->GpuChildren[i]);
            size_t s_r_c = cn->node->GetPotentialSize();
            for(size_t s_c = 0; s_c != s_r_c; ++s_c) {
                lambdaGlobal[cn->lambda + s_c] += lambdaBase[cn->lambda + s_c];
            }
        }
    } else {
        assert(false);
    }
}

// so this function runs in a single thread, meaning it's copying to each individual thread
// so instead  what we're going to do is copy from global memory
//

template<typename T, typename S>
__device__  void MPGraph<T,S>::CopyMessagesForEdge(T* lambdaSrc, T* lambdaDst, int e) const {
    GpuEdgeID* edge = deviceEdges[e];
    GpuMPNode* r_ptr = edge->childPtr->node;
    GpuMPNode* p_ptr = edge->parentPtr->node;

    size_t s_r_e = r_ptr->GetPotentialSize();
    size_t s_p_e = p_ptr->GetPotentialSize();

    // for (typename std::vector<MsgContainer>::const_iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn)
    GpuMsgContainer* pn;
    GpuMsgContainer* cn;
    GpuMsgContainer* p_hat;
    GpuMsgContainer* c_hat;
    for(size_t i = 0; i < r_ptr->numParents; i++)
    {
        pn = &(r_ptr->GpuParents[i]);
        CopyLambda(lambdaSrc + pn->lambda, lambdaDst + pn->lambda, s_r_e);
    }

    // for (typename std::vector<MsgContainer>::const_iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn)
    for(size_t i = 0; i < r_ptr->numChildren; i++)
    {
        cn = &(r_ptr->GpuChildren[i]);
        size_t s_r_c = cn->node->GetPotentialSize();
        CopyLambda(lambdaSrc + cn->lambda, lambdaDst + cn->lambda, s_r_c);
    }

    // for (typename std::vector<MsgContainer>::const_iterator p_hat = p_ptr->Parents.begin(), p_hat_e = p_ptr->Parents.end(); p_hat != p_hat_e; ++p_hat)
    for(size_t i = 0; i < p_ptr->numParents; i++)
    {
        p_hat = &(p_ptr->GpuParents[i]);
        CopyLambda(lambdaSrc + p_hat->lambda, lambdaDst + p_hat->lambda, s_p_e);
    }

    // for (typename std::vector<MsgContainer>::const_iterator c_hat = p_ptr->Children.begin(), c_hat_e = p_ptr->Children.end(); c_hat != c_hat_e; ++c_hat)
    for(size_t i = 0; i < p_ptr->numChildren; i++)
    {
        c_hat = &(p_ptr->GpuChildren[i]);
        if (c_hat->node != r_ptr) {
            size_t s_r_pc = c_hat->node->GetPotentialSize();
            CopyLambda(lambdaSrc + c_hat->lambda, lambdaDst + c_hat->lambda, s_r_pc);
        }
    }
}


template<typename T, typename S>
__device__  void MPGraph<T,S>::CopyMessagesForStar(T* lambdaSrc, T* lambdaDst, int r) const {
    GpuMPNode* r_ptr = deviceGraph[deviceValidRegionMapping[r]];
    size_t s_r_e = r_ptr->GetPotentialSize();

    GpuMsgContainer* pn;
    GpuMsgContainer* cn;
    GpuMsgContainer* p_hat;
    GpuMsgContainer* c_hat;
    // for (typename std::vector<MsgContainer>::const_iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn)
    for(size_t i = 0; i < r_ptr->numParents; i++)
    {
        pn = &(r_ptr->GpuParents[i]);
        CopyLambda(lambdaSrc + pn->lambda, lambdaDst + pn->lambda, s_r_e);

        GpuMPNode* p_ptr = pn->node;
        size_t s_p_e = p_ptr->GetPotentialSize();
        // for (typename std::vector<MsgContainer>::const_iterator p_hat = p_ptr->Parents.begin(), p_hat_e = p_ptr->Parents.end(); p_hat != p_hat_e; ++p_hat)
        for(size_t j = 0; j < p_ptr->numParents; j++)
        {
            p_hat = &(p_ptr->GpuParents[j]);
            CopyLambda(lambdaSrc + p_hat->lambda, lambdaDst + p_hat->lambda, s_p_e);
        }

        // for (typename std::vector<MsgContainer>::const_iterator c_hat = p_ptr->Children.begin(), c_hat_e = p_ptr->Children.end(); c_hat != c_hat_e; ++c_hat)
        for(size_t j = 0; j < p_ptr->numChildren; j++)
        {
            c_hat = &(p_ptr->GpuChildren[j]);
            if (c_hat->node != r_ptr) {
                size_t s_pc_e = c_hat->node->GetPotentialSize();
                CopyLambda(lambdaSrc + c_hat->lambda, lambdaDst + c_hat->lambda, s_pc_e);
            }
        }
    }

    // for (typename std::vector<MsgContainer>::const_iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn)
    for(size_t i = 0; i < r_ptr->numChildren; i++)
    {
        cn = &(r_ptr->GpuChildren[i]);
        size_t s_r_c = cn->node->GetPotentialSize();
        CopyLambda(lambdaSrc + cn->lambda, lambdaDst + cn->lambda, s_r_c);
    }
}

template<typename T, typename S>
__device__  void MPGraph<T,S>::ReparameterizeEdge(T* lambdaBase, int e, T epsilon, bool additiveUpdate, REdgeWorkspaceID& wspace) const {
    GpuEdgeID* edge = deviceEdges[e];
    GpuMPNode* r_ptr = edge->childPtr->node;
    GpuMPNode* p_ptr = edge->parentPtr->node;

    size_t s_r_e = r_ptr->GetPotentialSize();

    T c_p = p_ptr->c_r;
    T c_r = r_ptr->c_r;
    T frac = T(1) / (c_p + c_r);

// OOB errpor rght here
    size_t rNumVar = r_ptr->varIXsize;
    //std::vector<S> indivVarStates(rNumVar, 0);
    S* indivVarStates = wspace.IXMem;
    for(S *tmp=indivVarStates, *tmp_e=indivVarStates+rNumVar;tmp!=tmp_e;++tmp) {
        *tmp = 0;
    }
    for (size_t s_r = 0; s_r < s_r_e; ++s_r) {
        if (additiveUpdate) {//additive update
            T updateVal1 = ComputeReparameterizationPotential(lambdaBase, r_ptr, s_r);
            T updateVal2 = lambdaBase[edge->childPtr->lambda + s_r] + ComputeMu(lambdaBase, edge, indivVarStates, rNumVar, epsilon, wspace.MuMem, wspace.MuIXMem);
            //lambdaBase[edge->childPtr->lambda+s_r] += (c_p*frac*updateVal1 - c_r*frac*updateVal2);//use this line if global memory is equal to local memory
            lambdaBase[edge->childPtr->lambda + s_r] = (c_p*frac*updateVal1 - c_r*frac*updateVal2);//addition will be performed in UpdateEdge function
        } else {//absolute update
            T updateVal1 = ComputeReparameterizationPotential(lambdaBase, r_ptr, s_r) + lambdaBase[edge->childPtr->lambda+s_r];
            T updateVal2 = ComputeMu(lambdaBase, edge, indivVarStates, rNumVar, epsilon, wspace.MuMem, wspace.MuIXMem);
            lambdaBase[edge->childPtr->lambda + s_r] = (c_p*frac*updateVal1 - c_r*frac*updateVal2);
        }

        for (size_t varIX = 0; varIX < rNumVar; ++varIX) {
            ++indivVarStates[varIX];
            if (indivVarStates[varIX] == deviceCardinalities[r_ptr->varIX[varIX]]) {
                indivVarStates[varIX] = 0;
            } else {
                break;
            }
        }
    }
}


template<typename T, typename S>
__device__ T MPGraph<T,S>::ComputeMu(T* lambdaBase, GpuEdgeID* edge, S* indivVarStates, volatile size_t numVarsOverlap, T epsilon, T* workspaceMem, S* MuIXMem) const {
    GpuMPNode* r_ptr = edge->childPtr->node;
    GpuMPNode* p_ptr = edge->parentPtr->node;

   // numVarsOverlap = min(numVarsOverlap, edge->rStateMultipliersSize);
    size_t s_p_stat = 0;
    for (size_t k = 0; k<numVarsOverlap; ++k) {
        s_p_stat += indivVarStates[k]*edge->rStateMultipliers[k];
    }

    //size_t s_p_e = edge->newVarCumSize.back();
    size_t s_p_e = edge->newVarSize;
    size_t numVarNew = edge->newVarIXsize;

    T maxval = -std::numeric_limits<T>::max();
    T ecp = epsilon*p_ptr->c_r;
    /*T* mem = NULL;
    if (ecp != T(0)) {
        mem = new T[s_p_e];
    }*/
    T* mem = workspaceMem;

    //individual vars;
    //std::vector<S> indivNewVarStates(numVarNew, 0);
    for(S *tmpmem=MuIXMem, *tmpmem_e=MuIXMem+numVarNew;tmpmem!=tmpmem_e;++tmpmem) {
        *tmpmem = 0;
    }
    S* indivNewVarStates = MuIXMem;
    size_t s_p_real = s_p_stat;


    GpuMsgContainer* p_hat;
    GpuMsgContainer* c_hat;
    for (size_t s_p = 0; s_p != s_p_e; ++s_p) {
        //size_t s_p_real = s_p_stat;
        //for (size_t varIX = 0; varIX<numVarNew; ++varIX) {
        //	s_p_real += indivNewVarStates[varIX]*edge->newVarStateMultipliers[varIX];
        //}

	// TOTALLY A HACK TO GET THIS VALUE
        T buf = ((p_ptr)->pot == NULL) ? T(0) : (p_ptr)->pot[s_p_real];

        // for (typename std::vector<MsgContainer>::const_iterator p_hat = p_ptr->Parents.begin(), p_hat_e = p_ptr->Parents.end(); p_hat != p_hat_e; ++p_hat)
        for(size_t i = 0; i < p_ptr->numParents; i++)
        {
            p_hat = &(p_ptr->GpuParents[i]);
            buf -= lambdaBase[p_hat->lambda+s_p_real];
        }

        // for (typename std::vector<MsgContainer>::const_iterator c_hat = p_ptr->Children.begin(), c_hat_e = p_ptr->Children.end(); c_hat != c_hat_e; ++c_hat)
        for(size_t i = 0; i < p_ptr->numChildren; i++)
        {
            c_hat = &(p_ptr->GpuChildren[i]);
            if (c_hat->node != r_ptr) {
                buf += lambdaBase[c_hat->lambda+c_hat->Translator[s_p_real]];
            }
        }

        if (ecp != T(0)) {
            buf /= ecp;
            mem[s_p] = buf;
        }
        maxval = (buf>maxval) ? buf : maxval;

        for (size_t varIX = 0; varIX < numVarNew; ++varIX) {
            ++indivNewVarStates[varIX];
            if (indivNewVarStates[varIX] == deviceCardinalities[edge->newVarIX[varIX]]) {
                indivNewVarStates[varIX] = 0;
                s_p_real -= (deviceCardinalities[edge->newVarIX[varIX]]-1)*edge->newVarStateMultipliers[varIX];
            } else {
                s_p_real += edge->newVarStateMultipliers[varIX];
                break;
            }
        }
    }

    if (ecp != T(0)) {
        T sumVal = std::exp(mem[0] - maxval);
        for (size_t s_p = 1; s_p != s_p_e; ++s_p) {
            sumVal += std::exp(mem[s_p] - maxval);
        }
        maxval = ecp*(maxval + std::log(sumVal));
        //delete[] mem;
    }

    return maxval;
}

template<typename T, typename S>
__device__ void MPGraph<T,S>::UpdateRegion(T* lambdaBase, T* lambdaGlobal, int r, bool additiveUpdate) {
    if (lambdaBase == lambdaGlobal) {
        return;
    }
    GpuMPNode* r_ptr = deviceGraph[deviceValidRegionMapping[r]];

    GpuMsgContainer* p;
    size_t s_r_e = r_ptr->GetPotentialSize();
    if (additiveUpdate) {
        for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
            // for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p)
            for(size_t i = 0; i < r_ptr->numParents; i++)
            {
                p = &(r_ptr->GpuParents[i]);
                lambdaGlobal[p->lambda + s_r] += lambdaBase[p->lambda + s_r];
            }
        }
    } else {
        for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
            // for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p)
            for(size_t i = 0; i < r_ptr->numParents; i++)
            {
                p = &(r_ptr->GpuParents[i]);
                lambdaGlobal[p->lambda + s_r] = lambdaBase[p->lambda + s_r];
            }
        }
    }
}

template<typename T, typename S>
__device__ void MPGraph<T,S>::ReparameterizeRegion(T* lambdaBase, int r, T epsilon, bool additiveUpdate, RRegionWorkspaceID& wspace) {
    GpuMPNode* r_ptr = deviceGraph[deviceValidRegionMapping[r]];

    size_t ParentLocalIX;
    T* mu_p_r = wspace.RRegionMem;

    T sum_c_p = r_ptr->sum_c_r_c_p;
    //T sum_c_p = r_ptr->c_r;
    //for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
    //	sum_c_p += p->node->c_r;
    //}

    size_t s_r_e = r_ptr->GetPotentialSize();
    size_t rNumVar = r_ptr->varIXsize;
    //std::vector<S> indivVarStates(rNumVar, 0);
    S* indivVarStates = wspace.IXMem;
    for(S *tmp=indivVarStates, *tmp_e=indivVarStates+rNumVar;tmp!=tmp_e;++tmp) {
        *tmp = 0;
    }

    GpuMsgContainer* p;
    GpuMsgContainer* c;
    for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
        T phi_r_x_r_prime = (r_ptr->pot == NULL) ? T(0) : r_ptr->pot[s_r];

        // for (typename std::vector<MsgContainer>::const_iterator c = r_ptr->Children.begin(), c_e = r_ptr->Children.end(); c != c_e; ++c)
        for(size_t i = 0; i < r_ptr->numChildren; i++)
        {
            c = &(r_ptr->GpuChildren[i]);
            phi_r_x_r_prime += lambdaBase[c->lambda+c->Translator[s_r]];
        }

        ParentLocalIX = 0;
        // for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p, ++ParentLocalIX)
        for(size_t i = 0; i < r_ptr->numParents; ++i, ++ParentLocalIX)
        {
            p = &(r_ptr->GpuParents[i]);
            mu_p_r[ParentLocalIX] = ComputeMu(lambdaBase, p->edge, indivVarStates, rNumVar, epsilon, wspace.MuMem, wspace.MuIXMem);
            phi_r_x_r_prime += mu_p_r[ParentLocalIX];
        }

        phi_r_x_r_prime /= sum_c_p;

        ParentLocalIX = 0;
        // for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p, ++ParentLocalIX)
        for(size_t i = 0; i < r_ptr->numParents; i++, ++ParentLocalIX)
        {
            p = &(r_ptr->GpuParents[i]);
            GpuMPNode* ptr = p->node;//ptr points on parent, i.e., ptr->c_r = c_p!!!
            T value = ptr->c_r*phi_r_x_r_prime - mu_p_r[ParentLocalIX];
            lambdaBase[p->lambda+s_r] = ((additiveUpdate)?value-lambdaBase[p->lambda+s_r]:value);//the employed normalization is commutative
        }

        for (size_t varIX = 0; varIX < rNumVar; ++varIX) {
            ++indivVarStates[varIX];
            if (indivVarStates[varIX] == deviceCardinalities[r_ptr->varIX[varIX]]) {
                indivVarStates[varIX] = 0;
            } else {
                break;
            }
        }
    }

    // for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p)
    for(size_t i = 0; i < r_ptr->numParents; ++i)
    {
        p = &(r_ptr->GpuParents[i]);
        for (size_t s_r = s_r_e - 1; s_r != 0; --s_r) {
            lambdaBase[p->lambda+s_r] -= lambdaBase[p->lambda];
        }
        lambdaBase[p->lambda] = 0;
    }
}

template<typename T, typename S>
__device__ T MPGraph<T,S>::ComputeReparameterizationPotential(T* lambdaBase, const GpuMPNode* const r_ptr, const S s_r) const {
    T potVal = ((r_ptr->pot != NULL) ? r_ptr->pot[s_r] : T(0));

    GpuMsgContainer* pn;
    GpuMsgContainer* cn;
    // for (typename std::vector<MsgContainer>::const_iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn)
    for(size_t i = 0; i < r_ptr->numParents; ++i)
    {
        pn = &(r_ptr->GpuParents[i]);
        potVal -= lambdaBase[pn->lambda+s_r];
    }

    // for (typename std::vector<MsgContainer>::const_iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn)
    for(size_t i = 0; i < r_ptr->numChildren; i++)
    {
        cn = &(r_ptr->GpuChildren[i]);
        potVal += lambdaBase[cn->lambda+cn->Translator[s_r]];
    }

    return potVal;
}

template<typename T, typename S>
T MPGraph<T,S>::HostComputeReparameterizationPotential(T* lambdaBase, const MPNode* const r_ptr, const S s_r) const {
    T potVal = ((r_ptr->pot != NULL) ? r_ptr->pot[s_r] : T(0));

    for (typename std::vector<MsgContainer>::const_iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn) {
        potVal -= lambdaBase[pn->lambda+s_r];
    }

    for (typename std::vector<MsgContainer>::const_iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn) {
        potVal += lambdaBase[cn->lambda+cn->Translator[s_r]];
    }

    return potVal;
}


template<typename T, typename S>
__device__ T MPGraph<T,S>::ComputeDual(T* lambdaBase, T epsilon, DualWorkspaceID& dw) const {
    T dual = T(0);


    // for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r)
    for(size_t i = 0; i < deviceNodes; i++)
    {
        GpuMPNode* r_ptr = deviceGraph[i];
        T ecr = epsilon*r_ptr->c_r;
        size_t s_r_e = r_ptr->GetPotentialSize();

        T* mem = dw.DualWorkspace;

        T maxVal = -std::numeric_limits<T>::max();
        for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
            T potVal = ComputeReparameterizationPotential(lambdaBase, r_ptr, s_r);

            if (ecr != T(0)) {
                potVal /= ecr;
                mem[s_r] = potVal;
            }

            maxVal = ((potVal > maxVal) ? potVal : maxVal);
        }

        if (ecr != T(0)) {
            T sum = std::exp(mem[0] - maxVal);
            for (size_t s_r = 1; s_r != s_r_e; ++s_r) {
                sum += std::exp(mem[s_r] - maxVal);
            }
            dual += ecr*(maxVal + std::log(sum + T(1e-20)));
        } else {
            dual += maxVal;
        }
    }
    return dual;
}

template<typename T, typename S>
__device__ size_t MPGraph<T,S>::GetLambdaSize() const {
    return LambdaSize;
}

template<typename T, typename S>
size_t MPGraph<T,S>::HostGetLambdaSize() const {
    return LambdaSize;
}

template<typename T, typename S>
__device__ void MPGraph<T,S>::GradientUpdateEdge(T* lambdaBase, int e, T epsilon, T stepSize, bool additiveUpdate, GEdgeWorkspaceID& gew) {
    GpuEdgeID* edge = deviceEdges[e];
    GpuMPNode* r_ptr = edge->childPtr->node;
    GpuMPNode* p_ptr = edge->parentPtr->node;

    size_t s_r_e = r_ptr->GetPotentialSize();
    T* mem_r = gew.mem1;
    ComputeBeliefForRegion(r_ptr, lambdaBase, epsilon, mem_r, s_r_e);

    size_t s_p_e = p_ptr->GetPotentialSize();
    T* mem_p = gew.mem2;
    ComputeBeliefForRegion(p_ptr, lambdaBase, epsilon, mem_p, s_p_e);

    for (size_t s_p = 0; s_p < s_p_e; ++s_p) {
        mem_r[edge->childPtr->Translator[s_p]] -= mem_p[s_p];
    }

    if (additiveUpdate) {
        for (size_t s_r = 0; s_r < s_r_e; ++s_r) {
            //lambdaBase[edge->childPtr->lambda + s_r] += stepSize*mem_r[s_r];//use this line if global memory is identical to local memory
            lambdaBase[edge->childPtr->lambda + s_r] = stepSize*mem_r[s_r];//addition is performed in UpdateEdge
        }
    } else {
        for (size_t s_r = 0; s_r < s_r_e; ++s_r) {
            lambdaBase[edge->childPtr->lambda + s_r] += stepSize*mem_r[s_r];
        }
    }
}

template<typename T, typename S>
__device__ void MPGraph<T,S>::ComputeBeliefForRegion(GpuMPNode* r_ptr, T* lambdaBase, T epsilon, T* mem, size_t s_r_e) {
    T ecr = epsilon*r_ptr->c_r;

    T maxVal = -std::numeric_limits<T>::max();
    for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
        T potVal = ComputeReparameterizationPotential(lambdaBase, r_ptr, s_r);

        if (ecr != T(0)) {
            potVal /= ecr;
        }
        mem[s_r] = potVal;

        maxVal = ((potVal > maxVal) ? potVal : maxVal);
    }

    T sum = T(0);
    if (ecr != T(0)) {
        for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
            mem[s_r] = std::exp(mem[s_r] - maxVal);
            sum += mem[s_r];
        }
    } else {
        for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
            mem[s_r] = ((mem[s_r] - maxVal) > -1e-5) ? T(1) : T(0);
            sum += mem[s_r];
        }
    }
    for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
        mem[s_r] /= sum;
    }
}

template<typename T, typename S>
__device__ size_t MPGraph<T,S>::ComputeBeliefs(T* lambdaBase, T epsilon, T** belPtr, bool OnlyUnaries) {
    size_t BeliefSize = 0;
    // for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r)
    for(size_t i = 0; i < deviceNodes; i++)
    {
        GpuMPNode* r_ptr = deviceGraph[i];
        size_t s_r_e = r_ptr->GetPotentialSize();
        size_t numVars = r_ptr->varIXsize;
        if ((OnlyUnaries&&numVars == 1) || !OnlyUnaries) {
            BeliefSize += s_r_e;
        }
    }
    T* beliefs = new T[BeliefSize];

    T* mem = beliefs;
    if (belPtr != NULL) {
        *belPtr = beliefs;
    }

    // for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r)
    for(size_t i = 0; i < deviceNodes; i++)
    {
        GpuMPNode* r_ptr = deviceGraph[i];
        size_t numVars = r_ptr->varIXsize;
        if (OnlyUnaries&&numVars > 1) {
            continue;
        }

        size_t s_r_e = r_ptr->GetPotentialSize();
        ComputeBeliefForRegion(r_ptr, lambdaBase, epsilon, mem, s_r_e);

        r_ptr->tmp = (void*)mem;
        mem += s_r_e;
    }

    return BeliefSize;
}

template<typename T, typename S>
__device__  void MPGraph<T,S>::Marginalize(T* curBel, T* oldBel, GpuEdgeID* edge, const S* indivVarStates, size_t indivVarStatesLength, T& marg_new, T& marg_old) {
    //MPNode* r_ptr = edge->childPtr->node;
    //MPNode* p_ptr = edge->parentPtr->node;

    size_t numVarsOverlap = indivVarStatesLength;
    size_t s_p_stat = 0;
    for (size_t k = 0; k<numVarsOverlap; ++k) {
        s_p_stat += indivVarStates[k]*edge->rStateMultipliers[k];
    }

    size_t s_p_e = edge->newVarSize;
    size_t numVarNew = edge->newVarIXsize;

    //individual vars;
    // std::vector<S> indivNewVarStates(numVarNew, 0);
    S* indivNewVarStates = new S[numVarNew];
    for(size_t i = 0; i < numVarNew; i++)
    {
        indivNewVarStates[i] = 0;
    }

    for (size_t s_p = 0; s_p != s_p_e; ++s_p) {
        size_t s_p_real = s_p_stat;
        for (size_t varIX = 0; varIX<numVarNew; ++varIX) {
            s_p_real += indivNewVarStates[varIX]*edge->newVarStateMultipliers[varIX];
        }

        marg_new += curBel[s_p_real];
        marg_old += oldBel[s_p_real];

        for (size_t varIX = 0; varIX < numVarNew; ++varIX) {
            ++indivNewVarStates[varIX];
            if (indivNewVarStates[varIX] == deviceCardinalities[edge->newVarIX[varIX]]) {
                indivNewVarStates[varIX] = 0;
            } else {
                break;
            }
        }
    }

    // recycle allocated content
    delete [] indivNewVarStates;
}

template<typename T, typename S>
__device__ T MPGraph<T,S>::ComputeImprovement(T* curBel, T* oldBel) {//not efficient
    T imp = T(0);
    GpuMsgContainer* pn;

    // for (typename std::vector<MPNode*>::iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r)
    for(size_t i = 0; i < deviceNodes; i++)
    {
        GpuMPNode* r_ptr = deviceGraph[i];
        size_t s_r_e = r_ptr->GetPotentialSize();
        size_t belIX_r = ((T*)r_ptr->tmp) - curBel;

        // for (typename std::vector<MsgContainer>::const_iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn)
        for(size_t j = 0; j < r_ptr->numParents; j++)
        {
            pn = &(r_ptr->GpuParents[j]);
            GpuMPNode* p_ptr = pn->node;
            size_t belIX_p = ((T*)p_ptr->tmp) - curBel;

            T v1 = 0;
            T v2 = 0;
            size_t rNumVar = r_ptr->varIXsize;
            S* indivVarStates = new S[rNumVar];
            for(size_t i = 0; i < rNumVar; i++)
            {
                indivVarStates[i] = 0;
            }


            // std::vector<S> indivVarStates(rNumVar, 0);
            for(size_t s_r=0;s_r<s_r_e;++s_r)
            {
                T marg_old = T(0);
                T marg_new = T(0);
                Marginalize(curBel + belIX_p, oldBel + belIX_p, pn->edge, indivVarStates, rNumVar, marg_new, marg_old);

                v1 += curBel[belIX_r+s_r]*sqrtf(marg_old/oldBel[belIX_r+s_r]);
                v2 += marg_new*sqrtf(oldBel[belIX_r+s_r]/marg_old);

                for (size_t varIX = 0; varIX < rNumVar; ++varIX) {
                    ++indivVarStates[varIX];
                    if (indivVarStates[varIX] == deviceCardinalities[r_ptr->varIX[varIX]]) {
                        indivVarStates[varIX] = 0;
                    } else {
                        break;
                    }
                }
            }


            imp += logf(v1) + logf(v2);

            delete [] indivVarStates;
        }
    }

    return imp;
}

template<typename T, typename S>
__device__ void MPGraph<T,S>::DeleteBeliefs() {
    GpuMPNode* r_ptr = deviceGraph[0];
    delete[]((T*)r_ptr->tmp);
}


template<typename T, typename S>
ThreadSync<T,S>::ThreadSync(int nT, T* lambdaGlobal, T epsilon, MPGraph<T, S>* g) : state(NONE), numThreads(nT), lambdaGlobal(lambdaGlobal), epsilon(epsilon), g(g), currentlyStoppedThreads(0), prevDual(std::numeric_limits<T>::max())  {
    dw = g->HostAllocateDualWorkspaceMem(epsilon);
    state = INIT;
    LambdaForNoSync.assign(g->HostGetLambdaSize(),0);
}

template<typename T, typename S>
ThreadSync<T,S>::~ThreadSync() {
    g->HostDeAllocateDualWorkspaceMem(dw);
}

template<typename T, typename S>
bool ThreadSync<T,S>::checkSync() {
    //std::cout << state << std::endl;
    if (unlikely(state == INTR)) {
        std::unique_lock<std::mutex> lock(mtx);
        if (currentlyStoppedThreads < numThreads - 1) {
            ++currentlyStoppedThreads;
            cond_.wait(lock);
        } else if (currentlyStoppedThreads == numThreads - 1) {
            double timeMS = CTmr.Stop()*1000.;
            //std::cout << timeMS << "; " << CTmr1.Stop()*1000. << "; ";
            T dualVal = g->HostComputeDual(lambdaGlobal, epsilon, dw);
            //std::cout << timeMS <<"; " << CTmr1.Stop()*1000. << "; " << std::setprecision(15) << dualVal << std::endl;
            std::cout << timeMS <<"; " << CTmr1.Stop()*1000. << "; " << dualVal << std::endl;
            //if (dualVal>prevDual) {
            //	std::cout << " (*)";
            //}
            //std::cout << std::endl;
            prevDual = dualVal;
            CTmr.Start();
            state = NONE;
            currentlyStoppedThreads = 0;
            cond_.notify_all();
        }
    } else if (unlikely(state == TERM)) {
        std::unique_lock<std::mutex> lock(mtx);
        ++currentlyStoppedThreads;
        if (currentlyStoppedThreads == numThreads) {
            //T dualVal = g->ComputeDual(lambdaGlobal, epsilon, dw);
            //std::cout << std::setprecision(15) << dualVal << std::endl;
            std::cout << "All threads terminated." << std::endl;
            state = NONE;
            currentlyStoppedThreads = 0;
            cond_.notify_all();
        }
        return false;
    } else if (unlikely(state == INIT)) {
        std::unique_lock<std::mutex> lock(mtx);
        ++currentlyStoppedThreads;
        if (currentlyStoppedThreads == numThreads) {
            std::cout << "All threads running." << std::endl;
        }
        cond_.wait(lock);
    }
    return true;
}

template<typename T, typename S>
void ThreadSync<T,S>::interruptFunc() {
    std::unique_lock<std::mutex> lock(mtx);
    state = INTR;
    cond_.wait(lock);
}

template<typename T, typename S>
void ThreadSync<T,S>::terminateFunc() {
    std::unique_lock<std::mutex> lock(mtx);
    state = TERM;
    cond_.wait(lock);
}

template<typename T, typename S>
void ThreadSync<T,S>::ComputeDualNoSync() {
    ///std::cout << "line 1016" << std::endl;
    double timeMS = CTmr.Stop()*1000.;
    //std::cout << "line 1018" << std::endl;
    std::copy(lambdaGlobal, lambdaGlobal+LambdaForNoSync.size(), &LambdaForNoSync[0]);
   // std::cout << "line 1020" << std::endl;
    T dualVal = g->HostComputeDual(&LambdaForNoSync[0], epsilon, dw);
    std::cout << timeMS <<"; " << CTmr1.Stop()*1000. << "; " << dualVal << std::endl;
}

template<typename T, typename S>
void ThreadSync<T,S>::CudaComputeDualNoSync() {
    std::cout << "line 1016" << std::endl;
    double timeMS = CTmr.Stop()*1000.;
    std::cout << "line 1018" << std::endl;
    std::copy(lambdaGlobal, lambdaGlobal+LambdaForNoSync.size(), &LambdaForNoSync[0]);
    std::cout << "line 1020" << std::endl;
    T dualVal = g->HostComputeDual(&LambdaForNoSync[0], epsilon, dw);
    std::cout << timeMS <<"; " << CTmr1.Stop()*1000. << "; " << dualVal << std::endl;
}

template<typename T, typename S>
bool ThreadSync<T,S>::startFunc() {
    std::unique_lock<std::mutex> lock(mtx);
    if (currentlyStoppedThreads == numThreads) {
        state = NONE;
        currentlyStoppedThreads = 0;
        CTmr1.Start();
        CTmr.Start();
        cond_.notify_all();
        return true;
    } else {
        return false;
    }
}

template<typename T, typename S>
ThreadWorker<T,S>::ThreadWorker(ThreadSync<T, S>* ts, MPGraph<T, S>* g, T epsilon, int randomSeed, T* lambdaGlobal, T* stepsize) : cnt(0), ts(ts), thread_(NULL), g(g), epsilon(epsilon), randomSeed(randomSeed), lambdaGlobal(lambdaGlobal), stepsize(stepsize) {
    // msgSize = g->GetLambdaSize();
    // assert(msgSize > 0);
    // lambdaLocal.assign(msgSize, T(0));
    // lambdaBase = &lambdaLocal[0];
    //
    // //REdgeWorkspaceID* test;
    //
    // // see if cuda works
    //
    //
    // #if WHICH_FUNC==1
    //     rrw = g->AllocateReparameterizeRegionWorkspaceMem(epsilon);
    //     uid = new std::uniform_int_distribution<int>(0, g->NumberOfRegionsWithParents() - 1);
    // #elif WHICH_FUNC==2
    //     rew = g->AllocateReparameterizeEdgeWorkspaceMem(epsilon);
    //     uid = new std::uniform_int_distribution<int>(0, g->NumberOfEdges() - 1);
    // #elif WHICH_FUNC==3
    //     fw = g->AllocateFunctionUpdateWorkspaceMem();
    //     uid = new std::uniform_int_distribution<int>(0, g->NumberOfRegionsTotal()-1);
    // #endif
    //
    // eng.seed(randomSeed);
}

template<typename T, typename S>
ThreadWorker<T,S>::~ThreadWorker() {
// #if WHICH_FUNC==1
//     g->DeAllocateReparameterizeRegionWorkspaceMem(rrw);
// #elif WHICH_FUNC==2
//     g->DeAllocateReparameterizeEdgeWorkspaceMem(rew);
// #elif WHICH_FUNC==3
//     g->DeAllocateFunctionUpdateWorkspaceID(fw);
// #endif
//     if (uid != NULL) {
//         delete uid;
//         uid = NULL;
//     }
//     if (thread_ != NULL) {
//         delete thread_;
//         thread_ = NULL;
//     }
}

template<typename T, typename S>
void ThreadWorker<T,S>::start() {
    // thread_ = new std::thread(std::bind(&ThreadWorker::run, this));
    // thread_->detach();
}

template<typename T, typename S>
size_t ThreadWorker<T,S>::GetCount() {
    return cnt;
}

template<typename T, typename S>
void ThreadWorker<T,S>::run() {
    // std::cout << "Thread started" << std::endl;
    // int test = 0;
    // // ts->startFunc();
    // while (ts->checkSync()) {
    //
    //     #if WHICH_FUNC==1
    //             int ix = (*uid)(eng);
    //             g->CopyMessagesForStar(lambdaGlobal, lambdaBase, ix);
    //             g->ReparameterizeRegion(lambdaBase, ix, epsilon, false, rrw);
    //             g->UpdateRegion(lambdaBase, lambdaGlobal, ix, false);
    //     #elif WHICH_FUNC==2
    //             int ix = (*uid)(eng);
    //             g->CopyMessagesForEdge(lambdaGlobal, lambdaBase, ix);
    //             g->ReparameterizeEdge(lambdaBase, ix, epsilon, false, rew);
    //             g->UpdateEdge(lambdaBase, lambdaGlobal, ix, false);
    //     #elif WHICH_FUNC==3
    //             int ix = (*uid)(eng);
    //             g->CopyMessagesForLocalFunction(lambdaGlobal, lambdaBase, ix);
    //             g->ComputeLocalFunctionUpdate(lambdaBase, ix, epsilon, *stepsize, true, fw);
    //             g->UpdateLocalFunction(lambdaBase, lambdaGlobal, ix, true);
    //
    //     #endif
    //
    //     ++cnt;
    // }
}


template<typename T, typename S>
int AsyncRMPThread<T,S>::RunMP(MPGraph<T, S>& g, T epsilon, int numIterations, int numThreads, int WaitTimeInMS) {
    size_t msgSize = g.HostGetLambdaSize();
    //if (msgSize == 0) {
    //    typename MPGraph<T, S>::DualWorkspaceID dw = g.HostAllocateDualWorkspaceMem(epsilon);
    //     std::cout << "0: " << g.HostComputeDual(NULL, epsilon, dw) << std::endl;
    //     g.HostDeAllocateDualWorkspaceMem(dw);
    //     return 0;
    // }
    //
    // std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
    //
    // lambdaGlobal.assign(msgSize, T(0));
    // T* lambdaGlob = &lambdaGlobal[0];
    //
    //
    // // thread syncs are dumb - they're used for keeping track of
    // // the state of each thread, which is just totally unncessary.
    // ThreadSync<T, S> sy(numThreads, lambdaGlob, epsilon, &g);
    //
    // T stepsize = -0.1;
    // std::vector<ThreadWorker<T, S>*> ex(numThreads, NULL);
    // for (int k = 0; k < numThreads; ++k) {
    //     ex[k] = new ThreadWorker<T, S>(&sy, &g, epsilon, k, lambdaGlob, &stepsize);
    // }
    // for (int k = 0; k < numThreads; ++k) {
    //     ex[k]->start();
    // }
    //
    // while (!sy.startFunc()) {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    // }
    // for (int k = 0; k < numIterations; ++k)
    // {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(WaitTimeInMS));
    //     //sy.interruptFunc();
    //     sy.ComputeDualNoSync();
    // }
    // sy.terminateFunc();
    //
    // size_t regionUpdates = 0;
    // for(int k=0;k<numThreads;++k) {
    //     size_t tmp = ex[k]->GetCount();
    //     std::cout << "Thread " << k << ": " << tmp << std::endl;
    //     regionUpdates += tmp;
    //     delete ex[k];
    // }
    // std::cout << "Region updates: " << regionUpdates << std::endl;
    // std::cout << "Total regions:  " << g.NumberOfRegionsWithParents() << std::endl;
    //
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    // std::cout << "Terminating program." << std::endl;
    return 0;
}

template<typename T, typename S>
size_t AsyncRMPThread<T,S>::GetBeliefs(MPGraph<T, S>& g, T epsilon, T** belPtr, bool OnlyUnaries) {
    /*
    size_t msgSize = g.GetLambdaSize();
    if (msgSize == 0) {
         return g.ComputeBeliefs(NULL, epsilon, belPtr, OnlyUnaries);
    } else {
         if (lambdaGlobal.size() != msgSize) {
             std::cout << "Message size does not fit requirement. Reassigning." << std::endl;
             lambdaGlobal.assign(msgSize, T(0));
         }
         return g.ComputeBeliefs(&lambdaGlobal[0], epsilon, belPtr, OnlyUnaries);
    }
    */
    return size_t(0);
}

template<typename T, typename S>
RMP<T,S>::RMP()
{

}

template<typename T, typename S>
RMP<T,S>::~RMP()
{

}

template<typename T, typename S>
int RMP<T,S>::RunMP(MPGraph<T, S>& g, T epsilon)
{
return 0;
}


#endif
