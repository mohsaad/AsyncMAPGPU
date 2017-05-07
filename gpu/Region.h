
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
#include <cuda.h>

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

        bool DeallocateGpuGraphh();

    private:






        void AllocateNewGPUNode(MPNode* cpuNode, T c_r, const std::vector<S>& varIX, T* pot, S potSize);

        bool DeallocateGpuNode(GpuMPNode* node);



        bool DeallocateGpuEdge(GpuEdge* edge);
    

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

#endif
