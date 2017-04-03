
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


#define BLOCK_SIZE 256
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

    private:
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


    public:
        CUDA_HOSTDEV MPGraph();

        CUDA_HOSTDEV virtual ~MPGraph();

        CUDA_HOSTDEV int AddVariables(const std::vector<S>& card);

        CUDA_HOSTDEV const PotentialID AddPotential(const PotentialVector& potVals);

        CUDA_HOSTDEV const RegionID AddRegion(T c_r, const std::vector<S>& varIX, const PotentialID& p);

        CUDA_HOSTDEV int AddConnection(const RegionID& child, const RegionID& parent);

        CUDA_HOSTDEV int AllocateMessageMemory();

        CUDA_HOSTDEV const DualWorkspaceID AllocateDualWorkspaceMem(T epsilon) const;

    	CUDA_HOSTDEV void DeAllocateDualWorkspaceMem(DualWorkspaceID& dw) const;

    	CUDA_HOSTDEV T* GetMaxMemComputeMu(T epsilon) const;

        T* CudaGetMaxMemComputeMu(T epsilon) const;

        CUDA_HOSTDEV S* GetMaxMemComputeMuIXVar() const;

        S* CudaGetMaxMemComputeMuIXVar() const;

        CUDA_HOSTDEV const RRegionWorkspaceID AllocateReparameterizeRegionWorkspaceMem(T epsilon) const;

        CUDA_HOSTDEV void DeAllocateReparameterizeRegionWorkspaceMem(RRegionWorkspaceID& w) const;

    	CUDA_HOSTDEV const REdgeWorkspaceID AllocateReparameterizeEdgeWorkspaceMem(T epsilon) const;

        const REdgeWorkspaceID CudaAllocateReparameterizeEdgeWorkspaceMem(T epsilon) const;

    	CUDA_HOSTDEV void DeAllocateReparameterizeEdgeWorkspaceMem(REdgeWorkspaceID& w) const;

        void CudaDeAllocateReparameterizeEdgeWorkspaceMem(REdgeWorkspaceID& w) const;

        CUDA_HOSTDEV const GEdgeWorkspaceID AllocateGradientEdgeWorkspaceMem() const;

    	CUDA_HOSTDEV void DeAllocateGradientEdgeWorkspaceMem(GEdgeWorkspaceID& w) const;

        CUDA_HOSTDEV const FunctionUpdateWorkspaceID AllocateFunctionUpdateWorkspaceMem() const;

        CUDA_HOSTDEV void DeAllocateFunctionUpdateWorkspaceID(FunctionUpdateWorkspaceID& w) const;

        CUDA_HOSTDEV int FillEdge();

    	CUDA_HOSTDEV void ComputeCumulativeSize(MPNode* r_ptr, std::vector<S>& cumVarR);

        CUDA_HOSTDEV int FillTranslator();

    	CUDA_HOSTDEV size_t NumberOfRegionsTotal() const;

    	CUDA_HOSTDEV size_t NumberOfRegionsWithParents() const;

        CUDA_HOSTDEV size_t NumberOfEdges() const;

        CUDA_HOSTDEV void UpdateEdge(T* lambdaBase, T* lambdaGlobal, int e, bool additiveUpdate);

        CUDA_HOSTDEV void CopyLambda(T* lambdaSrc, T* lambdaDst, size_t s_r_e) const;

    	CUDA_HOSTDEV void CopyMessagesForLocalFunction(T* lambdaSrc, T* lambdaDst, int r) const;

        CUDA_HOSTDEV void ComputeLocalFunctionUpdate(T* lambdaBase, int r, T epsilon, T multiplier, bool additiveUpdate, FunctionUpdateWorkspaceID& w);

    	CUDA_HOSTDEV void UpdateLocalFunction(T* lambdaBase, T* lambdaGlobal, int r, bool additiveUpdate);

    	CUDA_HOSTDEV void CopyMessagesForEdge(T* lambdaSrc, T* lambdaDst, int e) const;

        //void CudaCopyMessagesForEdge(T* lambdaSrc, T* lambdaDst, int e) const ;

        CUDA_HOSTDEV void CopyMessagesForStar(T* lambdaSrc, T* lambdaDst, int r) const;

        CUDA_HOSTDEV void ReparameterizeEdge(T* lambdaBase, int e, T epsilon, bool additiveUpdate, REdgeWorkspaceID& wspace);

        CUDA_HOSTDEV T ComputeMu(T* lambdaBase, EdgeID* edge, S* indivVarStates, size_t numVarsOverlap, T epsilon, T* workspaceMem, S* MuIXMem);

        CUDA_HOSTDEV void UpdateRegion(T* lambdaBase, T* lambdaGlobal, int r, bool additiveUpdate);

        CUDA_HOSTDEV void ReparameterizeRegion(T* lambdaBase, int r, T epsilon, bool additiveUpdate, RRegionWorkspaceID& wspace);

        CUDA_HOSTDEV T ComputeReparameterizationPotential(T* lambdaBase, const MPNode* const r_ptr, const S s_r) const;

        CUDA_HOSTDEV T ComputeDual(T* lambdaBase, T epsilon, DualWorkspaceID& dw) const;

        CUDA_HOSTDEV size_t GetLambdaSize() const;

        CUDA_HOSTDEV void GradientUpdateEdge(T* lambdaBase, int e, T epsilon, T stepSize, bool additiveUpdate, GEdgeWorkspaceID& gew);

        CUDA_HOSTDEV void ComputeBeliefForRegion(MPNode* r_ptr, T* lambdaBase, T epsilon, T* mem, size_t s_r_e);

    	CUDA_HOSTDEV size_t ComputeBeliefs(T* lambdaBase, T epsilon, T** belPtr, bool OnlyUnaries);

        CUDA_HOSTDEV void Marginalize(T* curBel, T* oldBel, EdgeID* edge, const std::vector<S>& indivVarStates, T& marg_new, T& marg_old);

    	CUDA_HOSTDEV T ComputeImprovement(T* curBel, T* oldBel);

    	CUDA_HOSTDEV void DeleteBeliefs();



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
