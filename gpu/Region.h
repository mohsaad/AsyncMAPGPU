
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

                CUDA_HOSTDEV Region(T c_r, T* pot, S potSize, const std::vector<S>& varIX) : c_r(c_r), pot(pot), potSize(potSize), tmp(NULL), varIX(varIX)
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
        MPGraph();

        virtual ~MPGraph();

        int AddVariables(const std::vector<S>& card);

        const PotentialID AddPotential(const PotentialVector& potVals);

        const RegionID AddRegion(T c_r, const std::vector<S>& varIX, const PotentialID& p);

        int AddConnection(const RegionID& child, const RegionID& parent);

        int AllocateMessageMemory();

        const DualWorkspaceID AllocateDualWorkspaceMem(T epsilon) const;

    	void DeAllocateDualWorkspaceMem(DualWorkspaceID& dw) const;

    	T* GetMaxMemComputeMu(T epsilon) const;

        T* CudaGetMaxMemComputeMu(T epsilon) const;

        S* GetMaxMemComputeMuIXVar() const;

        S* CudaGetMaxMemComputeMuIXVar() const;

        const RRegionWorkspaceID AllocateReparameterizeRegionWorkspaceMem(T epsilon) const;

        void DeAllocateReparameterizeRegionWorkspaceMem(RRegionWorkspaceID& w) const;

    	const REdgeWorkspaceID AllocateReparameterizeEdgeWorkspaceMem(T epsilon) const;

        const REdgeWorkspaceID CudaAllocateReparameterizeEdgeWorkspaceMem(T epsilon) const;

    	void DeAllocateReparameterizeEdgeWorkspaceMem(REdgeWorkspaceID& w) const;

        void CudaDeAllocateReparameterizeEdgeWorkspaceMem(REdgeWorkspaceID& w) const;

        const GEdgeWorkspaceID AllocateGradientEdgeWorkspaceMem() const;

    	void DeAllocateGradientEdgeWorkspaceMem(GEdgeWorkspaceID& w) const;

        const FunctionUpdateWorkspaceID AllocateFunctionUpdateWorkspaceMem() const;

        void DeAllocateFunctionUpdateWorkspaceID(FunctionUpdateWorkspaceID& w) const;

        int FillEdge();

    	void ComputeCumulativeSize(MPNode* r_ptr, std::vector<S>& cumVarR);

        int FillTranslator();

    	size_t NumberOfRegionsTotal() const;

    	size_t NumberOfRegionsWithParents() const;

        size_t NumberOfEdges() const;

        void UpdateEdge(T* lambdaBase, T* lambdaGlobal, int e, bool additiveUpdate);

        void CopyLambda(T* lambdaSrc, T* lambdaDst, size_t s_r_e) const;

        void CudaCopyLambda(T* lambdaSrc, T* lambdaDst, size_t s_r_e) const;

    	void CopyMessagesForLocalFunction(T* lambdaSrc, T* lambdaDst, int r) const;

        void ComputeLocalFunctionUpdate(T* lambdaBase, int r, T epsilon, T multiplier, bool additiveUpdate, FunctionUpdateWorkspaceID& w);

    	void UpdateLocalFunction(T* lambdaBase, T* lambdaGlobal, int r, bool additiveUpdate);

    	void CopyMessagesForEdge(T* lambdaSrc, T* lambdaDst, int e) const;

        void CopyMessagesForStar(T* lambdaSrc, T* lambdaDst, int r) const;

        void ReparameterizeEdge(T* lambdaBase, int e, T epsilon, bool additiveUpdate, REdgeWorkspaceID& wspace);

        T ComputeMu(T* lambdaBase, EdgeID* edge, S* indivVarStates, size_t numVarsOverlap, T epsilon, T* workspaceMem, S* MuIXMem);

        void UpdateRegion(T* lambdaBase, T* lambdaGlobal, int r, bool additiveUpdate);

        void ReparameterizeRegion(T* lambdaBase, int r, T epsilon, bool additiveUpdate, RRegionWorkspaceID& wspace);

        T ComputeReparameterizationPotential(T* lambdaBase, const MPNode* const r_ptr, const S s_r) const;

        T ComputeDual(T* lambdaBase, T epsilon, DualWorkspaceID& dw) const;

        size_t GetLambdaSize() const;

        void GradientUpdateEdge(T* lambdaBase, int e, T epsilon, T stepSize, bool additiveUpdate, GEdgeWorkspaceID& gew);

        void ComputeBeliefForRegion(MPNode* r_ptr, T* lambdaBase, T epsilon, T* mem, size_t s_r_e);

    	size_t ComputeBeliefs(T* lambdaBase, T epsilon, T** belPtr, bool OnlyUnaries);

        void Marginalize(T* curBel, T* oldBel, EdgeID* edge, const std::vector<S>& indivVarStates, T& marg_new, T& marg_old);

    	T ComputeImprovement(T* curBel, T* oldBel);

    	void DeleteBeliefs();

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

        void interruptFunc();

        void terminateFunc();

        void ComputeDualNoSync();

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
