#include "Region.h"
#include <cuda.h>


template<typename T, typename S>
T* MPGraph<T,S>::CudaGetMaxMemComputeMu(T epsilon) const {
    size_t maxMem = 0;
    for (typename std::vector<EdgeID*>::const_iterator e = Edges.begin(), e_e = Edges.end(); e != e_e; ++e) {
        EdgeID* edge = *e;
        size_t s_p_e = edge->newVarSize;
        MPNode* p_ptr = edge->parentPtr->node;

        T ecp = epsilon*p_ptr->c_r;
        if (ecp != T(0)) {
            maxMem = (s_p_e > maxMem) ? s_p_e : maxMem;
        }
    }


    T* deviceMem;
    if(maxMem > 0)
    {
        cudaMalloc((void**)deviceMem, maxMem * sizeof(T));
    }
    else
    {
        deviceMem = NULL;
    }   

    return deviceMem;
}


template<typename T, typename S>
S* MPGraph<T,S>::CudaGetMaxMemComputeMuIXVar() const {
    size_t maxMem = 0;
    for (size_t r = 0; r < ValidRegionMapping.size(); ++r) {
        MPNode* r_ptr = Graph[ValidRegionMapping[r]];
        for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
            size_t tmp = p->edge->newVarIX.size();
            maxMem = (tmp>maxMem) ? tmp : maxMem;
        }
    }

    S* deviceMem;
    if(maxMem > 0)
    {
        cudaMalloc((void**)deviceMem, maxMem * sizeof(S));
    }
    else
    {
        deviceMem = NULL;
    }   

    return deviceMem;
}



/*
    So what we'll do here is allocate a bunch of arrays on the device, but return a struct on the host containing all those pointers.
*/
template<typename T, typename S>
__host__ const typename MPGraph<T,S>::REdgeWorkspaceID MPGraph<T,S>::CudaAllocateReparameterizeEdgeWorkspaceMem(T epsilon) const {
    size_t maxIXMem = 0;
    for(typename std::vector<EdgeID*>::const_iterator eb=Edges.begin(), eb_e=Edges.end();eb!=eb_e;++eb) {
        MPNode* r_ptr = (*eb)->childPtr->node;
        size_t rNumVar = r_ptr->varIX.size();
        maxIXMem = (rNumVar>maxIXMem) ? rNumVar : maxIXMem;
    }

    S* deviceIXMem;
    if(maxIXMem > 0)
    {
        cudaMalloc((void**)deviceIXMem, maxIXMem * sizeof(S));
    }
    else
    {
        deviceIXMem = NULL;
    }



    return REdgeWorkspaceID{ CudaGetMaxMemComputeMu(epsilon), CudaGetMaxMemComputeMuIXVar(), deviceIXMem };
}

/*
    Deallocation function for GPU pointers.
*/

template<typename T, typename S>
__host__ void MPGraph<T,S>::CudaDeAllocateReparameterizeEdgeWorkspaceMem(REdgeWorkspaceID& w) const {
    cudaFree(w.MuMem);
    cudaFree(w.MuIXMem);
    cudaFree(w.IXMem);
    w.MuMem = NULL;
    w.MuIXMem = NULL;
    w.IXMem = NULL;
}


template<typename T, typename S>
__device__ void MPGraph<T,S>::CudaCopyLambda(T* lambdaSrc, T* lambdaDst, size_t s_r_e) const
{
    for (T* ptr_e = lambdaSrc + s_r_e; ptr_e != lambdaSrc;) {
        *lambdaDst++ = *lambdaSrc++;
    }
}



