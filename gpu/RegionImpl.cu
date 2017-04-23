#include "Region.h"

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
bool MPGraph<T,S>::DeallocateGpuNode(GpuMPNode* node)
{
    GpuMPNode hostNode(0, NULL, NULL, 0, 0);
    gpuErrchk(cudaMemcpy(&hostNode, node, sizeof(GpuMPNode*), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(hostNode.varIX));
    gpuErrchk(cudaFree(hostNode.pot));
    gpuErrchk(cudaFree(hostNode.tmp));

    gpuErrchk(cudaFree(node));
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

    // add that to vector as well
    GpuCardinalities.clear();
    for(size_t i = 0; i < card.size(); i++)
    {
        GpuCardinalities.push_back(card[i]);
    }

    return 0;
}

template<typename T, typename S>
const typename MPGraph<T,S>::PotentialID MPGraph<T,S>::AddPotential(const typename MPGraph<T,S>::PotentialVector& potVals) {
    S PotID = S(Potentials.size());
    Potentials.push_back(potVals);

    // also push back on GPU
    GpuPotentials.push_back(potVals);

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

    GpuMPNode* gpuChild = CpuGpuMap[c];
    GpuMPNode* gpuParent = CpuGpuMap[p];

    GpuMsgContainer cCopy(0, gpuParent, NULL, NULL);
    GpuMsgContainer pCopy(0, gpuChild, NULL, NULL);


    GpuMsgContainer* CMsg;
    GpuMsgContainer* PMsg;
    gpuErrchk(cudaMalloc((void**)&CMsg, sizeof(GpuMsgContainer)));
    gpuErrchk(cudaMalloc((void**)&PMsg, sizeof(GpuMsgContainer)));


    // copy over stuff to device
    gpuErrchk(cudaMemcpy(CMsg, &cCopy, sizeof(cCopy), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(PMsg, &pCopy, sizeof(pCopy), cudaMemcpyHostToDevice));

    nodeParents[gpuChild].push_back(CMsg);
    nodeChildren[gpuParent].push_back(PMsg);

    // insert a copy here
    CpuGpuMsgMap.insert(std::pair<MsgContainer*, GpuMsgContainer*>(&(c->Parents.back()), nodeParents[CpuGpuMap[c]].back()));
    CpuGpuMsgMap.insert(std::pair<MsgContainer*, GpuMsgContainer*>(&(p->Children.back()), nodeChildren[CpuGpuMap[p]].back()));

    return 0;
}

template<typename T, typename S>
int MPGraph<T,S>::CopyMessageMemory()
{
    // first copy over nodes
    // specifically we need to copy over sum_c_r_c_p
    GpuMPNode* gpuNode;
    GpuMPNode hostNode(0,NULL,NULL,0,0);

    GpuMsgContainer* gpuContainer;
    GpuMsgContainer hostContainer(0, NULL, NULL, NULL);
 
    for(int i = 0; i < Graph.size(); i++)
    {
        gpuNode = CpuGpuMap[Graph[i]];

        // all we need to do for now is copy over sum_c_r_c_p
        // later, I think we'll allocate everything here
        gpuErrchk(cudaMemcpy(&hostNode, gpuNode, sizeof(hostNode), cudaMemcpyDeviceToHost));
        hostNode.sum_c_r_c_p = Graph[i]->sum_c_r_c_p;
        hostNode.GpuParents = NULL;
	hostNode.GpuChildren=NULL;

	gpuErrchk(cudaMalloc((void**)&hostNode.GpuParents, sizeof(hostContainer)*Graph[i].Parents.size()));
	gpuErrchk(cudaMalloc((void**)&hostNode.GpuChildren, sizeof(hostContainer)*Graph[i].Children.size()));

	gpuErrchk(cudaMemcpy(gpuNode, &hostNode, sizeof(hostNode), cudaMemcpyHostToDevice));



        // parents
        for(int j = 0; j < Graph[i]->Parents.size(); j++)
        {
            gpuContainer = CpuGpuMsgMap[&(Graph[i]->Parents[j])];
            gpuErrchk(cudaMemcpy(&hostContainer, gpuContainer, sizeof(hostContainer), cudaMemcpyDeviceToHost));

            hostContainer.lambda = Graph[i]->Parents[j].lambda;
            hostContainer.edge = CpuGpuEdgeMap[Graph[i]->Parents[j].edge];

            // fill out translator array
            gpuErrchk(cudaMalloc((void**)&(hostContainer.Translator), sizeof(S)* Graph[i]->Parents[j].Translator.size()));
            gpuErrchk(cudaMemcpy(hostContainer.Translator,&(Graph[i]->Parents[j].Translator[0]), sizeof(S)* Graph[i]->Parents[j].Translator.size(), cudaMemcpyHostToDevice));

            // copy back
            gpuErrchk(cudaMemcpy(gpuContainer, &hostContainer, sizeof(hostContainer), cudaMemcpyHostToDevice));

        }

        // children
        for(int j = 0; j < Graph[i]->Children.size(); j++)
        {
            gpuContainer = CpuGpuMsgMap[&(Graph[i]->Children[j])];
            gpuErrchk(cudaMemcpy(&hostContainer, gpuContainer, sizeof(hostContainer), cudaMemcpyDeviceToHost));

            hostContainer.lambda = Graph[i]->Children[j].lambda;
            hostContainer.edge = CpuGpuEdgeMap[Graph[i]->Children[j].edge];



            // fill out translator array
            gpuErrchk(cudaMalloc((void**)&(hostContainer.Translator), sizeof(S)* Graph[i]->Children[j].Translator.size()));
            gpuErrchk(cudaMemcpy(hostContainer.Translator,&(Graph[i]->Children[j].Translator[0]), sizeof(S)* Graph[i]->Children[j].Translator.size(), cudaMemcpyHostToDevice));

            // copy back
            gpuErrchk(cudaMemcpy(gpuContainer, &hostContainer, sizeof(hostContainer), cudaMemcpyHostToDevice));

        }
    }

    // copy over edges
    GpuEdgeID* gpuEdge;
    GpuEdgeID hostEdge {NULL, NULL, NULL, NULL, NULL, 0};
    for(int i = 0; i < Edges.size(); i++)
    {
        gpuEdge = CpuGpuEdgeMap[Edges[i]];
        gpuErrchk(cudaMemcpy(&hostEdge, gpuEdge, sizeof(hostEdge), cudaMemcpyDeviceToHost));

        hostEdge.newVarSize = Edges[i]->newVarSize;

        // now, allocate space for the three vectors
        gpuErrchk(cudaMalloc((void**)&(hostEdge.rStateMultipliers), sizeof(S)*Edges[i]->rStateMultipliers.size()));
        gpuErrchk(cudaMemcpy(hostEdge.rStateMultipliers, &(Edges[i]->rStateMultipliers[0]),sizeof(S)*Edges[i]->rStateMultipliers.size(), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc((void**)&hostEdge.newVarStateMultipliers, sizeof(S)*Edges[i]->newVarStateMultipliers.size()));
        gpuErrchk(cudaMemcpy(hostEdge.newVarStateMultipliers, &(Edges[i]->newVarStateMultipliers[0]),sizeof(S)*Edges[i]->newVarStateMultipliers.size(), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc((void**)&hostEdge.newVarIX, sizeof(S)*Edges[i]->newVarIX.size()));
        gpuErrchk(cudaMemcpy(hostEdge.newVarIX, &(Edges[i]->newVarIX[0]),sizeof(S)*Edges[i]->newVarIX.size(), cudaMemcpyHostToDevice));

        // copy over to GPU
        gpuErrchk(cudaMemcpy(gpuEdge, &hostEdge, sizeof(hostEdge), cudaMemcpyHostToDevice));

    }


    return 0;

}

template<typename T, typename S>
int MPGraph<T,S>::AllocateMessageMemory() {

    size_t lambdaOffset = 0;

    GpuEdgeID* gpuID = NULL;
    GpuEdgeID testID{NULL, NULL, NULL, NULL, NULL, 0};

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

                    gpuErrchk(cudaMalloc((void**)&gpuID, sizeof(GpuEdgeID)));
                    testID.parentPtr = CpuGpuMsgMap[&(*pn)];
                    testID.childPtr = CpuGpuMsgMap[&(*cn)];

                    gpuErrchk(cudaMemcpy(gpuID, &testID, sizeof(testID), cudaMemcpyHostToDevice));

                    CpuGpuEdgeMap.insert(std::pair<EdgeID*, GpuEdgeID*>(Edges.back(), gpuID));
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
const typename MPGraph<T,S>::DualWorkspaceID MPGraph<T,S>::AllocateDualWorkspaceMem(T epsilon) const {
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
void MPGraph<T,S>::DeAllocateDualWorkspaceMem(DualWorkspaceID& dw) const {
    delete[] dw.DualWorkspace;
}

template<typename T, typename S>
T* MPGraph<T,S>::GetMaxMemComputeMu(T epsilon) const {
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
    return ((maxMem > 0) ? new T[maxMem] : NULL);
}


template<typename T, typename S>
S* MPGraph<T,S>::GetMaxMemComputeMuIXVar() const {
    size_t maxMem = 0;
    for (size_t r = 0; r < ValidRegionMapping.size(); ++r) {
        MPNode* r_ptr = Graph[ValidRegionMapping[r]];
        for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
            size_t tmp = p->edge->newVarIX.size();
            maxMem = (tmp>maxMem) ? tmp : maxMem;
        }
    }

    return ((maxMem > 0) ? new S[maxMem] : NULL);
}




template<typename T, typename S>
const typename MPGraph<T,S>::RRegionWorkspaceID MPGraph<T,S>::AllocateReparameterizeRegionWorkspaceMem(T epsilon) const {
    size_t maxMem = 0;
    size_t maxIXMem = 0;
    for (size_t r = 0; r < ValidRegionMapping.size(); ++r) {
        MPNode* r_ptr = Graph[ValidRegionMapping[r]];
        size_t psz = r_ptr->Parents.size();
        maxMem = (psz>maxMem) ? psz : maxMem;
        size_t rvIX = r_ptr->varIX.size();
        maxIXMem = (rvIX>maxIXMem) ? rvIX : maxIXMem;
    }

    return RRegionWorkspaceID{ ((maxMem > 0) ? new T[maxMem] : NULL), GetMaxMemComputeMu(epsilon), GetMaxMemComputeMuIXVar(), ((maxIXMem > 0) ? new S[maxIXMem] : NULL) };
}



template<typename T, typename S>
void MPGraph<T,S>::DeAllocateReparameterizeRegionWorkspaceMem(RRegionWorkspaceID& w) const {
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
const typename MPGraph<T,S>::REdgeWorkspaceID MPGraph<T,S>::AllocateReparameterizeEdgeWorkspaceMem(T epsilon) const {
    size_t maxIXMem = 0;
    for(typename std::vector<EdgeID*>::const_iterator eb=Edges.begin(), eb_e=Edges.end();eb!=eb_e;++eb) {
        MPNode* r_ptr = (*eb)->childPtr->node;
        size_t rNumVar = r_ptr->varIX.size();
        maxIXMem = (rNumVar>maxIXMem) ? rNumVar : maxIXMem;
    }
    return REdgeWorkspaceID{ GetMaxMemComputeMu(epsilon), GetMaxMemComputeMuIXVar(), ((maxIXMem > 0) ? new S[maxIXMem] : NULL) };
}



template<typename T, typename S>
void MPGraph<T,S>::DeAllocateReparameterizeEdgeWorkspaceMem(REdgeWorkspaceID& w) const {
    delete[] w.MuMem;
    delete[] w.MuIXMem;
    delete[] w.IXMem;
    w.MuMem = NULL;
    w.MuIXMem = NULL;
    w.IXMem = NULL;
}



template<typename T, typename S>
const typename MPGraph<T,S>::GEdgeWorkspaceID MPGraph<T,S>::AllocateGradientEdgeWorkspaceMem() const {
    size_t memSZ = 0;
    for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
        MPNode* r_ptr = *r;
        size_t s_r_e = r_ptr->GetPotentialSize();
        memSZ = (s_r_e > memSZ) ? s_r_e : memSZ;
    }
    return GEdgeWorkspaceID{ ((memSZ>0) ? new T[memSZ] : NULL), ((memSZ>0) ? new T[memSZ] : NULL) };
}

template<typename T, typename S>
void MPGraph<T,S>::DeAllocateGradientEdgeWorkspaceMem(GEdgeWorkspaceID& w) const {
    delete[] w.mem1;
    delete[] w.mem2;
}

template<typename T, typename S>
const typename MPGraph<T,S>::FunctionUpdateWorkspaceID MPGraph<T,S>::AllocateFunctionUpdateWorkspaceMem() const {
    size_t memSZ = 0;
    for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
        MPNode* r_ptr = *r;
        size_t s_r_e = r_ptr->GetPotentialSize();
        memSZ = (s_r_e > memSZ) ? s_r_e : memSZ;
    }
    return FunctionUpdateWorkspaceID{ ((memSZ>0) ? new T[memSZ] : NULL)};
}

template<typename T, typename S>
void MPGraph<T,S>::DeAllocateFunctionUpdateWorkspaceID(FunctionUpdateWorkspaceID& w) const {
    delete[] w.mem;
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
size_t MPGraph<T,S>::NumberOfRegionsTotal() const {
    return Graph.size();
}

template<typename T, typename S>
size_t MPGraph<T,S>::NumberOfRegionsWithParents() const {
    return ValidRegionMapping.size();
}

template<typename T, typename S>
size_t MPGraph<T,S>::NumberOfEdges() const {
    return Edges.size();
}

template<typename T, typename S>
void MPGraph<T,S>::UpdateEdge(T* lambdaBase, T* lambdaGlobal, int e, bool additiveUpdate) {
    if (lambdaBase == lambdaGlobal) {
        assert(additiveUpdate == false);//change ReparameterizationEdge function to directly perform update and don't call UpdateEdge
        return;
    }

    EdgeID* edge = Edges[e];
    MPNode* r_ptr = edge->childPtr->node;
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
void MPGraph<T,S>::CopyLambda(T* lambdaSrc, T* lambdaDst, size_t s_r_e) const {
    //std::copy(lambdaSrc, lambdaSrc + s_r_e, lambdaDst);
    //memcpy((void*)(lambdaDst), (void*)(lambdaSrc), s_r_e*sizeof(T));
    for (T* ptr_e = lambdaSrc + s_r_e; ptr_e != lambdaSrc;) {
        *lambdaDst++ = *lambdaSrc++;
    }
}



template<typename T, typename S>
void MPGraph<T,S>::CopyMessagesForLocalFunction(T* lambdaSrc, T* lambdaDst, int r) const {
    MPNode* r_ptr = Graph[r];
    size_t s_r_e = r_ptr->GetPotentialSize();

    for (typename std::vector<MsgContainer>::const_iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn) {
        CopyLambda(lambdaSrc + pn->lambda, lambdaDst + pn->lambda, s_r_e);
    }

    for (typename std::vector<MsgContainer>::const_iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn) {
        size_t s_r_c = cn->node->GetPotentialSize();
        CopyLambda(lambdaSrc + cn->lambda, lambdaDst + cn->lambda, s_r_c);
    }
}

template<typename T, typename S>
void MPGraph<T,S>::ComputeLocalFunctionUpdate(T* lambdaBase, int r, T epsilon, T multiplier, bool additiveUpdate, FunctionUpdateWorkspaceID& w) {
    assert(additiveUpdate==true);
    MPNode* r_ptr = Graph[r];
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

    for (typename std::vector<MsgContainer>::iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
        for (size_t s_r = 0; s_r < s_r_e; ++s_r) {
            lambdaBase[p->lambda+s_r] = -mem[s_r];
        }
    }

    for (typename std::vector<MsgContainer>::const_iterator c = r_ptr->Children.begin(), c_e = r_ptr->Children.end(); c != c_e; ++c) {
        size_t s_r_c = c->node->GetPotentialSize();
        std::fill_n(lambdaBase+c->lambda, s_r_c, T(0));
        for(size_t s_r = 0;s_r < s_r_e;++s_r) {
            lambdaBase[c->lambda+c->Translator[s_r]] += mem[s_r];
        }
    }
}

template<typename T, typename S>
void MPGraph<T,S>::UpdateLocalFunction(T* lambdaBase, T* lambdaGlobal, int r, bool additiveUpdate) {
    if (lambdaBase == lambdaGlobal) {
        return;
    }
    MPNode* r_ptr = Graph[r];
    size_t s_r_e = r_ptr->GetPotentialSize();

    if (additiveUpdate) {
        for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
            for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
                lambdaGlobal[p->lambda + s_r] += lambdaBase[p->lambda + s_r];
            }
        }

        for (typename std::vector<MsgContainer>::const_iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn) {
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
void MPGraph<T,S>::CopyMessagesForEdge(T* lambdaSrc, T* lambdaDst, int e) const {
    EdgeID* edge = Edges[e];
    MPNode* r_ptr = edge->childPtr->node;
    MPNode* p_ptr = edge->parentPtr->node;

    size_t s_r_e = r_ptr->GetPotentialSize();
    size_t s_p_e = p_ptr->GetPotentialSize();

    for (typename std::vector<MsgContainer>::const_iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn) {
        CopyLambda(lambdaSrc + pn->lambda, lambdaDst + pn->lambda, s_r_e);
    }

    for (typename std::vector<MsgContainer>::const_iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn) {
        size_t s_r_c = cn->node->GetPotentialSize();
        CopyLambda(lambdaSrc + cn->lambda, lambdaDst + cn->lambda, s_r_c);
    }

    for (typename std::vector<MsgContainer>::const_iterator p_hat = p_ptr->Parents.begin(), p_hat_e = p_ptr->Parents.end(); p_hat != p_hat_e; ++p_hat) {
        CopyLambda(lambdaSrc + p_hat->lambda, lambdaDst + p_hat->lambda, s_p_e);
    }

    for (typename std::vector<MsgContainer>::const_iterator c_hat = p_ptr->Children.begin(), c_hat_e = p_ptr->Children.end(); c_hat != c_hat_e; ++c_hat) {
        if (c_hat->node != r_ptr) {
            size_t s_r_pc = c_hat->node->GetPotentialSize();
            CopyLambda(lambdaSrc + c_hat->lambda, lambdaDst + c_hat->lambda, s_r_pc);
        }
    }
}


template<typename T, typename S>
void MPGraph<T,S>::CopyMessagesForStar(T* lambdaSrc, T* lambdaDst, int r) const {
    MPNode* r_ptr = Graph[ValidRegionMapping[r]];
    size_t s_r_e = r_ptr->GetPotentialSize();

    for (typename std::vector<MsgContainer>::const_iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn) {
        CopyLambda(lambdaSrc + pn->lambda, lambdaDst + pn->lambda, s_r_e);

        MPNode* p_ptr = pn->node;
        size_t s_p_e = p_ptr->GetPotentialSize();
        for (typename std::vector<MsgContainer>::const_iterator p_hat = p_ptr->Parents.begin(), p_hat_e = p_ptr->Parents.end(); p_hat != p_hat_e; ++p_hat) {
            CopyLambda(lambdaSrc + p_hat->lambda, lambdaDst + p_hat->lambda, s_p_e);
        }

        for (typename std::vector<MsgContainer>::const_iterator c_hat = p_ptr->Children.begin(), c_hat_e = p_ptr->Children.end(); c_hat != c_hat_e; ++c_hat) {
            if (c_hat->node != r_ptr) {
                size_t s_pc_e = c_hat->node->GetPotentialSize();
                CopyLambda(lambdaSrc + c_hat->lambda, lambdaDst + c_hat->lambda, s_pc_e);
            }
        }
    }

    for (typename std::vector<MsgContainer>::const_iterator cn = r_ptr->Children.begin(), cn_e = r_ptr->Children.end(); cn != cn_e; ++cn) {
        size_t s_r_c = cn->node->GetPotentialSize();
        CopyLambda(lambdaSrc + cn->lambda, lambdaDst + cn->lambda, s_r_c);
    }
}

template<typename T, typename S>
void MPGraph<T,S>::ReparameterizeEdge(T* lambdaBase, int e, T epsilon, bool additiveUpdate, REdgeWorkspaceID& wspace) {
    EdgeID* edge = Edges[e];
    MPNode* r_ptr = edge->childPtr->node;
    MPNode* p_ptr = edge->parentPtr->node;

    size_t s_r_e = r_ptr->GetPotentialSize();

    T c_p = p_ptr->c_r;
    T c_r = r_ptr->c_r;
    T frac = T(1) / (c_p + c_r);

    size_t rNumVar = r_ptr->varIX.size();
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
            if (indivVarStates[varIX] == Cardinalities[r_ptr->varIX[varIX]]) {
                indivVarStates[varIX] = 0;
            } else {
                break;
            }
        }
    }
}


template<typename T, typename S>
T MPGraph<T,S>::ComputeMu(T* lambdaBase, EdgeID* edge, S* indivVarStates, size_t numVarsOverlap, T epsilon, T* workspaceMem, S* MuIXMem) {
    MPNode* r_ptr = edge->childPtr->node;
    MPNode* p_ptr = edge->parentPtr->node;

    //size_t numVarsOverlap = indivVarStates.size();
    size_t s_p_stat = 0;
    for (size_t k = 0; k<numVarsOverlap; ++k) {
        s_p_stat += indivVarStates[k]*edge->rStateMultipliers[k];
    }

    //size_t s_p_e = edge->newVarCumSize.back();
    size_t s_p_e = edge->newVarSize;
    size_t numVarNew = edge->newVarIX.size();

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
    for (size_t s_p = 0; s_p != s_p_e; ++s_p) {
        //size_t s_p_real = s_p_stat;
        //for (size_t varIX = 0; varIX<numVarNew; ++varIX) {
        //	s_p_real += indivNewVarStates[varIX]*edge->newVarStateMultipliers[varIX];
        //}

        T buf = (p_ptr->pot == NULL) ? T(0) : p_ptr->pot[s_p_real];

        for (typename std::vector<MsgContainer>::const_iterator p_hat = p_ptr->Parents.begin(), p_hat_e = p_ptr->Parents.end(); p_hat != p_hat_e; ++p_hat) {
            buf -= lambdaBase[p_hat->lambda+s_p_real];
        }

        for (typename std::vector<MsgContainer>::const_iterator c_hat = p_ptr->Children.begin(), c_hat_e = p_ptr->Children.end(); c_hat != c_hat_e; ++c_hat) {
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
            if (indivNewVarStates[varIX] == Cardinalities[edge->newVarIX[varIX]]) {
                indivNewVarStates[varIX] = 0;
                s_p_real -= (Cardinalities[edge->newVarIX[varIX]]-1)*edge->newVarStateMultipliers[varIX];
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
void MPGraph<T,S>::UpdateRegion(T* lambdaBase, T* lambdaGlobal, int r, bool additiveUpdate) {
    if (lambdaBase == lambdaGlobal) {
        return;
    }
    MPNode* r_ptr = Graph[ValidRegionMapping[r]];

    size_t s_r_e = r_ptr->GetPotentialSize();
    if (additiveUpdate) {
        for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
            for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
                lambdaGlobal[p->lambda + s_r] += lambdaBase[p->lambda + s_r];
            }
        }
    } else {
        for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
            for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
                lambdaGlobal[p->lambda + s_r] = lambdaBase[p->lambda + s_r];
            }
        }
    }
}

template<typename T, typename S>
void MPGraph<T,S>::ReparameterizeRegion(T* lambdaBase, int r, T epsilon, bool additiveUpdate, RRegionWorkspaceID& wspace) {
    MPNode* r_ptr = Graph[ValidRegionMapping[r]];

    size_t ParentLocalIX;
    T* mu_p_r = wspace.RRegionMem;

    T sum_c_p = r_ptr->sum_c_r_c_p;
    //T sum_c_p = r_ptr->c_r;
    //for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
    //	sum_c_p += p->node->c_r;
    //}

    size_t s_r_e = r_ptr->GetPotentialSize();
    size_t rNumVar = r_ptr->varIX.size();
    //std::vector<S> indivVarStates(rNumVar, 0);
    S* indivVarStates = wspace.IXMem;
    for(S *tmp=indivVarStates, *tmp_e=indivVarStates+rNumVar;tmp!=tmp_e;++tmp) {
        *tmp = 0;
    }
    for (size_t s_r = 0; s_r != s_r_e; ++s_r) {
        T phi_r_x_r_prime = (r_ptr->pot == NULL) ? T(0) : r_ptr->pot[s_r];

        for (typename std::vector<MsgContainer>::const_iterator c = r_ptr->Children.begin(), c_e = r_ptr->Children.end(); c != c_e; ++c) {
            phi_r_x_r_prime += lambdaBase[c->lambda+c->Translator[s_r]];
        }

        ParentLocalIX = 0;
        for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p, ++ParentLocalIX) {
            mu_p_r[ParentLocalIX] = ComputeMu(lambdaBase, p->edge, indivVarStates, rNumVar, epsilon, wspace.MuMem, wspace.MuIXMem);
            phi_r_x_r_prime += mu_p_r[ParentLocalIX];
        }

        phi_r_x_r_prime /= sum_c_p;

        ParentLocalIX = 0;
        for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p, ++ParentLocalIX) {
            MPNode* ptr = p->node;//ptr points on parent, i.e., ptr->c_r = c_p!!!
            T value = ptr->c_r*phi_r_x_r_prime - mu_p_r[ParentLocalIX];
            lambdaBase[p->lambda+s_r] = ((additiveUpdate)?value-lambdaBase[p->lambda+s_r]:value);//the employed normalization is commutative
        }

        for (size_t varIX = 0; varIX < rNumVar; ++varIX) {
            ++indivVarStates[varIX];
            if (indivVarStates[varIX] == Cardinalities[r_ptr->varIX[varIX]]) {
                indivVarStates[varIX] = 0;
            } else {
                break;
            }
        }
    }

    for (typename std::vector<MsgContainer>::const_iterator p = r_ptr->Parents.begin(), p_e = r_ptr->Parents.end(); p != p_e; ++p) {
        for (size_t s_r = s_r_e - 1; s_r != 0; --s_r) {
            lambdaBase[p->lambda+s_r] -= lambdaBase[p->lambda];
        }
        lambdaBase[p->lambda] = 0;
    }
}

template<typename T, typename S>
T MPGraph<T,S>::ComputeReparameterizationPotential(T* lambdaBase, const MPNode* const r_ptr, const S s_r) const {
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
T MPGraph<T,S>::ComputeDual(T* lambdaBase, T epsilon, DualWorkspaceID& dw) const {
    T dual = T(0);

    for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
        MPNode* r_ptr = *r;
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
size_t MPGraph<T,S>::GetLambdaSize() const {
    return LambdaSize;
}

template<typename T, typename S>
void MPGraph<T,S>::GradientUpdateEdge(T* lambdaBase, int e, T epsilon, T stepSize, bool additiveUpdate, GEdgeWorkspaceID& gew) {
    EdgeID* edge = Edges[e];
    MPNode* r_ptr = edge->childPtr->node;
    MPNode* p_ptr = edge->parentPtr->node;

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
void MPGraph<T,S>::ComputeBeliefForRegion(MPNode* r_ptr, T* lambdaBase, T epsilon, T* mem, size_t s_r_e) {
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
size_t MPGraph<T,S>::ComputeBeliefs(T* lambdaBase, T epsilon, T** belPtr, bool OnlyUnaries) {
    size_t BeliefSize = 0;
    for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
        MPNode* r_ptr = *r;
        size_t s_r_e = r_ptr->GetPotentialSize();
        size_t numVars = r_ptr->varIX.size();
        if ((OnlyUnaries&&numVars == 1) || !OnlyUnaries) {
            BeliefSize += s_r_e;
        }
    }
    T* beliefs = new T[BeliefSize];

    T* mem = beliefs;
    if (belPtr != NULL) {
        *belPtr = beliefs;
    }

    for (typename std::vector<MPNode*>::const_iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
        MPNode* r_ptr = *r;
        size_t numVars = r_ptr->varIX.size();
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
void MPGraph<T,S>::Marginalize(T* curBel, T* oldBel, EdgeID* edge, const std::vector<S>& indivVarStates, T& marg_new, T& marg_old) {
    //MPNode* r_ptr = edge->childPtr->node;
    //MPNode* p_ptr = edge->parentPtr->node;

    size_t numVarsOverlap = indivVarStates.size();
    size_t s_p_stat = 0;
    for (size_t k = 0; k<numVarsOverlap; ++k) {
        s_p_stat += indivVarStates[k]*edge->rStateMultipliers[k];
    }

    size_t s_p_e = edge->newVarSize;
    size_t numVarNew = edge->newVarIX.size();

    //individual vars;
    std::vector<S> indivNewVarStates(numVarNew, 0);
    for (size_t s_p = 0; s_p != s_p_e; ++s_p) {
        size_t s_p_real = s_p_stat;
        for (size_t varIX = 0; varIX<numVarNew; ++varIX) {
            s_p_real += indivNewVarStates[varIX]*edge->newVarStateMultipliers[varIX];
        }

        marg_new += curBel[s_p_real];
        marg_old += oldBel[s_p_real];

        for (size_t varIX = 0; varIX < numVarNew; ++varIX) {
            ++indivNewVarStates[varIX];
            if (indivNewVarStates[varIX] == Cardinalities[edge->newVarIX[varIX]]) {
                indivNewVarStates[varIX] = 0;
            } else {
                break;
            }
        }
    }
}

template<typename T, typename S>
T MPGraph<T,S>::ComputeImprovement(T* curBel, T* oldBel) {//not efficient
    T imp = T(0);

    for (typename std::vector<MPNode*>::iterator r = Graph.begin(), r_e = Graph.end(); r != r_e; ++r) {
        MPNode* r_ptr = *r;
        size_t s_r_e = r_ptr->GetPotentialSize();
        size_t belIX_r = ((T*)r_ptr->tmp) - curBel;

        for (typename std::vector<MsgContainer>::const_iterator pn = r_ptr->Parents.begin(), pn_e = r_ptr->Parents.end(); pn != pn_e; ++pn) {
            MPNode* p_ptr = pn->node;
            size_t belIX_p = ((T*)p_ptr->tmp) - curBel;

            T v1 = 0;
            T v2 = 0;
            size_t rNumVar = r_ptr->varIX.size();
            std::vector<S> indivVarStates(rNumVar, 0);
            for(size_t s_r=0;s_r<s_r_e;++s_r) {
                T marg_old = T(0);
                T marg_new = T(0);
                Marginalize(curBel + belIX_p, oldBel + belIX_p, pn->edge, indivVarStates, marg_new, marg_old);

                v1 += curBel[belIX_r+s_r]*std::sqrt(marg_old/oldBel[belIX_r+s_r]);
                v2 += marg_new*std::sqrt(oldBel[belIX_r+s_r]/marg_old);

                for (size_t varIX = 0; varIX < rNumVar; ++varIX) {
                    ++indivVarStates[varIX];
                    if (indivVarStates[varIX] == Cardinalities[r_ptr->varIX[varIX]]) {
                        indivVarStates[varIX] = 0;
                    } else {
                        break;
                    }
                }
            }

            imp += std::log(v1) + std::log(v2);
        }
    }

    return imp;
}

template<typename T, typename S>
void MPGraph<T,S>::DeleteBeliefs() {
    MPNode* r_ptr = *Graph.begin();
    delete[]((float*)r_ptr->tmp);
}


template<typename T, typename S>
ThreadSync<T,S>::ThreadSync(int nT, T* lambdaGlobal, T epsilon, MPGraph<T, S>* g) : state(NONE), numThreads(nT), lambdaGlobal(lambdaGlobal), epsilon(epsilon), g(g), currentlyStoppedThreads(0), prevDual(std::numeric_limits<T>::max())  {
    dw = g->AllocateDualWorkspaceMem(epsilon);
    state = INIT;
    LambdaForNoSync.assign(g->GetLambdaSize(),0);
}

template<typename T, typename S>
ThreadSync<T,S>::~ThreadSync() {
    g->DeAllocateDualWorkspaceMem(dw);
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
            T dualVal = g->ComputeDual(lambdaGlobal, epsilon, dw);
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
    std::cout << "line 1016" << std::endl;
    double timeMS = CTmr.Stop()*1000.;
    std::cout << "line 1018" << std::endl;
    std::copy(lambdaGlobal, lambdaGlobal+LambdaForNoSync.size(), &LambdaForNoSync[0]);
    std::cout << "line 1020" << std::endl;
    T dualVal = g->ComputeDual(&LambdaForNoSync[0], epsilon, dw);
    std::cout << timeMS <<"; " << CTmr1.Stop()*1000. << "; " << dualVal << std::endl;
}

template<typename T, typename S>
void ThreadSync<T,S>::CudaComputeDualNoSync() {
    std::cout << "line 1016" << std::endl;
    double timeMS = CTmr.Stop()*1000.;
    std::cout << "line 1018" << std::endl;
    std::copy(lambdaGlobal, lambdaGlobal+LambdaForNoSync.size(), &LambdaForNoSync[0]);
    std::cout << "line 1020" << std::endl;
    T dualVal = g->ComputeDual(&LambdaForNoSync[0], epsilon, dw);
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
    msgSize = g->GetLambdaSize();
    assert(msgSize > 0);
    lambdaLocal.assign(msgSize, T(0));
    lambdaBase = &lambdaLocal[0];

    //REdgeWorkspaceID* test;

    // see if cuda works


    #if WHICH_FUNC==1
        rrw = g->AllocateReparameterizeRegionWorkspaceMem(epsilon);
        uid = new std::uniform_int_distribution<int>(0, g->NumberOfRegionsWithParents() - 1);
    #elif WHICH_FUNC==2
        rew = g->AllocateReparameterizeEdgeWorkspaceMem(epsilon);
        uid = new std::uniform_int_distribution<int>(0, g->NumberOfEdges() - 1);
    #elif WHICH_FUNC==3
        fw = g->AllocateFunctionUpdateWorkspaceMem();
        uid = new std::uniform_int_distribution<int>(0, g->NumberOfRegionsTotal()-1);
    #endif

    eng.seed(randomSeed);
}

template<typename T, typename S>
ThreadWorker<T,S>::~ThreadWorker() {
#if WHICH_FUNC==1
    g->DeAllocateReparameterizeRegionWorkspaceMem(rrw);
#elif WHICH_FUNC==2
    g->DeAllocateReparameterizeEdgeWorkspaceMem(rew);
#elif WHICH_FUNC==3
    g->DeAllocateFunctionUpdateWorkspaceID(fw);
#endif
    if (uid != NULL) {
        delete uid;
        uid = NULL;
    }
    if (thread_ != NULL) {
        delete thread_;
        thread_ = NULL;
    }
}

template<typename T, typename S>
void ThreadWorker<T,S>::start() {
    thread_ = new std::thread(std::bind(&ThreadWorker::run, this));
    thread_->detach();
}

template<typename T, typename S>
size_t ThreadWorker<T,S>::GetCount() {
    return cnt;
}

template<typename T, typename S>
void ThreadWorker<T,S>::run() {
    std::cout << "Thread started" << std::endl;
    int test = 0;
    // ts->startFunc();
    while (ts->checkSync()) {

        #if WHICH_FUNC==1
                int ix = (*uid)(eng);
                g->CopyMessagesForStar(lambdaGlobal, lambdaBase, ix);
                g->ReparameterizeRegion(lambdaBase, ix, epsilon, false, rrw);
                g->UpdateRegion(lambdaBase, lambdaGlobal, ix, false);
        #elif WHICH_FUNC==2
                int ix = (*uid)(eng);
                g->CopyMessagesForEdge(lambdaGlobal, lambdaBase, ix);
                g->ReparameterizeEdge(lambdaBase, ix, epsilon, false, rew);
                g->UpdateEdge(lambdaBase, lambdaGlobal, ix, false);
        #elif WHICH_FUNC==3
                int ix = (*uid)(eng);
                g->CopyMessagesForLocalFunction(lambdaGlobal, lambdaBase, ix);
                g->ComputeLocalFunctionUpdate(lambdaBase, ix, epsilon, *stepsize, true, fw);
                g->UpdateLocalFunction(lambdaBase, lambdaGlobal, ix, true);

        #endif

        ++cnt;
    }
}


template<typename T, typename S>
int AsyncRMPThread<T,S>::RunMP(MPGraph<T, S>& g, T epsilon, int numIterations, int numThreads, int WaitTimeInMS) {
    size_t msgSize = g.GetLambdaSize();
    if (msgSize == 0) {
        typename MPGraph<T, S>::DualWorkspaceID dw = g.AllocateDualWorkspaceMem(epsilon);
        std::cout << "0: " << g.ComputeDual(NULL, epsilon, dw) << std::endl;
        g.DeAllocateDualWorkspaceMem(dw);
        return 0;
    }

    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

    lambdaGlobal.assign(msgSize, T(0));
    T* lambdaGlob = &lambdaGlobal[0];


    // thread syncs are dumb - they're used for keeping track of
    // the state of each thread, which is just totally unncessary.
    ThreadSync<T, S> sy(numThreads, lambdaGlob, epsilon, &g);

    T stepsize = -0.1;
    std::vector<ThreadWorker<T, S>*> ex(numThreads, NULL);
    for (int k = 0; k < numThreads; ++k) {
        ex[k] = new ThreadWorker<T, S>(&sy, &g, epsilon, k, lambdaGlob, &stepsize);
    }
    for (int k = 0; k < numThreads; ++k) {
        ex[k]->start();
    }

    while (!sy.startFunc()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    for (int k = 0; k < numIterations; ++k)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(WaitTimeInMS));
        //sy.interruptFunc();
        sy.ComputeDualNoSync();
    }
    sy.terminateFunc();

    size_t regionUpdates = 0;
    for(int k=0;k<numThreads;++k) {
        size_t tmp = ex[k]->GetCount();
        std::cout << "Thread " << k << ": " << tmp << std::endl;
        regionUpdates += tmp;
        delete ex[k];
    }
    std::cout << "Region updates: " << regionUpdates << std::endl;
    std::cout << "Total regions:  " << g.NumberOfRegionsWithParents() << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::cout << "Terminating program." << std::endl;
    return 0;
}

template<typename T, typename S>
size_t AsyncRMPThread<T,S>::GetBeliefs(MPGraph<T, S>& g, T epsilon, T** belPtr, bool OnlyUnaries) {
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
    return size_t(-1);
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
    size_t msgSize = g.GetLambdaSize();
    if (msgSize == 0) {
        typename MPGraph<T, S>::DualWorkspaceID dw = g.AllocateDualWorkspaceMem(epsilon);
        std::cout << "0: " << g.ComputeDual(NULL, epsilon, dw) << std::endl;
        g.DeAllocateDualWorkspaceMem(dw);
        return 0;
    }

    std::vector<T> lambdaGlobal(msgSize, T(0));

    typename MPGraph<T, S>::DualWorkspaceID dw = g.AllocateDualWorkspaceMem(epsilon);
    typename MPGraph<T, S>::RRegionWorkspaceID rrw = g.AllocateReparameterizeRegionWorkspaceMem(epsilon);
    for (int iter = 0; iter < 20; ++iter) {
        for (int k = 0; k < int(g.NumberOfRegionsWithParents()); ++k) {
            g.ReparameterizeRegion(&lambdaGlobal[0], k, epsilon, false, rrw);
        }
        std::cout << iter << ": " << g.ComputeDual(&lambdaGlobal[0], epsilon, dw) << std::endl;
    }
    g.DeAllocateReparameterizeRegionWorkspaceMem(rrw);
    g.DeAllocateDualWorkspaceMem(dw);
    return 0;
}
