#ifndef SOFA_GPU_CUDA_CUDASUBSETMAPPING_H
#define SOFA_GPU_CUDA_CUDASUBSETMAPPING_H

#include "CudaTypes.h"
#include <sofa/component/mapping/SubsetMapping.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>

namespace sofa
{

namespace component
{

namespace mapping
{

template <>
class SubsetMappingInternalData<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>
{
public:
    sofa::gpu::cuda::CudaVector<int> map;
    int maxNOut;
    sofa::gpu::cuda::CudaVector<int> mapT;
    SubsetMappingInternalData() : maxNOut(0)
    {}

    void clear(int reserve=0)
    {
        map.clear();
        map.reserve(reserve);
    }

    int addPoint(int fromIndex)
    {
        int i = map.size();
        map.resize(i+1);
        map[i] = fromIndex;
        return i;
    }

    void init(int insize)
    {
        if (map.empty()) return;
        // compute mapT
        std::vector<int> nout(insize);
        for (unsigned int i=0; i<map.size(); i++)
            nout[map[i]]++;
        for (int i=0; i<insize; i++)
            if (nout[i] > maxNOut) maxNOut = nout[i];
        if (maxNOut <= 1)
        {
            // at most one duplicated points per input. mapT is not necessary
            mapT.clear();
        }
        else
        {
            int nbloc = (insize+BSIZE-1)/BSIZE;
            std::cout << "CudaSubsetMapping: mapT with "<<maxNOut<<" entries per DOF and "<<nbloc<<" blocs."<<std::endl;
            mapT.resize(nbloc*(BSIZE*maxNOut));
            for (unsigned int i=0; i<mapT.size(); i++)
                mapT[i] = -1;
            nout.clear();
            nout.resize(insize);
            for (unsigned int i=0; i<map.size(); i++)
            {
                int index = map[i];
                int num = nout[index]++;
                int b = (index / BSIZE); index -= b*BSIZE;
                mapT[(maxNOut*b+num)*BSIZE+index] = i;
            }
        }
    }
};

template <>
void SubsetMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::postInit();

template <>
void SubsetMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in );

template <>
void SubsetMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );

template <>
void SubsetMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );

template <>
void SubsetMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::postInit();

template <>
void SubsetMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in );

template <>
void SubsetMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
