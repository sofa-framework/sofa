#ifndef SOFA_GPU_CUDA_CUDASUBSETMAPPING_INL
#define SOFA_GPU_CUDA_CUDASUBSETMAPPING_INL

#include "CudaSubsetMapping.h"
#include <sofa/component/mapping/SubsetMapping.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void SubsetMappingCuda3f_apply(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f_applyJ(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in);
    void SubsetMappingCuda3f_applyJT1(unsigned int size, const void* map, void* out, const void* in);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace mapping
{

using namespace gpu::cuda;
/*
template <>
void SubsetMappingInternalData<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::clear(int reserve)
{
    map.clear();
    map.reserve(reserve);
}

template <>
int SubsetMappingInternalData<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::addPoint(int fromIndex)
{
    int i = map.size();
    map.resize(i+1);
    map[i] = fromIndex;
    return i;
}

template <>
void SubsetMappingInternalData<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::init(int insize)
{
    if (map.empty()) return;
    // compute mapT
    std::vector<int> nout(insize);
    for (unsigned int i=0;i<map.size();i++)
        nout[map[i]]++;
    for (unsigned int i=0;i<insize;i++)
        if (nout[i] > maxNOut) maxNOut = nout[i];
    int nbloc = (insize+BSIZE-1)/BSIZE;
    std::cout << "CudaSubsetMapping: mapT with "<<maxNOut<<" entries per DOF and "<<nbloc<<" blocs."<<std::endl;
    mapT.resize(nbloc*(BSIZE*maxNOut));
    for (unsigned int i=0;i<mapT.size();i++)
        mapT[i] = -1;
    nout.clear();
    nout.resize(insize);
    for (unsigned int i=0;i<map.size();i++)
    {
        int index = map[i].in_index;
        int num = nout[index]++;
        int b = (index / BSIZE); index -= b*BSIZE;
        mapT[(maxNOut*b+num)*BSIZE+index] = i;
    }
}
*/
template <>
void SubsetMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::postInit()
{
    const IndexArray& indices = this->f_indices.getValue();
    if (!indices.empty())
    {
        this->data.clear(indices.size());
        for (unsigned int i=0; i<indices.size(); i++)
            this->data.addPoint(indices[i]);
        this->data.init(this->fromModel->getX()->size());
    }
}

template <>
void SubsetMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::postInit()
{
    const IndexArray& indices = this->f_indices.getValue();
    if (!indices.empty())
    {
        this->data.clear(indices.size());
        for (unsigned int i=0; i<indices.size(); i++)
            this->data.addPoint(indices[i]);
        this->data.init(this->fromModel->getX()->size());
    }
}

template <>
void SubsetMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(data.map.size());
    SubsetMappingCuda3f_apply(data.map.size(), data.map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template <>
void SubsetMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(data.map.size());
    SubsetMappingCuda3f_applyJ(data.map.size(), data.map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template <>
void SubsetMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    if (data.map.size() == 0) return;
    unsigned int insize = out.size();
    if (data.mapT.empty())
        SubsetMappingCuda3f_applyJT1(data.map.size(), data.map.deviceRead(), out.deviceWrite(), in.deviceRead());
    else
        SubsetMappingCuda3f_applyJT(insize, data.maxNOut, data.mapT.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template <>
void SubsetMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(data.map.size());
    SubsetMappingCuda3f_apply(data.map.size(), data.map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template <>
void SubsetMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(data.map.size());
    SubsetMappingCuda3f_apply(data.map.size(), data.map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
