#ifndef SOFA_GPU_CUDA_CUDATETRAHEDRONFEMFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDATETRAHEDRONFEMFORCEFIELD_INL

#include "CudaTetrahedronFEMForceField.h"
#include <sofa/component/forcefield/TetrahedronFEMForceField.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void TetrahedronFEMForceFieldCuda3f_addForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, const void* velems, void* f, const void* x, const void* v);
    void TetrahedronFEMForceFieldCuda3f_addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, const void* velems, void* df, const void* dx);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

using namespace gpu::cuda;

template <>
void TetrahedronFEMForceField<CudaVec3fTypes>::reinit()
{
    _strainDisplacements.resize( _indexedElements->size() );
    _materialsStiffnesses.resize(_indexedElements->size() );

    const VecElement& elems = *_indexedElements;

    _rotations.resize( elems.size() );
    _rotatedInitialElements.resize(elems.size());

    for (unsigned int i=0; i<elems.size(); i++)
    {
        Index a = elems[i][0];
        Index b = elems[i][1];
        Index c = elems[i][2];
        Index d = elems[i][3];
        computeMaterialStiffness(i,a,b,c,d);
        initLarge(i,a,b,c,d);
    }

    std::map<int,int> nelems;
    for (unsigned int i=0; i<elems.size(); i++)
    {
        const Element& e = elems[i];
        for (unsigned int j=0; j<e.size(); j++)
            ++nelems[e[j]];
    }
    int nmax = 0;
    for (std::map<int,int>::const_iterator it = nelems.begin(); it != nelems.end(); ++it)
        if (it->second > nmax)
            nmax = it->second;
    int v0 = 0;
    int nbv = 0;
    if (!nelems.empty())
    {
        v0 = nelems.begin()->first;
        nbv = nelems.rbegin()->first - v0 + 1;
    }
    data.init(elems.size(), v0, nbv, nmax);

    nelems.clear();
    for (unsigned int i=0; i<elems.size(); i++)
    {
        const Element& e = elems[i];
        const Coord& a = _rotatedInitialElements[i][0];
        const Coord& b = _rotatedInitialElements[i][1];
        const Coord& c = _rotatedInitialElements[i][2];
        const Coord& d = _rotatedInitialElements[i][3];
        data.setE(i, e, a, b, c, d, _materialsStiffnesses[i], _strainDisplacements[i]);
        for (unsigned int j=0; j<e.size(); j++)
            data.setV(e[j], nelems[e[j]]++, i*e.size()+j);
    }
}

template <>
void TetrahedronFEMForceField<gpu::cuda::CudaVec3fTypes>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    f.resize(x.size());
    TetrahedronFEMForceFieldCuda3f_addForce(
        data.elems.size(),
        data.nbVertex,
        data.nbElementPerVertex,
        data.elems.deviceRead(),
        data.state.deviceWrite(),
        data.velems.deviceRead(),
        (      Deriv*)f.deviceWrite() + data.vertex0,
        (const Coord*)x.deviceRead()  + data.vertex0,
        (const Deriv*)v.deviceRead()  + data.vertex0);
}

template <>
void TetrahedronFEMForceField<gpu::cuda::CudaVec3fTypes>::addDForce (VecDeriv& df, const VecDeriv& dx)
{
    df.resize(dx.size());
    TetrahedronFEMForceFieldCuda3f_addDForce(
        data.elems.size(),
        data.nbVertex,
        data.nbElementPerVertex,
        data.elems.deviceRead(),
        data.state.deviceWrite(),
        data.velems.deviceRead(),
        (      Deriv*)df.deviceWrite() + data.vertex0,
        (const Deriv*)dx.deviceRead()  + data.vertex0);
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
