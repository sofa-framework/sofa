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

    std::vector<int> activeElems;
    for (unsigned int i=0; i<elems.size(); i++)
    {
        if (!_trimgrid || _trimgrid->isCubeActive(i/6))
        {
            activeElems.push_back(i);
        }
    }

    for (unsigned int i=0; i<activeElems.size(); i++)
    {
        int ei = activeElems[i];
        Index a = elems[ei][0];
        Index b = elems[ei][1];
        Index c = elems[ei][2];
        Index d = elems[ei][3];
        computeMaterialStiffness(ei,a,b,c,d);
        initLarge(ei,a,b,c,d);
    }

    std::map<int,int> nelems;
    for (unsigned int i=0; i<activeElems.size(); i++)
    {
        int ei = activeElems[i];
        const Element& e = elems[ei];
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
    data.init(activeElems.size(), v0, nbv, nmax);

    nelems.clear();
    for (unsigned int i=0; i<activeElems.size(); i++)
    {
        int ei = activeElems[i];
        const Element& e = elems[ei];
        const Coord& a = _rotatedInitialElements[ei][0];
        const Coord& b = _rotatedInitialElements[ei][1];
        const Coord& c = _rotatedInitialElements[ei][2];
        const Coord& d = _rotatedInitialElements[ei][3];
        data.setE(i, e, a, b, c, d, _materialsStiffnesses[ei], _strainDisplacements[ei]);
        for (unsigned int j=0; j<e.size(); j++)
            data.setV(e[j], nelems[e[j]]++, i*e.size()+j);
    }
}

template <>
void TetrahedronFEMForceField<gpu::cuda::CudaVec3fTypes>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    // Count active cubes in topology
    if (_trimgrid)
    {
        int nactive = 0;
        int ncubes = _trimgrid->getNbCubes();
        for (int i=0; i<ncubes; i++)
            if (_trimgrid->isCubeActive(i)) ++nactive;
        if (data.elems.size() != 6*nactive)
            reinit();
    }


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

#if 0
    // compare with CPU version

    const VecElement& elems = *_indexedElements;
    for (unsigned int i=0; i<elems.size(); i++)
    {
        Index a = elems[i][0];
        Index b = elems[i][1];
        Index c = elems[i][2];
        Index d = elems[i][3];
        Transformation Rt;
        computeRotationLarge(Rt, x, a, b, c);
        const TetrahedronFEMForceFieldInternalData<gpu::cuda::CudaVec3fTypes>::GPUElementState& s = data.state[i];
        const TetrahedronFEMForceFieldInternalData<gpu::cuda::CudaVec3fTypes>::GPUElement& e = data.elems[i];
        Mat3x3f Rdiff = Rt-s.Rt;
        if ((Rdiff[0].norm2()+Rdiff[1].norm2()+Rdiff[2].norm2()) > 0.000001f)
        {
            std::cout << "CPU Rt "<<i<<" = "<<Rt<<std::endl;
            std::cout << "GPU Rt "<<i<<" = "<<s.Rt<<std::endl;
            std::cout << "DIFF   "<<i<<" = "<<Rdiff<<std::endl;
        }
        Coord xb = Rt*(x[b]-x[a]);
        Coord xc = Rt*(x[c]-x[a]);
        Coord xd = Rt*(x[d]-x[a]);

        Displacement D;
        D[0] = 0;
        D[1] = 0;
        D[2] = 0;
        D[3] = _rotatedInitialElements[i][1][0] - xb[0];
        D[4] = _rotatedInitialElements[i][1][1] - xb[1];
        D[5] = _rotatedInitialElements[i][1][2] - xb[2];
        D[6] = _rotatedInitialElements[i][2][0] - xc[0];
        D[7] = _rotatedInitialElements[i][2][1] - xc[1];
        D[8] = _rotatedInitialElements[i][2][2] - xc[2];
        D[9] = _rotatedInitialElements[i][3][0] - xd[0];
        D[10]= _rotatedInitialElements[i][3][1] - xd[1];
        D[11]= _rotatedInitialElements[i][3][2] - xd[2];
        Vec<6,float> S = -((_materialsStiffnesses[i]) * ((_strainDisplacements[i]).multTranspose(D)))*(e.bx);

        Vec<6,float> Sdiff = S-s.S;

        if (Sdiff.norm2() > 0.0001f)
        {
            std::cout << "    D "<<i<<" = "<<D<<std::endl;
            std::cout << "CPU S "<<i<<" = "<<S<<std::endl;
            std::cout << "GPU S "<<i<<" = "<<s.S<<std::endl;
            std::cout << "DIFF   "<<i<<" = "<<Sdiff<<std::endl;
        }

    }
#endif

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
