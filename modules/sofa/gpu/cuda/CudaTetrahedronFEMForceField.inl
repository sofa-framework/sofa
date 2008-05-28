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
    void TetrahedronFEMForceFieldCuda3f_addForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v);
    void TetrahedronFEMForceFieldCuda3f_addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor);

    void TetrahedronFEMForceFieldCuda3f1_addForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v);
    void TetrahedronFEMForceFieldCuda3f1_addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor);

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

    void TetrahedronFEMForceFieldCuda3d_addForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v);
    void TetrahedronFEMForceFieldCuda3d_addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor);

    void TetrahedronFEMForceFieldCuda3d1_addForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v);
    void TetrahedronFEMForceFieldCuda3d1_addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor);

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

} // extern "C"

template<>
class CudaKernelsTetrahedronFEMForceField<CudaVec3fTypes>
{
public:
    static void addForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v)
    {   TetrahedronFEMForceFieldCuda3f_addForce(nbElem, nbVertex, nbElemPerVertex, elems, state, eforce, velems, f, x, v); }
    static void addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor)
    {   TetrahedronFEMForceFieldCuda3f_addDForce(nbElem, nbVertex, nbElemPerVertex, elems, state, eforce, velems, df, dx, factor); }
};

template<>
class CudaKernelsTetrahedronFEMForceField<CudaVec3f1Types>
{
public:
    static void addForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v)
    {   TetrahedronFEMForceFieldCuda3f1_addForce(nbElem, nbVertex, nbElemPerVertex, elems, state, eforce, velems, f, x, v); }
    static void addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor)
    {   TetrahedronFEMForceFieldCuda3f1_addDForce(nbElem, nbVertex, nbElemPerVertex, elems, state, eforce, velems, df, dx, factor); }
};

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

template<>
class CudaKernelsTetrahedronFEMForceField<CudaVec3dTypes>
{
public:
    static void addForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v)
    {   TetrahedronFEMForceFieldCuda3d_addForce(nbElem, nbVertex, nbElemPerVertex, elems, state, eforce, velems, f, x, v); }
    static void addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor)
    {   TetrahedronFEMForceFieldCuda3d_addDForce(nbElem, nbVertex, nbElemPerVertex, elems, state, eforce, velems, df, dx, factor); }
};

template<>
class CudaKernelsTetrahedronFEMForceField<CudaVec3d1Types>
{
public:
    static void addForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, void* eforce, const void* velems, void* f, const void* x, const void* v)
    {   TetrahedronFEMForceFieldCuda3d1_addForce(nbElem, nbVertex, nbElemPerVertex, elems, state, eforce, velems, f, x, v); }
    static void addDForce(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, void* eforce, const void* velems, void* df, const void* dx, double factor)
    {   TetrahedronFEMForceFieldCuda3d1_addDForce(nbElem, nbVertex, nbElemPerVertex, elems, state, eforce, velems, df, dx, factor); }
};

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

using namespace gpu::cuda;

template<class TCoord, class TDeriv, class TReal>
void TetrahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::reinit(Main* m)
{
    Data& data = m->data;
    m->_strainDisplacements.resize( m->_indexedElements->size() );
    m->_materialsStiffnesses.resize(m->_indexedElements->size() );

    const VecElement& elems = *m->_indexedElements;

    VecCoord& p = *m->mstate->getX0();
    (*m->f_initialPoints.beginEdit()) = p;

    m->_rotations.resize( m->_indexedElements->size() );
    m->_initialRotations.resize( m->_indexedElements->size() );
    m->_rotationIdx.resize(m->_indexedElements->size() *4);
    m->_rotatedInitialElements.resize(m->_indexedElements->size());

    std::vector<int> activeElems;
    for (unsigned int i=0; i<elems.size(); i++)
    {
        if (!m->_trimgrid || m->_trimgrid->isCubeActive(i/6))
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
        m->computeMaterialStiffness(ei,a,b,c,d);
        m->initLarge(ei,a,b,c,d);
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
        const Coord& a = m->_rotatedInitialElements[ei][0];
        const Coord& b = m->_rotatedInitialElements[ei][1];
        const Coord& c = m->_rotatedInitialElements[ei][2];
        const Coord& d = m->_rotatedInitialElements[ei][3];
        data.setE(i, e, a, b, c, d, m->_materialsStiffnesses[ei], m->_strainDisplacements[ei]);
        for (unsigned int j=0; j<e.size(); j++)
            data.setV(e[j], nelems[e[j]]++, i*e.size()+j);
    }
}

template<class TCoord, class TDeriv, class TReal>
void TetrahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addForce(Main* m, VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    if (m->needUpdateTopology)
    {
        reinit(m);
        m->needUpdateTopology = false;
    }
    Data& data = m->data;
    // Count active cubes in topology
    if (m->_trimgrid)
    {
        int nactive = 0;
#ifdef SOFA_NEW_HEXA
        int ncubes = m->_trimgrid->getNbHexas();
#else
        int ncubes = m->_trimgrid->getNbCubes();
#endif
        for (int i=0; i<ncubes; i++)
            if (m->_trimgrid->isCubeActive(i)) ++nactive;
        if ((int)data.size() != 6*nactive)
            m->reinit();
    }


    f.resize(x.size());
    Kernels::addForce(
        data.size(),
        data.nbVertex,
        data.nbElementPerVertex,
        data.elems.deviceRead(),
        data.state.deviceWrite(),
        data.eforce.deviceWrite(),
        data.velems.deviceRead(),
        (      Deriv*)f.deviceWrite() + data.vertex0,
        (const Coord*)x.deviceRead()  + data.vertex0,
        (const Deriv*)v.deviceRead()  + data.vertex0);

#if 0
    // compare with CPU version

    const VecElement& elems = *m->_indexedElements;
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
        D[3] = m->_rotatedInitialElements[i][1][0] - xb[0];
        D[4] = m->_rotatedInitialElements[i][1][1] - xb[1];
        D[5] = m->_rotatedInitialElements[i][1][2] - xb[2];
        D[6] = m->_rotatedInitialElements[i][2][0] - xc[0];
        D[7] = m->_rotatedInitialElements[i][2][1] - xc[1];
        D[8] = m->_rotatedInitialElements[i][2][2] - xc[2];
        D[9] = m->_rotatedInitialElements[i][3][0] - xd[0];
        D[10]= m->_rotatedInitialElements[i][3][1] - xd[1];
        D[11]= m->_rotatedInitialElements[i][3][2] - xd[2];
        Vec<6,Real> S = -((m->_materialsStiffnesses[i]) * ((m->_strainDisplacements[i]).multTranspose(D)))*(e.bx);

        Vec<6,Real> Sdiff = S-s.S;

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

template<class TCoord, class TDeriv, class TReal>
void TetrahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addDForce (Main* m, VecDeriv& df, const VecDeriv& dx, double kFactor, double /*bFactor*/)
{
    Data& data = m->data;
    df.resize(dx.size());
    Kernels::addDForce(
        data.size(),
        data.nbVertex,
        data.nbElementPerVertex,
        data.elems.deviceRead(),
        data.state.deviceRead(),
        data.eforce.deviceWrite(),
        data.velems.deviceRead(),
        (      Deriv*)df.deviceWrite() + data.vertex0,
        (const Deriv*)dx.deviceRead()  + data.vertex0,
        kFactor);
}

// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaTetrahedronFEMForceField_ImplMethods(T) \
    template<> void TetrahedronFEMForceField< T >::reinit() \
    { data.reinit(this); } \
    template<> void TetrahedronFEMForceField< T >::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v) \
    { data.addForce(this, f, x, v); } \
    template<> void TetrahedronFEMForceField< T >::addDForce(VecDeriv& df, const VecDeriv& dx, double kFactor, double bFactor) \
    { data.addDForce(this, df, dx, kFactor, bFactor); }

CudaTetrahedronFEMForceField_ImplMethods(gpu::cuda::CudaVec3fTypes);
CudaTetrahedronFEMForceField_ImplMethods(gpu::cuda::CudaVec3f1Types);

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

CudaTetrahedronFEMForceField_ImplMethods(gpu::cuda::CudaVec3dTypes);
CudaTetrahedronFEMForceField_ImplMethods(gpu::cuda::CudaVec3d1Types);

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

#undef CudaTetrahedronFEMForceField_ImplMethods

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
