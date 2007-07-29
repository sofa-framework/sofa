#ifndef SOFA_GPU_CUDA_CUDAUNIFORMMASS_INL
#define SOFA_GPU_CUDA_CUDAUNIFORMMASS_INL

#include "CudaUniformMass.h"
#include <sofa/component/mass/UniformMass.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void UniformMassCuda3f_addMDx(unsigned int size, float mass, void* res, const void* dx);
    void UniformMassCuda3f_accFromF(unsigned int size, float mass, void* a, const void* f);
    void UniformMassCuda3f_addForce(unsigned int size, const float *mg, void* f);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace mass
{

using namespace gpu::cuda;

// -- Mass interface
template <>
void UniformMass<CudaVec3fTypes, float>::addMDx(VecDeriv& res, const VecDeriv& dx, double factor)
{
    UniformMassCuda3f_addMDx(dx.size(), (float)(mass.getValue()*factor), res.deviceWrite(), dx.deviceRead());
}

template <>
void UniformMass<CudaVec3fTypes, float>::accFromF(VecDeriv& a, const VecDeriv& f)
{
    UniformMassCuda3f_accFromF(f.size(), mass.getValue(), a.deviceWrite(), f.deviceRead());
}

template <>
void UniformMass<CudaVec3fTypes, float>::addForce(VecDeriv& f, const VecCoord&, const VecDeriv&)
{
    // weight
    Vec3d g ( this->getContext()->getLocalGravity() );
    Deriv theGravity;
    DataTypes::set( theGravity, g[0], g[1], g[2]);
    Deriv mg = theGravity * mass.getValue();
    UniformMassCuda3f_addForce(f.size(), mg.ptr(), f.deviceWrite());
}

template <>
bool UniformMass<gpu::cuda::CudaVec3fTypes, float>::addBBox(double* minBBox, double* maxBBox)
{
    const VecCoord& x = *this->mstate->getX();
    //if (!x.isHostValid()) return false; // Do not recompute bounding box if it requires to transfer data from device
    for (unsigned int i=0; i<x.size(); i++)
    {
        //const Coord& p = x[i];
        const Coord& p = x.getCached(i);
        for (int c=0; c<3; c++)
        {
            if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
            if (p[c] < minBBox[c]) minBBox[c] = p[c];
        }
    }
    return true;
}

} // namespace mass

} // namespace component

} // namespace sofa

#endif
