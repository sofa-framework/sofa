#ifndef SOFA_CONTRIB_CUDA_CUDAUNIFORMMASS_INL
#define SOFA_CONTRIB_CUDA_CUDAUNIFORMMASS_INL

#include "CudaUniformMass.h"
#include "Sofa/Components/UniformMass.inl"

namespace Sofa
{

namespace Contrib
{

namespace CUDA
{

extern "C"
{
    void UniformMassCuda3f_addMDx(unsigned int size, float mass, void* res, const void* dx);
    void UniformMassCuda3f_accFromF(unsigned int size, float mass, void* a, const void* f);
    void UniformMassCuda3f_addForce(unsigned int size, const float *mg, void* f);
}

} // namespace CUDA

} // namespace Contrib

namespace Components
{

using namespace Contrib::CUDA;

// -- Mass interface
template <>
void UniformMass<CudaVec3fTypes, float>::addMDx(VecDeriv& res, const VecDeriv& dx)
{
    UniformMassCuda3f_addMDx(dx.size(), mass, res.deviceWrite(), dx.deviceRead());
}

template <>
void UniformMass<CudaVec3fTypes, float>::accFromF(VecDeriv& a, const VecDeriv& f)
{
    UniformMassCuda3f_accFromF(f.size(), mass, a.deviceWrite(), f.deviceRead());
}

template <>
void UniformMass<CudaVec3fTypes, float>::addForce(VecDeriv& f, const VecCoord&, const VecDeriv&)
{
    // weight
    const double* g = this->getContext()->getLocalGravity();
    Deriv theGravity;
    DataTypes::set( theGravity, g[0], g[1], g[2]);
    Deriv mg = theGravity * mass;
    UniformMassCuda3f_addForce(f.size(), mg, f.deviceWrite());
}

} // namespace Components

} // namespace Sofa

#endif
