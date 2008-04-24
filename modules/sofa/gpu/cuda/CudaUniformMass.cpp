#include "CudaTypes.h"
#include "CudaUniformMass.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/gl/Axis.h>

namespace sofa
{

namespace component
{

namespace mass
{

template <>
double UniformMass<gpu::cuda::CudaRigid3fTypes,sofa::defaulttype::Rigid3fMass>::getPotentialEnergy( const VecCoord& x )
{
    double e = 0;
    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    for (unsigned int i=0; i<x.size(); i++)
    {
        e += g*mass.getValue().mass*x[i].getCenter();
    }
    return e;
}

template <>
sofa::defaulttype::Vector3::value_type UniformMass<gpu::cuda::CudaRigid3fTypes,sofa::defaulttype::Rigid3fMass>::getElementMass(unsigned int )
{
    return (sofa::defaulttype::Vector3::value_type)(mass.getValue().mass);
}

template <>
void UniformMass<gpu::cuda::CudaRigid3fTypes, Rigid3fMass>::draw()
{
    if (!getContext()->getShowBehaviorModels())
        return;
    const VecCoord& x = *mstate->getX();
    defaulttype::Vec3d len;

    // The moment of inertia of a box is:
    //   m->_I(0,0) = M/REAL(12.0) * (ly*ly + lz*lz);
    //   m->_I(1,1) = M/REAL(12.0) * (lx*lx + lz*lz);
    //   m->_I(2,2) = M/REAL(12.0) * (lx*lx + ly*ly);
    // So to get lx,ly,lz back we need to do
    //   lx = sqrt(12/M * (m->_I(1,1)+m->_I(2,2)-m->_I(0,0)))
    // Note that RigidMass inertiaMatrix is already divided by M
    double m00 = mass.getValue().inertiaMatrix[0][0];
    double m11 = mass.getValue().inertiaMatrix[1][1];
    double m22 = mass.getValue().inertiaMatrix[2][2];
    len[0] = sqrt(m11+m22-m00);
    len[1] = sqrt(m00+m22-m11);
    len[2] = sqrt(m00+m11-m22);

    for (unsigned int i=0; i<x.size(); i++)
    {
        helper::gl::Axis::draw(x[i].getCenter(), x[i].getOrientation(), len);
    }
}

} // namespace mass

} // namespace component

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaUniformMass)

int UniformMassCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::mass::UniformMass<CudaVec3fTypes,float> >()
        .add< component::mass::UniformMass<CudaVec3f1Types,float> >()
        .add< component::mass::UniformMass<CudaRigid3fTypes,sofa::defaulttype::Rigid3fMass> >()
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
