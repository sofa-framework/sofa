#include <sofa/component/forcefield/SphereForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(SphereForceField)

int SphereForceFieldClass = core::RegisterObject("Repulsion applied by a sphere toward the exterior")
#ifndef SOFA_FLOAT
        .add< SphereForceField<Vec3dTypes> >()
        .add< SphereForceField<Vec2dTypes> >()
        .add< SphereForceField<Vec1dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< SphereForceField<Vec3fTypes> >()
        .add< SphereForceField<Vec2fTypes> >()
        .add< SphereForceField<Vec1fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class SphereForceField<Vec3dTypes>;
template class SphereForceField<Vec2dTypes>;
template class SphereForceField<Vec1dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SphereForceField<Vec3fTypes>;
template class SphereForceField<Vec2fTypes>;
template class SphereForceField<Vec1fTypes>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa
