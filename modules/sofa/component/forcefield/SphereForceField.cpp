#include <sofa/component/forcefield/SphereForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/MechanicalObject.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

template class SphereForceField<Vec3dTypes>;
template class SphereForceField<Vec3fTypes>;
template class SphereForceField<Vec2dTypes>;
template class SphereForceField<Vec2fTypes>;
template class SphereForceField<Vec1dTypes>;
template class SphereForceField<Vec1fTypes>;
template class SphereForceField<Vec6dTypes>;
template class SphereForceField<Vec6fTypes>;


SOFA_DECL_CLASS(SphereForceField)

int SphereForceFieldClass = core::RegisterObject("Repulsion applied by a sphere toward the exterior")
        .add< SphereForceField<Vec3dTypes> >()
        .add< SphereForceField<Vec3fTypes> >()
        .add< SphereForceField<Vec2dTypes> >()
        .add< SphereForceField<Vec2fTypes> >()
        .add< SphereForceField<Vec1dTypes> >()
        .add< SphereForceField<Vec1fTypes> >()
        .add< SphereForceField<Vec6dTypes> >()
        .add< SphereForceField<Vec6fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa
