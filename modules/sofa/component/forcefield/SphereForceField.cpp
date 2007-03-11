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


SOFA_DECL_CLASS(SphereForceField)

int SphereForceFieldClass = core::RegisterObject("Repulsion applied by a sphere toward the exterior")
        .add< SphereForceField<Vec3dTypes> >()
        .add< SphereForceField<Vec3fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa
