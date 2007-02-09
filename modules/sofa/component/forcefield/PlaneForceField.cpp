#include <sofa/component/forcefield/PlaneForceField.inl>
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

template class PlaneForceField<Vec3dTypes>;
template class PlaneForceField<Vec3fTypes>;


SOFA_DECL_CLASS(PlaneForceField)

int PlaneForceFieldClass = core::RegisterObject("Repulsion applied by a plane toward the exterior (half-space)")
        .add< PlaneForceField<Vec3dTypes> >()
        .add< PlaneForceField<Vec3fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa
