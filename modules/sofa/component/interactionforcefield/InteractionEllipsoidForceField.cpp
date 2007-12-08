#include <sofa/component/interactionforcefield/InteractionEllipsoidForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

using namespace sofa::defaulttype;

template class InteractionEllipsoidForceField<Vec3dTypes, Rigid3dTypes>;
//template class InteractionEllipsoidForceField<Vec3fTypes, Rigid3fTypes>;
//template class InteractionEllipsoidForceField<Vec3dTypes, Vec3dTypes>;
//template class InteractionEllipsoidForceField<Vec3fTypes, Vec3fTypes>;
/*
template class InteractionEllipsoidForceField<Vec2dTypes, Rigid2dTypes>;
template class InteractionEllipsoidForceField<Vec2fTypes, Rigid2dTypes>;
*/

SOFA_DECL_CLASS(InteractionEllipsoidForceField)

int EllipsoidForceFieldClass = core::RegisterObject("Repulsion applied by an ellipsoid toward the exterior or the interior")
        .add< InteractionEllipsoidForceField<Vec3dTypes, Rigid3dTypes> >()
//.add< InteractionEllipsoidForceField<Vec3fTypes, Rigid3fTypes> >()
//.add< InteractionEllipsoidForceField<Vec3dTypes, Vec3dTypes> >()
//.add< InteractionEllipsoidForceField<Vec3fTypes, Vec3fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa
