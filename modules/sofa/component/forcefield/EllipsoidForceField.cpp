#include <sofa/component/forcefield/EllipsoidForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

template class EllipsoidForceField<Vec3dTypes>;
template class EllipsoidForceField<Vec3fTypes>;
template class EllipsoidForceField<Vec2dTypes>;
template class EllipsoidForceField<Vec2fTypes>;
template class EllipsoidForceField<Vec1dTypes>;
template class EllipsoidForceField<Vec1fTypes>;
//template class EllipsoidForceField<Vec6dTypes>;
//template class EllipsoidForceField<Vec6fTypes>;


SOFA_DECL_CLASS(EllipsoidForceField)

int EllipsoidForceFieldClass = core::RegisterObject("Repulsion applied by an ellipsoid toward the exterior or the interior")
        .add< EllipsoidForceField<Vec3dTypes> >()
        .add< EllipsoidForceField<Vec3fTypes> >()
        .add< EllipsoidForceField<Vec2dTypes> >()
        .add< EllipsoidForceField<Vec2fTypes> >()
        .add< EllipsoidForceField<Vec1dTypes> >()
        .add< EllipsoidForceField<Vec1fTypes> >()
//.add< EllipsoidForceField<Vec6dTypes> >()
//.add< EllipsoidForceField<Vec6fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa
