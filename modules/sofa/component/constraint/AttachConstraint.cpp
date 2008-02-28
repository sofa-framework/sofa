#include <sofa/component/constraint/AttachConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/simulation/tree/GNode.h>
#include <sofa/component/mass/UniformMass.h>

namespace sofa
{

namespace component
{

namespace constraint
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

SOFA_DECL_CLASS(AttachConstraint)

int AttachConstraintClass = core::RegisterObject("Attach given pair of particles, projecting the positions of the second particles to the first ones")
        .add< AttachConstraint<Vec3dTypes> >()
        .add< AttachConstraint<Vec3fTypes> >()
        .add< AttachConstraint<Vec2dTypes> >()
        .add< AttachConstraint<Vec2fTypes> >()
        .add< AttachConstraint<Vec1dTypes> >()
        .add< AttachConstraint<Vec1fTypes> >()
//.add< AttachConstraint<Vec6dTypes> >()
//.add< AttachConstraint<Vec6fTypes> >()
        .add< AttachConstraint<Rigid3dTypes> >()
        .add< AttachConstraint<Rigid3fTypes> >()
        .add< AttachConstraint<Rigid2dTypes> >()
        .add< AttachConstraint<Rigid2fTypes> >()
        ;

template class AttachConstraint<Vec3dTypes>;
template class AttachConstraint<Vec3fTypes>;
template class AttachConstraint<Vec2dTypes>;
template class AttachConstraint<Vec2fTypes>;
template class AttachConstraint<Vec1dTypes>;
template class AttachConstraint<Vec1fTypes>;
//template class AttachConstraint<Vec6dTypes>;
//template class AttachConstraint<Vec6fTypes>;
template class AttachConstraint<Rigid3dTypes>;
template class AttachConstraint<Rigid3fTypes>;
template class AttachConstraint<Rigid2dTypes>;
template class AttachConstraint<Rigid2fTypes>;

} // namespace constraint

} // namespace component

} // namespace sofa

