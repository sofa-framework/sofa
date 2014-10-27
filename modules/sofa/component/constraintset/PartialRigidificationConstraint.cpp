#include <sofa/component/constraintset/PartialRigidificationConstraint.inl>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

SOFA_DECL_CLASS(PartialRigidificationConstraint)

int PartialRigidificationConstraintClass = core::RegisterObject("PartialRigidificationConstraint")
#ifndef SOFA_FLOAT
        .add< PartialRigidificationConstraint<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< PartialRigidificationConstraint<Rigid3fTypes> >()
#endif
		;

#ifndef SOFA_FLOAT
template class PartialRigidificationConstraint<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class PartialRigidificationConstraint<Rigid3fTypes>;
#endif

} // namespace constraintset

} // namespace component

} // namespace sofa
