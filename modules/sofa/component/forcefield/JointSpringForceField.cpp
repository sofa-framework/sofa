// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#include <sofa/component/forcefield/JointSpringForceField.inl>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/ObjectFactory.h>
//#include <typeinfo>


namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

template class JointSpringForceField<Rigid3dTypes>;
template class JointSpringForceField<Rigid3fTypes>;

SOFA_DECL_CLASS(JointSpringForceField)

// Register in the Factory
int JointSpringForceFieldClass = core::RegisterObject("Springs for Rigids")
        .add< JointSpringForceField<Rigid3dTypes> >()
        .add< JointSpringForceField<Rigid3fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa

