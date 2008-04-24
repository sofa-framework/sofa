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


SOFA_DECL_CLASS(JointSpringForceField)

// Register in the Factory
int JointSpringForceFieldClass = core::RegisterObject("Springs for Rigids")
#ifndef SOFA_FLOAT
        .add< JointSpringForceField<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< JointSpringForceField<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class JointSpringForceField<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class JointSpringForceField<Rigid3fTypes>;
#endif


} // namespace forcefield

} // namespace component

} // namespace sofa

