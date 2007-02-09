// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#include <sofa/component/forcefield/SpringForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/ObjectFactory.h>
//#include <typeinfo>


namespace sofa
{

namespace component
{

namespace forcefield
{

SOFA_DECL_CLASS(SpringForceField)

using namespace sofa::defaulttype;

template class SpringForceField<Vec3dTypes>;
template class SpringForceField<Vec3fTypes>;

// Register in the Factory
int SpringForceFieldClass = core::RegisterObject("Springs")
        .add< SpringForceField<Vec3dTypes> >()
        .add< SpringForceField<Vec3fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa

