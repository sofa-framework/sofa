// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#include <sofa/component/forcefield/StiffSpringForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(StiffSpringForceField)

// Register in the Factory
int StiffSpringForceFieldClass = core::RegisterObject("Stiff springs for implicit integration")
#ifndef SOFA_FLOAT
        .add< StiffSpringForceField<Vec3dTypes> >()
        .add< StiffSpringForceField<Vec2dTypes> >()
        .add< StiffSpringForceField<Vec1dTypes> >()
        .add< StiffSpringForceField<Vec6dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< StiffSpringForceField<Vec3fTypes> >()
        .add< StiffSpringForceField<Vec2fTypes> >()
        .add< StiffSpringForceField<Vec1fTypes> >()
        .add< StiffSpringForceField<Vec6fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class StiffSpringForceField<Vec3dTypes>;
template class StiffSpringForceField<Vec2dTypes>;
template class StiffSpringForceField<Vec1dTypes>;
template class StiffSpringForceField<Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class StiffSpringForceField<Vec3fTypes>;
template class StiffSpringForceField<Vec2fTypes>;
template class StiffSpringForceField<Vec1fTypes>;
template class StiffSpringForceField<Vec6fTypes>;
#endif


} // namespace forcefield

} // namespace component

} // namespace sofa

