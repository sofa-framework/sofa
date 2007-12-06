//
// C++ Implementation: ConicalForceField
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <sofa/component/forcefield/ConicalForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

template class ConicalForceField<Vec3dTypes>;
template class ConicalForceField<Vec3fTypes>;
//template class ConicalForceField<Vec2dTypes>;
//template class ConicalForceField<Vec2fTypes>;
//template class ConicalForceField<Vec1dTypes>;
//template class ConicalForceField<Vec1fTypes>;
//template class ConicalForceField<Vec6dTypes>;
//template class ConicalForceField<Vec6fTypes>;


SOFA_DECL_CLASS(ConicalForceField)

int ConicalForceFieldClass = core::RegisterObject("Repulsion applied by a cone toward the exterior")
        .add< ConicalForceField<Vec3dTypes> >()
        .add< ConicalForceField<Vec3fTypes> >()
//.add< ConicalForceField<Vec2dTypes> >()
//.add< ConicalForceField<Vec2fTypes> >()
//.add< ConicalForceField<Vec1dTypes> >()
//.add< ConicalForceField<Vec1fTypes> >()
//.add< ConicalForceField<Vec6dTypes> >()
//.add< ConicalForceField<Vec6fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa
