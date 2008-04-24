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


SOFA_DECL_CLASS(ConicalForceField)

int ConicalForceFieldClass = core::RegisterObject("Repulsion applied by a cone toward the exterior")
#ifndef SOFA_FLOAT
        .add< ConicalForceField<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ConicalForceField<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class ConicalForceField<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class ConicalForceField<Vec3fTypes>;
#endif


} // namespace forcefield

} // namespace component

} // namespace sofa
