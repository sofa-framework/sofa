//
// C++ Implementation: TriangleBendingSprings
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <sofa/component/forcefield/TriangleBendingSprings.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

template class TriangleBendingSprings<Vec3fTypes>;
template class TriangleBendingSprings<Vec3dTypes>;


SOFA_DECL_CLASS(TriangleBendingSprings)

// Register in the Factory
int TriangleBendingSpringsClass = core::RegisterObject("Springs added to a traingular mesh to prevent bending")
        .add< TriangleBendingSprings<Vec3dTypes> >()
        .add< TriangleBendingSprings<Vec3fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa

