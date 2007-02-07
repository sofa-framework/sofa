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
#include <sofa/simulation/tree/xml/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

SOFA_DECL_CLASS(TriangleBendingSprings)

using namespace sofa::defaulttype;

template class TriangleBendingSprings<Vec3fTypes>;
template class TriangleBendingSprings<Vec3dTypes>;

template<class DataTypes>
void create(TriangleBendingSprings<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    XML::createWithParent< TriangleBendingSprings<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
    obj->parseFields(arg->getAttributeMap() );
}

Creator<simulation::tree::xml::ObjectFactory, TriangleBendingSprings<Vec3dTypes> > TriangleBendingSpringsVec3dClass("TriangleBendingSprings", true);
Creator<simulation::tree::xml::ObjectFactory, TriangleBendingSprings<Vec3fTypes> > TriangleBendingSpringsVec3fClass("TriangleBendingSprings", true);

}

}
