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
#include "TriangleBendingSprings.inl"
#include "Common/Vec3Types.h"
#include "Common/ObjectFactory.h"

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(TriangleBendingSprings)

using namespace Common;

template class TriangleBendingSprings<Vec3fTypes>;
template class TriangleBendingSprings<Vec3dTypes>;

template<class DataTypes>
void create(TriangleBendingSprings<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< TriangleBendingSprings<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    obj->parseFields(arg->getAttributeMap() );
}

Creator< ObjectFactory, TriangleBendingSprings<Vec3dTypes> > TriangleBendingSpringsVec3dClass("TriangleBendingSprings", true);
Creator< ObjectFactory, TriangleBendingSprings<Vec3fTypes> > TriangleBendingSpringsVec3fClass("TriangleBendingSprings", true);

}

}
