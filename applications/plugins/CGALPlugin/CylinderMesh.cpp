/*
 * CylinderMesh.cpp
 *
 *  Created on: 21 mar. 2010
 *      Author: Yiyi
 */
#define CGALPLUGIN_CYLINDERMESH_CPP

#include <cgal_config.h>
#include "CylinderMesh.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

SOFA_DECL_CLASS(CylinderMesh)

using namespace sofa::defaulttype;
using namespace cgal;

int CylinderMeshClass = sofa::core::RegisterObject("Generate a regular tetrahedron mesh of a cylinder")
#ifndef SOFA_FLOAT
        .add< CylinderMesh<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< CylinderMesh<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_CGALPLUGIN_API CylinderMesh<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_CGALPLUGIN_API CylinderMesh<Vec3fTypes>;
#endif //SOFA_DOUBLE
