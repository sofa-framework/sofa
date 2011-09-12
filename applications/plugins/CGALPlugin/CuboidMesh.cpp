/*
 * CylinderMesh.cpp
 *
 *  Created on: 12 sep. 2011
 *      Author: Yiyi
 */
#define CGALPLUGIN_CUBOIDMESH_CPP

#include <cgal_config.h>
#include "CuboidMesh.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

SOFA_DECL_CLASS(CuboidMesh)

using namespace sofa::defaulttype;
using namespace cgal;

int CuboidMeshClass = sofa::core::RegisterObject("Generate a regular tetrahedron mesh of a cuboid")
#ifndef SOFA_FLOAT
        .add< CuboidMesh<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< CuboidMesh<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_CGALPLUGIN_API CuboidMesh<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_CGALPLUGIN_API CuboidMesh<Vec3fTypes>;
#endif //SOFA_DOUBLE
