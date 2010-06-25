/*
 * DecimateMesh.cpp
 *
 *  Created on: 2nd of June 2010
 *      Author: Olivier
 */
#define CGALPLUGIN_DECIMETEMESH_CPP

#include <cgal_config.h>
#include "DecimateMesh.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

SOFA_DECL_CLASS(DecimateMesh)

using namespace sofa::defaulttype;
using namespace cgal;

int DecimateMeshClass = sofa::core::RegisterObject("Simplification of a mesh by the process of reducing the number of faces")
#ifndef SOFA_FLOAT
        .add< DecimateMesh<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< DecimateMesh<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_CGALPLUGIN_API DecimateMesh<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_CGALPLUGIN_API DecimateMesh<Vec3fTypes>;
#endif //SOFA_DOUBLE
