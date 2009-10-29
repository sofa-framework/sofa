/*
 * MeshGenerationFromPolyhedron.cpp
 *
 *  Created on: 27 oct. 2009
 *      Author: froy
 */
#define CGALPLUGIN_MESHGENERATIONFROMPOLYHEDRON_CPP

#include <cgal_config.h>
#include "MeshGenerationFromPolyhedron.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

SOFA_DECL_CLASS(MeshGenerationFromPolyhedron)

using namespace sofa::defaulttype;
using namespace cgal;

int MeshGenerationFromPolyhedronClass = sofa::core::RegisterObject("Generate tetrahedral mesh from triangular mesh")
#ifndef SOFA_FLOAT
        .add< MeshGenerationFromPolyhedron<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< MeshGenerationFromPolyhedron<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_CGALPLUGIN_API MeshGenerationFromPolyhedron<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_CGALPLUGIN_API MeshGenerationFromPolyhedron<Vec3fTypes>;
#endif //SOFA_DOUBLE
