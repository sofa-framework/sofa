/*
 * TriangularConvexHull3D.cpp
 *
 *  Created on: 27 oct. 2009
 *      Author: froy
 */
#define CGALPLUGIN_TRIANGULARCONVEXHULL3D_CPP

#include <cgal_config.h>
#include "TriangularConvexHull3D.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

SOFA_DECL_CLASS(TriangularConvexHull3D)

using namespace sofa::defaulttype;
using namespace cgal;

int TriangularConvexHull3DClass = sofa::core::RegisterObject("Generate triangular convex hull around points")
#ifndef SOFA_FLOAT
        .add< TriangularConvexHull3D<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< TriangularConvexHull3D<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_CGALPLUGIN_API TriangularConvexHull3D<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_CGALPLUGIN_API TriangularConvexHull3D<Vec3fTypes>;
#endif //SOFA_DOUBLE
