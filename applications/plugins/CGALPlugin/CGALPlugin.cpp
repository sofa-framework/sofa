/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
 *                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
 *                                                                             *
 * This library is free software; you can redistribute it and/or modify it     *
 * under the terms of the GNU Lesser General Public License as published by    *
 * the Free Software Foundation; either version 2.1 of the License, or (at     *
 * your option) any later version.                                             *
 *                                                                             *
 * This library is distributed in the hope that it will be useful, but WITHOUT *
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
 * for more details.                                                           *
 *                                                                             *
 * You should have received a copy of the GNU Lesser General Public License    *
 * along with this library; if not, write to the Free Software Foundation,     *
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
 *******************************************************************************
 *                               SOFA :: Plugins                               *
 *                                                                             *
 * Authors: The SOFA Team and external contributors (see Authors.txt)          *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/
#include <CGALPlugin/CGALPlugin.h>
#include <sofa/core/Plugin.h>

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <CGALPlugin/CuboidMesh.inl>
#include <CGALPlugin/DecimateMesh.inl>
#include <CGALPlugin/TriangularConvexHull3D.inl>
#include <CGALPlugin/MeshGenerationFromPolyhedron.inl>
#ifdef CGALPLUGIN_SOFA_HAVE_PLUGIN_IMAGE
# include <CGALPlugin/MeshGenerationFromImage.inl>
#endif


using namespace sofa::defaulttype;
using namespace cgal;


class CGALPlugin: public sofa::core::Plugin {
public:
    CGALPlugin(): Plugin("CGALPlugin") {
        setDescription("Use CGAL functionalities in SOFA.");
        setVersion("0.1");
        setLicense("LGPL");

#ifdef SOFA_FLOAT
        addComponent< CuboidMesh<Vec3fTypes> >();
#else
        addComponent< CuboidMesh<Vec3dTypes> >();
#endif
        setDescription("CuboidMesh", "Generate a regular tetrahedron mesh of a cuboid.");
#if !defined(SOFA_DOUBLE) && !defined(SOFA_FLOAT)
        addTemplateInstance< CuboidMesh<Vec3fTypes> >();
#endif

#ifdef SOFA_FLOAT
        addComponent< DecimateMesh<Vec3fTypes> >();
#else
        addComponent< DecimateMesh<Vec3dTypes> >();
#endif
        setDescription("DecimateMesh", "Simplification of a mesh by the process of reducing the number of faces.");
#if !defined(SOFA_DOUBLE) && !defined(SOFA_FLOAT)
        addTemplateInstance< DecimateMesh<Vec3fTypes> >();
#endif

#ifdef SOFA_FLOAT
        addComponent< MeshGenerationFromPolyhedron<Vec3fTypes> >();
#else
        addComponent< MeshGenerationFromPolyhedron<Vec3dTypes> >();
#endif
        setDescription("MeshGenerationFromPolyhedron", "Generate tetrahedral mesh from triangular mesh.");
#if !defined(SOFA_DOUBLE) && !defined(SOFA_FLOAT)
        addTemplateInstance< MeshGenerationFromPolyhedron<Vec3fTypes> >();
#endif

#ifdef SOFA_FLOAT
        addComponent< TriangularConvexHull3D<Vec3fTypes> >();
#else
        addComponent< TriangularConvexHull3D<Vec3dTypes> >();
#endif
        setDescription("TriangularConvexHull3D", "Generate tetrahedral mesh from triangular mesh.");
#if !defined(SOFA_DOUBLE) && !defined(SOFA_FLOAT)
        addTemplateInstance< TriangularConvexHull3D<Vec3fTypes> >();
#endif

#ifdef CGALPLUGIN_SOFA_HAVE_PLUGIN_IMAGE
# ifdef SOFA_FLOAT
        addComponent< MeshGenerationFromImage<Vec3fTypes, ImageUC> >();
# else
        addComponent< MeshGenerationFromImage<Vec3dTypes, ImageUC> >();
# endif
        setDescription("MeshGenerationFromImage", "Generate tetrahedral mesh from image.");
# if !defined(SOFA_DOUBLE) && !defined(SOFA_FLOAT)
        addTemplateInstance< MeshGenerationFromImage<Vec3fTypes, ImageUC> >();
# endif
#endif

    }
};

SOFA_PLUGIN(CGALPlugin);
