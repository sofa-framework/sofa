/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_PCUBE_H
#define SOFA_HELPER_PCUBE_H

#include <sofa/helper/helper.h>

namespace sofa
{
namespace helper
{
namespace polygon_cube_intersection
{

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
 *                                                                           *
 *                       POLYGON-CUBE INTERSECTION                           *
 *                       by Don Hatch & Daniel Green                         *
 *                       January 1994                                        *
 *                                                                           *
 *    CONTENTS:                                                              *
 *        polygon_intersects_cube()                                          *
 *        fast_polygon_intersects_cube()                                     *
 *        trivial_vertex_tests()                                             *
 *        segment_intersects_cube()                                          *
 *        polygon_contains_point_3d()                                        *
 *                                                                           *
 *                                                                           *
 *  This module contains  routines that test points,  segments and polygons  *
 *  for intersections  with the  unit cube defined  as the  axially aligned  *
 *  cube of edge length 1 centered  at the origin.  Polygons may be convex,  *
 *  concave or  self-intersecting.  Also contained is a routine  that tests  *
 *  whether a point  is within a polygon.  All routines  are intended to be  *
 *  fast and robust. Note that the cube and polygons are defined to include  *
 *  their boundaries.                                                        *
 *                                                                           *
 *  The  fast_polygon_intersects_cube  routine  is  meant  to  replace  the  *
 *  triangle-cube  intersection routine  in  Graphics Gems  III by  Douglas  *
 *  Voorhies.   While  that  original  algorithm  is  still sound,   it  is  *
 *  specialized  for triangles  and  the  implementation contained  several  *
 *  bugs and inefficiencies.  The trivial_vertex_tests routine defined here  *
 *  is  almost an  exact copy  of the  trivial point-plane  tests from  the  *
 *  beginning of Voorhies' algorithm but broken out into a separate routine  *
 *  which is called by  fast_polygon_intersects_cube.  The segment-cube and  *
 *  polygon-cube intersection algorithms have been completely rewritten.     *
 *                                                                           *
 *  Notice that trivial_vertex_tests can be  used to quickly test an entire  *
 *  set of vertices  for trivial reject or accept.  This  can be useful for  *
 *  testing  polyhedra  or  entire  polygon  meshes.   When  used  to  test  *
 *  polyhedra, remember  that these  routines only  test points,  edges and  *
 *  surfaces, not volumes.  If no such intersection is reported, the caller  *
 *  should be  aware that the volume  of the polyhedra could  still contain  *
 *  the entire  unit box which  would then need to  be checked for  with an  *
 *  additional point-within-polyhedron test.  The origin would be a natural  *
 *  point to check in such a test.                                           *
 *                                                                           *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


#ifndef real
#define real float
#endif

/*
 *                   POLYGON INTERSECTS CUBE
 *
 * Tells how the given polygon intersects the cube of edge length 1 centered
 * at the origin.
 * If any vertex or edge of the polygon intersects the cube,
 * a value of 1 will be returned.
 * Otherwise the value returned will be the multiplicity of containment
 * of the cross-section of the cube in the polygon; this may
 * be interpreted as a boolean value in any of the standard
 * ways; e.g. the even-odd rule (it's inside the polygon iff the
 * result is odd) or the winding rule (it's inside the polygon iff
 * the result is nonzero).
 *
 * The "polynormal" argument is a vector perpendicular to the polygon.  It
 * need not be of unit length.  It is suggested that Newell's method be used
 * to calculate polygon normals (See Graphics Gems III).  Zero-lengthed normals
 * are quite acceptable for degenerate polygons but are not acceptable
 * otherwise.  In particular, beware of zero-length normals which Newell's
 * method can return for certain self-intersecting polygons (for example
 * a bow-tie quadrilateral).
 *
 * The already_know_verts_are_outside_cube flag is unused by this routine
 * but may be useful for alternate implementations.
 *
 * The already_know_edges_are_outside_cube flag is useful when testing polygon
 * meshes with shared edges in order to not test the same edge more than once.
 *
 * Note: usually users of this module would not want to call this routine
 * directly unless they have previously tested the vertices with the trivial
 * vertex test below.  Normally one would call the fast_polygon_intersects_cube
 * utility instead which combines both of these tests.
 */
extern SOFA_HELPER_API int
polygon_intersects_cube(int nverts, const real verts[/* nverts */][3],
        const real polynormal[3],
        int already_know_verts_are_outside_cube,
        int already_know_edges_are_outside_cube);


/*
 *                   FAST POLYGON INTERSECTS CUBE
 *
 * This is a version of the same polygon-cube intersection that first calls
 * trivial_vertex_tests() to hopefully skip the more expensive definitive test.
 * It simply calls polygon_intersects_cube() when that fails.
 * Note that unlike polygon_intersects_cube(), this routine does use the
 * already_know_verts_are_outside_cube argument.
 */
extern SOFA_HELPER_API int
fast_polygon_intersects_cube(int nverts, const real verts[/* nverts */][3],
        const real polynormal[3],
        int already_know_verts_are_outside_cube,
        int already_know_edges_are_outside_cube);


/*
 *                   TRIVIAL VERTEX TESTS
 *
 * Returns 1 if any of the vertices are inside the cube of edge length 1
 * centered at the origin (trivial accept), 0 if all vertices are outside
 * of any testing plane (trivial reject), -1 otherwise (couldn't help).
 */
extern SOFA_HELPER_API int
trivial_vertex_tests(int nverts, const real verts[/* nverts */][3],
        int already_know_verts_are_outside_cube);


/*
 *                   SEGMENT INTERSECTS CUBE
 *
 * Returns 1 if the given line segment intersects the cube of edge length 1
 * centered at the origin, 0 otherwise.
 */
extern SOFA_HELPER_API int
segment_intersects_cube(const real v0[3], const real v1[3]);


/*
 *                   POLYGON CONTAINS POINT 3D
 *
 * Tells whether a given polygon with nonzero area contains a point which is
 * assumed to lie in the plane of the polygon.
 * Actually returns the multiplicity of containment.  This will always be 1
 * or 0 for non-self-intersecting planar polygons with the normal in the
 * standard direction (towards the eye when looking at the polygon so that
 * it's CCW).
 */
extern SOFA_HELPER_API int
polygon_contains_point_3d(int nverts, const real verts[/* nverts */][3],
        const real polynormal[3],
        real point[3]);

/*
 *  Calculate a vector perpendicular to a planar polygon.
 *  If the polygon is non-planar, a "best fit" plane will be used.
 *  The polygon may be concave or even self-intersecting,
 *  but it should have nonzero area or the result will be a zero vector
 *  (e.g. the "bowtie" quad).
 *  The length of vector will be twice the area of the polygon.
 *  NOTE:  This algorithm gives the same answer as Newell's method
 *  (see Graphics Gems III) but is slightly more efficient than Newell's
 *  for triangles and quads (slightly less efficient for higher polygons).
 */
SOFA_HELPER_API real *
get_polygon_normal(real normal[3],
        int nverts, const real verts[/* nverts */][3]);

}
}
}

#endif
