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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_INIT_BULLET_COLLISION_INIT
#define SOFA_INIT_BULLET_COLLISION_INIT

#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_BULLETCOLLISIONDETECTION
#  define SOFA_BULLETCOLLISIONDETECTION_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_BULLETCOLLISIONDETECTION_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

namespace sofa
{

namespace component
{
/** \mainpage BulletCollisionDetection - a plugin using Bullet collision pipeline within Sofa
 * This plugin contains the Bullet collision pipeline itself named BulletCollisionDetection, where
 * broad phase and narrow phase are performed. So in the Sofa scene, you must place the element
 * BulletCollisionDetection instead of any broad or narrow phase. An important thing to know is
 * that this plugin works well with constraints (LMConstraintSolver for example) and no with penality contacts because
 * it is not robust to collision models penetration. But some improvement has been done with the collision model margin
 * to allow this for most of the primitives (if the primitives don't penetrate).
 *
 *
 * It contains also the element BulletIntersection which must replace any kind of Sofa intersection
 * method. So it has also to be placed in the Sofa scene.
 *
 * It contains models already existing in Sofa and inheriting from them :
 * - BulletTriangleModel
 * - BulletSphereModel
 * - BulletCapsuleModel BulletRigidCapsuleModel
 * - BulletOBBModel
 * - BulletCylinderModel
 * There are also a collision model that don't derive from any of the Sofa collision models :
 * - BulletConvexHullModel
 * Note that BulletConvexHullModel is special because its parametirization can completely change its behaviour :
 * - it contains the parameter called computeConvexHullDecomposition, if set to true, the convex hull is decomposed in convex parts
 * - the parameter concavityThreeshold is a real that parameterizes the convex hull decomposition, the more this parameter will be high, the more the decomposition will be gross
 * - you can visualize the decomposition by setting the parameter drawConvexHullDecomposition to true
 * - another important thing is that its loading is similar to a triangle model because it needs a mesh but it is templated by a rigid mechanical object which position and orientation define also the position and orientation of the BulletConvexHull if positionDefined is set to true and if it is not, the mechanical must be of size 0 and its position/orientation will be computed at runtime as the barycenter of the entry points
 *
 * All collision models have the field called margin. It enlarges the primitives (only in bullet, not in sofa) using the value of margin.
 * For a box, its extents are enlarged, for a sphere or a capsule, their radius are enlarged.
 * Only the BulletConvexHullModel and the BulletTriangleModel use a real margin, meaning that the primitives are not enlarged, and
 * the collision closer than the value margin are detected.
 *
 * You can find some examples in the folder examples:
 * - BulletConvexHullDemo.scn which decomposes two objects show the decomposition:
 * \image html BulletConvexHullDemo.png
 * - GlobalBulletCollision.scn is an example of all primitives falling into a salad bowl, it is using sofa python :
 * \image html GlobalBulletCollision.png
 * - BulletSphere.scn is a very simple scene :
 * \image html BulletSphere.png
 *
 * Dependencies:
 * - You must have bullet 2.82 installed and specify its location in the CMakeCache.txt by filling the paths BULLET_INCLUDE_PATH (for bullet include files) and BULLET_LIB_PATH (for bullet binary files).
 *   WARNING, when installing bullet write in the bullet CMakeLists.txt the line add_definitions(-DBULLET_TRIANGLE_COLLISION)
 * - HACD (a part of bullet for convex hull decomposition) must also be installed at the same locations than bullet
 *
 * Issues:
 * - Currently, the plugin doesn't work well with CompliantContacts
 * - Some improvements to static meshes could be done by using another Bullet mesh class in this case
 */


} // namespace component

} // namespace sofa

#endif

