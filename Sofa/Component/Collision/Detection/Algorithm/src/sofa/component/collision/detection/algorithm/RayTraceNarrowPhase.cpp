/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/component/collision/detection/algorithm/RayTraceNarrowPhase.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/geometry/CubeModel.h>
#include <sofa/component/collision/geometry/TriangleOctreeModel.h>
#include <sofa/component/collision/geometry/TriangleModel.h>

namespace sofa::component::collision::detection::algorithm
{
using namespace sofa::component::collision::geometry;

using sofa::core::collision::TDetectionOutputVector;
using sofa::helper::TriangleOctree;


int RayTraceNarrowPhaseClass = core::RegisterObject("Collision detection using TriangleOctreeModel").add < RayTraceNarrowPhase > ();


void RayTraceNarrowPhase::findPairsVolume (CubeCollisionModel * cm1, CubeCollisionModel * cm2)
{
    /*Obtain the CollisionModel at the lowest level, in this case it must be a TriangleOctreeModel */

    TriangleOctreeModel *tm1 =
            dynamic_cast < TriangleOctreeModel * >(cm1->getLast ());
    TriangleOctreeModel *tm2 =
            dynamic_cast < TriangleOctreeModel * >(cm2->getLast ());
    if (!tm1 || !tm2)
        return;


    /*construct the octree of both models, when it still doesn't exisits */
    if (!tm1->octreeRoot)
    {

        tm1->buildOctree ();
    }

    if (!tm2->octreeRoot)
    {

        tm2->buildOctree ();

    }

    /* get the output vector for a TriangleOctreeModel, TriangleOctreeModel Collision*/
    /*Get the cube representing the bounding box of both Models */
    // sofa::core::collision::DetectionOutputVector *& contacts=outputsMap[std::make_pair(tm1, tm2)];
    core::collision::DetectionOutputVector*& contacts = this->getDetectionOutputs(tm1, tm2);


    if (contacts == nullptr)
    {
        contacts = new
                sofa::core::collision::TDetectionOutputVector <
                        TriangleOctreeModel, TriangleOctreeModel >;

    }




    TDetectionOutputVector < TriangleOctreeModel,
    TriangleOctreeModel > *outputs =
            static_cast < TDetectionOutputVector < TriangleOctreeModel,
            TriangleOctreeModel > *>(contacts);

    Cube cube1 (cm1, 0);
    const Cube cube2 (cm2, 0);


    const auto& minVect2 = cube2.minVect ();
    const auto& maxVect2 = cube2.maxVect ();
    const int size = tm1->getSize ();

    for (int j = 0; j < size; j++)
    {

        /*creates a Triangle for each object being tested */
        Triangle tri1 (tm1, j);


        /*cosAngle will store the angle between the triangle from t1 and his corresponding in t2 */
        double cosAngle;
        /*cosAngle will store the angle between the triangle from t1 and another triangle on t1 that is crossed by the -normal of tri1*/
        double cosAngle2;
        /*resTriangle and resTriangle2 will store the triangle result from the trace method */
        int resTriangle = -1;
        int resTriangle2 = -1;
        sofa::type::Vec3 trianglePoints[4];
        //bool found = false;
        int nPoints = 0;
        sofa::type::Vec3 normau[3];
        /*if it fails to find a correspondence between the triangles it tries the raytracing procedure */
        /*test if this triangle was tested before */

        /*set the triangle as tested */
        const int flags = tri1.flags();

        /*test only the points related to this triangle */
        if (flags & TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P1)
        {
            normau[nPoints] = tm1->pNorms[tri1.p1Index ()];
            trianglePoints[nPoints++] = tri1.p1 ();

        }
        if (flags & TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P2)
        {
            normau[nPoints] = tm1->pNorms[tri1.p2Index ()];
            trianglePoints[nPoints++] = tri1.p2 ();
        }
        if (flags & TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P3)
        {
            normau[nPoints] = tm1->pNorms[tri1.p3Index ()];
            trianglePoints[nPoints++] = tri1.p3 ();
        }

        for (int t = 0; t < nPoints; t++)
        {

            const auto& point = trianglePoints[t];

            if ((point[0] < (minVect2[0]))
                || (point[0] > maxVect2[0] )
                || (point[1] < minVect2[1] )
                || (point[1] > maxVect2[1] )
                || (point[2] < minVect2[2] )
                || (point[2] > maxVect2[2] ))
                continue;
            /*res and res2 will store the point of intercection and the distance from the point */
            TriangleOctree::traceResult res, res2;
            /*search a triangle on t2 */
            resTriangle =
                    tm2->octreeRoot->
                            trace (point /*+ normau[t] * (contactDistance / 2) */ ,
                                   -normau[t], res);
            if (resTriangle == -1)
                continue;
            Triangle triang2 (tm2, resTriangle);
            cosAngle = dot (tri1.n (), triang2.n ());
            if (cosAngle > 0)
                continue;
            /*search a triangle on t1, to be sure that the triangle found on t2 isn't outside the t1 object */

            /*if there is no triangle in t1  that is crossed by the tri1 normal (resTriangle2==-1), it means that t1 is not an object with a closed volume, so we can't continue.
              If the distance from the  point to the triangle on t1 is less than the distance to the triangle on t2 it means that the corresponding point is outside t1, and is not a good point */
            resTriangle2 =
                    tm1->octreeRoot->trace (point, -normau[t], res2);



            if (resTriangle2 == -1 || res2.t < res.t)
            {

                continue;
            }


            Triangle tri3 (tm1, resTriangle2);
            cosAngle2 = dot (tri1.n (), tri3.n ());
            if (cosAngle2 > 0)
                continue;

            const sofa::type::Vec3 Q =
                    (triang2.p1 () * (1.0 - res.u - res.v)) +
                    (triang2.p2 () * res.u) + (triang2.p3 () * res.v);

            outputs->resize (outputs->size () + 1);
            sofa::core::collision::DetectionOutput *detection = &*(outputs->end () - 1);


            detection->elem =
                    std::pair <
                            core::CollisionElementIterator,
                            core::CollisionElementIterator > (tri1, triang2);
            detection->point[0] = point;

            detection->point[1] = Q;

            detection->normal = normau[t];

            detection->value = -(res.t);

            detection->id = tri1.getIndex()*3+t;
            //found = true;

        }

    }

}


void RayTraceNarrowPhase::addCollisionPair (const std::pair <
        core::CollisionModel *,
        core::CollisionModel * >&cmPair)
{
    CubeCollisionModel *cm1 = dynamic_cast < CubeCollisionModel * >(cmPair.first);
    CubeCollisionModel *cm2 = dynamic_cast < CubeCollisionModel * >(cmPair.second);
    if (cm1 && cm2)
    {
        findPairsVolume (cm1, cm2);
        findPairsVolume (cm2, cm1);
    }
}

} // namespace sofa::component::collision::detection::algorithm
