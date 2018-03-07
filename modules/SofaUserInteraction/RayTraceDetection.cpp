/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaMeshCollision/RayTriangleIntersection.h>

#include <SofaBaseCollision/Sphere.h>
#include <SofaMeshCollision/TriangleModel.inl>
#include <SofaGeneralMeshCollision/TriangleOctreeModel.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaMeshCollision/Line.h>
#include <SofaMeshCollision/Point.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/core/ObjectFactory.h>
#include <SofaUserInteraction/RayTraceDetection.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Node.h>
#include <map>
#include <queue>
#include <stack>
#include <sofa/helper/system/gl.h>

#include <sofa/helper/system/thread/CTime.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace collision;
using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;

SOFA_DECL_CLASS (RayTraceDetection)
int RayTraceDetectionClass =
    core::
    RegisterObject
    ("Collision detection using TriangleOctreeModel").add <
    RayTraceDetection > ();

using namespace core::objectmodel;

RayTraceDetection::
RayTraceDetection ():bDraw (initData
            (&bDraw, false, "draw",
                    "enable/disable display of results"))
{
}


void RayTraceDetection::findPairsVolume (CubeModel * cm1, CubeModel * cm2)
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


    if (contacts == NULL)
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
    Cube cube2 (cm2, 0);


    const Vector3 & minVect2 = cube2.minVect ();
    const Vector3 & maxVect2 = cube2.maxVect ();
    int size = tm1->getSize ();

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
        Vector3 trianglePoints[4];
        //bool found = false;
        int nPoints = 0;
        Vector3 normau[3];
        /*if it fails to find a correspondence between the triangles it tries the raytracing procedure */
        /*test if this triangle was tested before */

        /*set the triangle as tested */
        int flags = tri1.flags();

        /*test only the points related to this triangle */
        if (flags & TriangleModel::FLAG_P1)
        {
            normau[nPoints] = tm1->pNorms[tri1.p1Index ()];
            trianglePoints[nPoints++] = tri1.p1 ();

        }
        if (flags & TriangleModel::FLAG_P2)
        {
            normau[nPoints] = tm1->pNorms[tri1.p2Index ()];
            trianglePoints[nPoints++] = tri1.p2 ();
        }
        if (flags & TriangleModel::FLAG_P3)
        {
            normau[nPoints] = tm1->pNorms[tri1.p3Index ()];
            trianglePoints[nPoints++] = tri1.p3 ();
        }

        for (int t = 0; t < nPoints; t++)
        {

            Vector3 point = trianglePoints[t];

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

            Vector3 Q =
                (triang2.p1 () * (1.0 - res.u - res.v)) +
                (triang2.p2 () * res.u) + (triang2.p3 () * res.v);

            outputs->resize (outputs->size () + 1);
            DetectionOutput *detection = &*(outputs->end () - 1);


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

void RayTraceDetection::addCollisionModel (core::CollisionModel * cm)
{
    if (cm->empty ())
        return;
    for (sofa::helper::vector < core::CollisionModel * >::iterator it =
            collisionModels.begin (); it != collisionModels.end (); ++it)
    {
        core::CollisionModel * cm2 = *it;
        if (!cm->isSimulated() && !cm2->isSimulated())
            continue;
        if (!cm->canCollideWith (cm2))
            continue;

        bool swapModels = false;
        core::collision::ElementIntersector* intersector = intersectionMethod->findIntersector(cm, cm2, swapModels);
        if (intersector == NULL)
            continue;

        core::CollisionModel* cm1 = (swapModels?cm2:cm);
        cm2 = (swapModels?cm:cm2);


        // Here we assume a single root element is present in both models
        if (intersector->canIntersect (cm1->begin (), cm2->begin ()))
        {

            cmPairs.push_back (std::make_pair (cm1, cm2));
        }
    }
    collisionModels.push_back (cm);
}

void RayTraceDetection::addCollisionPair (const std::pair <
        core::CollisionModel *,
        core::CollisionModel * >&cmPair)
{
    CubeModel *cm1 = dynamic_cast < CubeModel * >(cmPair.first);
    CubeModel *cm2 = dynamic_cast < CubeModel * >(cmPair.second);
    if (cm1 && cm2)
    {
        //ctime_t t0, t1, t2;
        /*t0 = */CTime::getRefTime ();
        findPairsVolume (cm1, cm2);


        /*t1 = */CTime::getRefTime ();

        findPairsVolume (cm2, cm1);
        /*t2 = */CTime::getRefTime ();
    }
}

void RayTraceDetection::draw (const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!bDraw.getValue ())
        return;

    glDisable (GL_LIGHTING);
    glColor3f (1.0, 0.0, 1.0);
    glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
    glLineWidth (3);
    glPointSize (5);

    const DetectionOutputMap& outputsMap = this->getDetectionOutputs();

    for (DetectionOutputMap::const_iterator it = outputsMap.begin ();
            it != outputsMap.end (); ++it)
    {
        TDetectionOutputVector < TriangleOctreeModel,
                               TriangleOctreeModel > *outputs =
                                       static_cast <
                                       sofa::core::collision::TDetectionOutputVector <
                                       TriangleOctreeModel, TriangleOctreeModel > *>(it->second);
        for (TDetectionOutputVector < TriangleOctreeModel,
                TriangleOctreeModel >::iterator it2 = (outputs)->begin ();
                it2 != outputs->end (); ++it2)
        {
            glBegin (GL_LINES);
            glVertex3d (it2->point[0][0], it2->point[0][1],
                    it2->point[0][2]);
            glVertex3d (it2->point[1][0], it2->point[1][1],
                    it2->point[1][2]);
            glEnd ();
            serr << it2->point[0] << " " << it2->
                    point[0] << sendl;
            it2->elem.first.draw(vparams);
            it2->elem.second.draw(vparams);
        }
    }
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
    glLineWidth (1);
    glPointSize (1);
#endif /* SOFA_NO_OPENGL */
}

}				// namespace collision

}				// namespace component

}				// namespace sofa
