/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/

#include <sofa/component/collision/Sphere.h>
#include <sofa/component/collision/Triangle.h>
#include <sofa/component/collision/TriangleOctreeModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/component/collision/Line.h>
#include <sofa/component/collision/Point.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/core/componentmodel/collision/DetectionOutput.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/collision/OctreeDetection.h>
#include <sofa/simulation/tree/GNode.h>
#include <map>
#include <queue>
#include <stack>
#include <GL/gl.h>
#include <GL/glut.h>

#include <sofa/helper/system/thread/CTime.h>

/* for debugging the collision method */
#ifdef _WIN32
#include <windows.h>
#endif

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::collision;
using namespace collision;
using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;

SOFA_DECL_CLASS (Octree)
int OctreeDetectionClass =
    core::
    RegisterObject
    ("Collision detection using extensive pair-wise tests").add <
    OctreeDetection > ();

using namespace core::objectmodel;

OctreeDetection::
OctreeDetection ():bDraw (dataField
            (&bDraw, false, "draw",
                    "enable/disable display of results"))
{
}
void OctreeDetection::findPairsSurface (CubeModel * cm1,
        CubeModel * cm2)
{
    core::CollisionModel * finalcm1 = cm1->getLast ();
    core::CollisionModel * finalcm2 = cm2->getLast ();
//      std::cerr << "Model " << gettypename (typeid (*finalcm1)) << std::
//        endl;
//      std::cerr << "Model " << gettypename (typeid (*finalcm2)) << std::
//        endl;

    DetectionOutputVector & outputs =
        outputsMap[std::make_pair (finalcm1, finalcm2)];
    Cube cube1 (cm1, 0);
    Cube cube2 (cm2, 0);
    const Vector3 *minVect = &cube1.minVect ();
    const Vector3 *maxVect = &cube1.maxVect ();

//      Vector3 bbmax;
//      Vector3 bbmin;
//      for (int i=0;i<3;i++)
//      {
//              bbmax[i]=(minVect1[i]>minVect2[i]?minVect1[i]:minVect2[i]);
//              bbmax[i]=(maxVect1[i]<maxVect2[i]?maxVect1[i]:maxVect2[i]);
//      }
    TriangleOctreeModel *tm =
        dynamic_cast < TriangleOctreeModel * >(finalcm1);
    PointModel *pm = dynamic_cast < PointModel * >(finalcm2);

    if (!(tm && pm))
    {
        minVect = &cube2.minVect ();
        maxVect = &cube2.maxVect ();

        tm = dynamic_cast < TriangleOctreeModel * >(finalcm2);
        pm = dynamic_cast < PointModel * >(finalcm1);
        if (!(tm && pm))
            return;
    }
//std::cerr<<"get inteersector"<<std::endl;
    core::componentmodel::collision::ElementIntersector * intersector =
        NULL;
    if (pm && tm)
        intersector = intersectionMethod->findIntersector (pm, tm);
    if (!intersector)
        return;


    if (!tm->octreeRoot)
        tm->buildOctree ();


    core::componentmodel::behavior::MechanicalState < Vec3Types >
    *mstate = pm->getMechanicalState ();
    Vec3Types::VecCoord * points = mstate->getX ();

    for (int j = 0; j < points->size (); j++)
    {
        Point pt (pm, j);
        if (pt.p ()[0] < (*maxVect)[0] && pt.p ()[0] > (*minVect)[0] &&
            pt.p ()[1] < (*maxVect)[1] && pt.p ()[1] > (*minVect)[1] &&
            pt.p ()[1] < (*maxVect)[1] && pt.p ()[1] > (*minVect)[1])
        {
//      std::cerr<<pt.p()<<std::endl;
            vector < TriangleOctree * >octree1;
            octree1.push_back (tm->octreeRoot);
            while (octree1.size ())
            {
                TriangleOctree & t1 = *octree1.back ();
                octree1.pop_back ();
                //TriangleOctree & t2 = *tm2->octreeRoot;

                if (t1.objects.size ())
                {
//std::cerr<<"Octree2 size!"<<std::endl;
                    for (int o1 = 0; o1 < t1.objects.size (); o1++)
                    {
                        //std::cerr<<"elem Pairs added"<<std::endl;
                        TriangleModel *tmm1 =
                            dynamic_cast < TriangleModel * >(tm);
                        Triangle tri1 (tmm1, (int) t1.objects[o1]);
                        //Triangle  tri2(tmm2,(int) t2.objects[o2]);

                        if (intersector->canIntersect (pt, tri1))
                        {
                            intersector->intersect (pt, tri1, outputs);
                            //elemPairs.push_back (std::make_pair (pt, tri1));
                        }
                    }
                }




                double size2 = t1.size / 2;
                int dx = (pt.p ()[0] >= (t1.x + size2)) ? 1 : 0;
                int dy = (pt.p ()[1] >= (t1.y + size2)) ? 1 : 0;
                int dz = (pt.p ()[2] >= (t1.z + size2)) ? 1 : 0;
                int i = dx * 4 + dy * 2 + dz;
                if (t1.childVec[i])
                    octree1.push_back (t1.childVec[i]);

            }
        }
    }

}
void OctreeDetection::findPairsSurfaceTriangleSimple (CubeModel * cm1,
        CubeModel * cm2)
{
    core::CollisionModel * finalcm1 = cm1->getLast ();
    core::CollisionModel * finalcm2 = cm2->getLast ();
    std::cerr << "Model " << gettypename (typeid (*finalcm1)) << std::
            endl;
    std::cerr << "Model " << gettypename (typeid (*finalcm2)) << std::
            endl;

    DetectionOutputVector & outputs =
        outputsMap[std::make_pair (finalcm1, finalcm2)];
    Cube cube1 (cm1, 0);
    Cube cube2 (cm2, 0);
    const Vector3 *minVect1 = &cube1.minVect ();
    const Vector3 *maxVect1 = &cube1.maxVect ();

    const Vector3 *minVect2 = &cube2.minVect ();
    const Vector3 *maxVect2 = &cube2.maxVect ();

//      Vector3 bbmax;
//      Vector3 bbmin;
//      for (int i=0;i<3;i++)
//      {
//              bbmax[i]=(minVect1[i]>minVect2[i]?minVect1[i]:minVect2[i]);
//              bbmax[i]=(maxVect1[i]<maxVect2[i]?maxVect1[i]:maxVect2[i]);
//      }
    TriangleOctreeModel *tm1 =
        dynamic_cast < TriangleOctreeModel * >(finalcm1);
    TriangleOctreeModel *tm2 =
        dynamic_cast < TriangleOctreeModel * >(finalcm2);

    if (!tm1 || !tm2)
        return;
//std::cerr<<"get inteersector"<<std::endl;
    core::componentmodel::collision::ElementIntersector * intersector =
        NULL;
    if (tm1 && tm2)
        intersector = intersectionMethod->findIntersector (tm1, tm2);
    if (!intersector)
        return;
    for (int i = 0; i < tm1->elems.size (); i++)
        for (int j = 0; j < tm2->elems.size (); j++)
        {
            Triangle tri1 (tm1, (int) i);
            Triangle tri2 (tm2, (int) j);
            if (intersector->canIntersect (tri1, tri2))
            {
                //std::cerr<<"intersect triangle"<<std::endl;
                intersector->intersect (tri1, tri2, outputs);
            }


        }

}


void OctreeDetection::findPairsSurfaceTriangle (CubeModel * cm1,
        CubeModel * cm2)
{
    core::CollisionModel * finalcm1 = cm1->getLast ();
    core::CollisionModel * finalcm2 = cm2->getLast ();
    //std::cerr << "Model " << gettypename (typeid (*finalcm1)) << std::
    //endl;
    //std::cerr << "Model " << gettypename (typeid (*finalcm2)) << std::
    //endl;

    DetectionOutputVector & outputs =
        outputsMap[std::make_pair (finalcm1, finalcm2)];
    Cube cube1 (cm1, 0);
    Cube cube2 (cm2, 0);
    const Vector3 *minVect1 = &cube1.minVect ();
    const Vector3 *maxVect1 = &cube1.maxVect ();

    const Vector3 *minVect2 = &cube2.minVect ();
    const Vector3 *maxVect2 = &cube2.maxVect ();

//      Vector3 bbmax;
//      Vector3 bbmin;
//      for (int i=0;i<3;i++)
//      {
//              bbmax[i]=(minVect1[i]>minVect2[i]?minVect1[i]:minVect2[i]);
//              bbmax[i]=(maxVect1[i]<maxVect2[i]?maxVect1[i]:maxVect2[i]);
//      }
    TriangleOctreeModel *tm1 =
        dynamic_cast < TriangleOctreeModel * >(finalcm1);
    TriangleOctreeModel *tm2 =
        dynamic_cast < TriangleOctreeModel * >(finalcm2);

    if (!tm1 || !tm2)
        return;
//std::cerr<<"get inteersector"<<std::endl;
    core::componentmodel::collision::ElementIntersector * intersector =
        NULL;
    if (tm1 && tm2)
        intersector = intersectionMethod->findIntersector (tm1, tm2);
    if (!intersector)
        return;

    vector < TriangleOctree * >octree1;
    vector < TriangleOctree * >octree2;
    if (!tm1->octreeRoot)
    {
        std::cerr << "build" << std::endl;
        tm1->buildOctree ();
    }

    if (!tm2->octreeRoot)
    {
        std::cerr << "build" << std::endl;
        tm2->buildOctree ();
    }
    octree1.push_back (tm1->octreeRoot);
    vector < bool > testedTriangle (tm1->elems.size (), false);
    while (octree1.size ())
    {
        TriangleOctree & t1 = *octree1.back ();
        octree1.pop_back ();

        vector < TriangleOctree * >octree2;
        octree2.push_back (tm2->octreeRoot);
        if (t1.objects.size ()
            &&
            !(((*minVect2)[0] > t1.x + t1.size || t1.x > (*maxVect2)[0])
                    || ((*minVect2)[1] > t1.y + t1.size
                            || t1.y > (*maxVect2)[1])
                    || ((*minVect2)[2] > t1.z + t1.size
                            || t1.z > (*maxVect2)[2])))
        {
            while (octree2.size ())
            {
                TriangleOctree & t2 = *octree2.back ();
                octree2.pop_back ();
                //TriangleOctree & t2 = *tm2->octreeRoot;

                if (t2.objects.size ())
                {
                    for (int o1 = 0; o1 < t1.objects.size (); o1++)
                        for (int o2 = 0; o2 < t2.objects.size (); o2++)
                        {
                            //std::cerr<<"elem Pairs added"<<std::endl;

                            TriangleModel *tmm1 =
                                dynamic_cast < TriangleModel * >(tm1);
                            TriangleModel *tmm2 =
                                dynamic_cast < TriangleModel * >(tm2);
                            Triangle tri1 (tmm1, (int) t1.objects[o1]);
                            Triangle tri2 (tmm2, (int) t2.objects[o2]);
                            if (intersector->intersect (tri1, tri2,
                                    outputs) <= 0)

                            {
                                //continue;
//std::cerr<<"ray"<<std::endl;
                                if (1 && testedTriangle[t1.objects[o1]])
                                    continue;
                                testedTriangle[t1.objects[o1]] = true;
                                double cosAngle;
                                int resTriangle = -1;
                                int resTriangle2 = -1;

                                Vector3 point;
                                Vector3 trianglePoints[3];
                                int nPoints = 0;
                                if (tri1.flags () & TriangleModel::FLAG_P1)
                                    trianglePoints[nPoints++] = tri1.p1 ();
                                if (tri1.flags () & TriangleModel::FLAG_P2)
                                    trianglePoints[nPoints++] = tri1.p2 ();
                                if (tri1.flags () & TriangleModel::FLAG_P3)
                                    trianglePoints[nPoints++] = tri1.p3 ();


                                for (int t = 0; t < nPoints; t++)
                                {

                                    point = trianglePoints[t];
                                    //point=(tri1.p1()+tri1.p2()+tri1.p3())/3;


                                    TriangleOctree::traceResult res;
                                    TriangleOctree::traceResult res2;
                                    resTriangle =
                                        t2.trace (point, -tri1.n (), res);
                                    resTriangle2 =
                                        t1.trace (point, -tri1.n (), res2);
                                    if (resTriangle2 != -1
                                        && resTriangle2 != t1.objects[o1]
                                        && res2.t < res.t)
                                        resTriangle = -1;
                                    if (resTriangle == -1)
                                        continue;

                                    Triangle *triang2 =
                                        new Triangle (tm2, resTriangle);

                                    cosAngle =
                                        dot (tri1.n (), triang2->n ());
                                    if (cosAngle < 0)
                                    {
                                        int indice;
                                        double times;
                                        if (tri1.n ()[0])
                                            indice = 0;
                                        else if (tri1.n ()[1])
                                            indice = 1;
                                        else if (tri1.n ()[2])
                                            indice = 2;
                                        Vector3 Q =
                                            (1 - res.u -
                                                    res.v) * triang2->p1 () +
                                            res.u * triang2->p2 () +
                                            res.v * triang2->p3 ();
                                        outputs.resize (outputs.size () +
                                                1);
                                        DetectionOutput *detection =
                                            &*(outputs.end () - 1);
                                        detection->elem =
                                            std::pair <
                                            core::CollisionElementIterator,
                                            core::CollisionElementIterator >
                                            (tri1, *triang2);
                                        detection->point[0] = point;

                                        times =
                                            (Q[indice] -
                                                    point[indice]) /
                                            tri1.n ()[indice];
                                        detection->point[1] =
                                            point - tri1.n () * times;
                                        //detection->normal = (point - Q) / res.t;
                                        detection->normal = tri1.n ();
                                        //std::cerr<<"point normal"<< tri1.n()<<"norm2:"<<detection->normal<<std::endl;
                                        detection->distance = res.t;

                                    }
                                }
                            }
                            else
                            {
                                testedTriangle[t1.objects[o1]] = true;
                            }
                        }
                }
                if (t2.size <= t1.size)
                {
                    for (int i = 0; i < 8; i++)
                        if (t2.childVec[i])
                            octree2.push_back (t2.childVec[i]);


                }
                else
                {
                    double size2 = t2.size / 2;
                    int dx = (t1.x >= (t2.x + size2)) ? 1 : 0;
                    int dy = (t1.y >= (t2.y + size2)) ? 1 : 0;
                    int dz = (t1.z >= (t2.z + size2)) ? 1 : 0;
                    int i = dx * 4 + dy * 2 + dz;
                    if (t2.childVec[i])
                        octree2.push_back (t2.childVec[i]);
                }
            }
        }

        for (int i = 0; i < 8; i++)
            if (t1.childVec[i])
            {
                /*if (t1.childVec[i]->x < ((*minVect2)[0]-2)
                   && t1.childVec[i]->y <( (*minVect2)[2]-2)
                   && t1.childVec[i]->z < ((*minVect2)[3]-2)
                   && (t1.childVec[i]->x+t1.childVec[i]->size) > ((*maxVect2)[0]+2)
                   && (t1.childVec[i]->y+t1.childVec[i]->size) > ((*maxVect2)[2]+2)
                   && (t1.childVec[i]->z+t1.childVec[i]->size) > ((*maxVect2)[3]+2))
                 */
                {
                    octree1.push_back (t1.childVec[i]);
                }
            }
    }
}

void OctreeDetection::findPairsVolume (CubeModel * cm1, CubeModel * cm2)
{
//      std::cerr<<"Model "<<gettypename(typeid(*cm1))<<std::endl;
//      std::cerr<<"Model "<<gettypename(typeid(*cm2))<<std::endl;
    core::CollisionModel * finalcm1 = cm1->getLast ();
    core::CollisionModel * finalcm2 = cm2->getLast ();
    DetectionOutputVector & outputs =
        outputsMap[std::make_pair (finalcm1, finalcm2)];
    Cube cube1 (cm1, 0);
    Cube cube2 (cm2, 0);
    const Vector3 *minVect = &cube1.minVect ();
    const Vector3 *maxVect = &cube1.maxVect ();

//      Vector3 bbmax;
//      Vector3 bbmin;
//      for (int i=0;i<3;i++)
//      {
//              bbmax[i]=(minVect1[i]>minVect2[i]?minVect1[i]:minVect2[i]);
//              bbmax[i]=(maxVect1[i]<maxVect2[i]?maxVect1[i]:maxVect2[i]);
//      }
    TriangleOctreeModel *tm1 =
        dynamic_cast < TriangleOctreeModel * >(finalcm1);
    TriangleOctreeModel *tm2 =
        dynamic_cast < TriangleOctreeModel * >(finalcm2);

    if (!(tm1 && tm2))
    {
        return;
    }

    core::componentmodel::collision::ElementIntersector * intersector =
        NULL;
    if (tm1 && tm2)
        intersector = intersectionMethod->findIntersector (tm1, tm2);
    if (!intersector)
        return;
    if (!tm1->octreeRoot)
        tm1->buildOctree ();


    if (!tm2->octreeRoot)
        tm2->buildOctree ();


    core::componentmodel::behavior::MechanicalState < Vec3Types >
    *mstate = tm1->getMechanicalState ();
    Vec3Types::VecCoord * points = mstate->getX ();
    vector < bool > testedPoints;
    TriangleOctree::traceResult res;
    TriangleOctree::traceResult res2;
    testedPoints.resize (points->size ());
    for (int j = 0; j < tm1->elems.size (); j++)
    {
        int trianglePoints[3];
        trianglePoints[0] = tm1->elems[j].i1;
        trianglePoints[1] = tm1->elems[j].i2;
        trianglePoints[2] = tm1->elems[j].i3;
        trianglePoints[0];
        testedPoints[trianglePoints[0]];

        Vector3 point (0, 0, 0);
        for (int k = 0; k < 3; k++)
        {
            point = (*points)[trianglePoints[k]];
//
//              point += (*points)[trianglePoints[k]];
//            }
//          point /= 3;
            if ( /*(!testedPoints[trianglePoints[k]])&& */ (point[0]) <
                    (*maxVect)[0]
                    && (point[1]) < (*maxVect)[1]
                    && (point[2]) < (*maxVect)[2]
                    && (point[0]) > (*minVect)[0]
                    && (point[1]) > (*minVect)[1]
                    && (point[2]) > (*minVect)[2])
            {

                int resTriangle = -1;
                int resTriangle2 = -1;
                double cosAngle, cosAngle2;

                //testedPoints[trianglePoints[k]]=true;

                resTriangle =
                    tm2->octreeRoot->trace (point, -(tm1->elems[j].normal),
                            res);
                resTriangle2 =
                    tm1->octreeRoot->trace (point, -(tm1->elems[j].normal),
                            res2);
                if (resTriangle2 == -1 || resTriangle2 != j
                    && res2.t < res.t)
                {
                    resTriangle = -1;
                    //std::cerr<<j<<" res"<<resTriangle2<<" t:"<<res.t<<" t2:"<<res2.t<<std::endl;
                }
                //if(res.t>20)  resTriangle=-1;
                //      resTriangle=tm2->octreeRoot->trace(Vector3(0,10,0),Vector3(0,-1,0),res);
                //std::cerr<<"rest:"<<resTriangle<<std::endl;
                //std::cerr<<"achou par0        "<<point<<" norm "<<-tm1->elems[j].normal<< std::endl;
                if (resTriangle == -1)
                    continue;
                cosAngle =
                    dot (tm1->elems[j].normal,
                            tm2->elems[resTriangle].normal);
                cosAngle2 =
                    dot (tm1->elems[j].normal,
                            tm1->elems[resTriangle2].normal);
                if (cosAngle < 0 /*&&cosAngle2<0 */ )
                {
                    int index = 0;
                    double ratio;
                    //std::cerr<<"t1        "<<tm1->getName()<<"t2 "<< tm2->getName()<<std::endl;
                    Triangle *triang1 = new Triangle (tm2, resTriangle);
                    Vector3 Q =
                        (1 - res.u - res.v) * triang1->p1 () +
                        res.u * triang1->p2 () + res.v * triang1->p3 ();
                    outputs.resize (outputs.size () + 1);
                    if (tm1->elems[resTriangle2].normal[0])
                        index = 0;
                    else if (tm1->elems[resTriangle2].normal[1])
                        index = 1;
                    else if (tm1->elems[resTriangle2].normal[2])
                        index = 2;

                    //std::cerr<<Q<<"u  "<<res.u<<" v "<<res.v<<std::endl;
                    DetectionOutput *detection = &*(outputs.end () - 1);

                    detection->elem =
                        std::pair < core::CollisionElementIterator,
                        core::CollisionElementIterator >
                        (*(new Triangle (tm1, j)), *triang1);
                    detection->point[0] = point;



                    ratio =
                        (point -
                                Q)[index] / tm1->elems[resTriangle2].normal[index];
                    detection->point[1] =
                        point - tm1->elems[resTriangle2].normal * ratio;
                    detection->normal = (point - detection->point[1]);

                    detection->distance = detection->normal.norm ();
                    detection->normal /= detection->distance;


                    //      sleep(1);
                }
                //}
            }
        }
    }
    //sleep(1);

}

void OctreeDetection::addCollisionModel (core::CollisionModel * cm)
{
    if (cm->empty ())
        return;
    for (sofa::helper::vector < core::CollisionModel * >::iterator it =
            collisionModels.begin (); it != collisionModels.end (); ++it)
    {
        core::CollisionModel * cm2 = *it;
        if (cm->isStatic () && cm2->isStatic ())
            continue;
        if (!cm->canCollideWith (cm2))
            continue;
        core::componentmodel::collision::ElementIntersector *
        intersector = intersectionMethod->findIntersector (cm, cm2);
        if (intersector == NULL)
            continue;

        // // Here we assume multiple root elements are present in both models
        // bool collisionDetected = false;
        // core::CollisionElementIterator begin1 = cm->begin();
        // core::CollisionElementIterator end1 = cm->end();
        // core::CollisionElementIterator begin2 = cm2->begin();
        // core::CollisionElementIterator end2 = cm2->end();
        // for (core::CollisionElementIterator it1 = begin1; it1 != end1; ++it1)
        // {
        //     for (core::CollisionElementIterator it2 = begin2; it2 != end2; ++it2)
        //     {
        //         //if (!it1->canCollideWith(it2)) continue;
        //         if (intersector->canIntersect(it1, it2))
        //         {
        //             collisionDetected = true;
        //             break;
        //         }
        //     }
        //     if (collisionDetected) break;
        // }
        // if (collisionDetected)

        // Here we assume a single root element is present in both models
        if (intersector->canIntersect (cm->begin (), cm2->begin ()))
        {

            //std::cout << "Broad phase "<<cm->getLast()->getName()<<" - "<<cm2->getLast()->getName()<<std::endl;
            cmPairs.push_back (std::make_pair (cm, cm2));
        }
    }
    collisionModels.push_back (cm);
}

void OctreeDetection::addCollisionPair (const std::pair <
        core::CollisionModel *,
        core::CollisionModel * >&cmPair)
{
    typedef std::pair < std::pair < core::CollisionElementIterator,
            core::CollisionElementIterator >,
            std::pair < core::CollisionElementIterator,
            core::CollisionElementIterator > >TestPair;


    CubeModel *cm1 = dynamic_cast < CubeModel * >(cmPair.first);	//->getNext();
    CubeModel *cm2 = dynamic_cast < CubeModel * >(cmPair.second);	//->getNext();
    //std::cerr<<
    if (cm1 && cm2)
    {
        ctime_t t0, t1, t2;
        t0 = CTime::getRefTime ();
        findPairsSurfaceTriangle (cm1, cm2);
        t1 = CTime::getRefTime ();
        findPairsSurfaceTriangle (cm2, cm1);
        t2 = CTime::getRefTime ();
        std::cerr << "Octree construction:" << (t1 -
                t0) /
                ((double) CTime::getRefTicksPerSec () /
                        1000) << " traceVolume:" << (t2 -
                                t1) /
                ((double) CTime::getRefTicksPerSec () / 1000) << std::endl;

    }
}

void OctreeDetection::draw ()
{
    if (!bDraw.getValue ())
        return;

    glDisable (GL_LIGHTING);
    glColor3f (1.0, 0.0, 1.0);
    glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
    glLineWidth (3);
    glPointSize (5);

    for (DetectionOutputMap::iterator it = outputsMap.begin ();
            it != outputsMap.end (); it++)
    {
        DetectionOutputVector & outputs = it->second;
        for (DetectionOutputVector::iterator it2 = outputs.begin ();
                it2 != outputs.end (); it2++)
        {
            it2->elem.first.draw ();
            it2->elem.second.draw ();
        }
    }
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
    glLineWidth (1);
    glPointSize (1);
}

}				// namespace collision

}				// namespace component

}				// namespace sofa
