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
#include <SofaMeshCollision/TriangleModel.inl>
#include <SofaGeneralMeshCollision/TriangleOctree.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaMeshCollision/RayTriangleIntersection.h>
#include <SofaMeshCollision/RayTriangleIntersection.h>

#include <SofaMeshCollision/Triangle.h>
#include <sofa/core/CollisionElement.h>
#include <sofa/core/ObjectFactory.h>
#include <vector>
#include <sofa/helper/system/thread/CTime.h>

#include <cmath>
#include <sofa/helper/system/gl.h>


namespace sofa
{

namespace component
{

namespace collision
{

using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;

TriangleOctree::~TriangleOctree()
{
    for(int i=0; i<8; i++)
    {
        if(childVec[i])
        {
            delete childVec[i];
            childVec[i]=NULL;
        }
    }
}

void TriangleOctree::draw (const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    defaulttype::Vector3 center;
    if ( objects.size ())
    {
        center =
            (defaulttype::Vector3 (x, y, z) + defaulttype::Vector3 (size / 2, size / 2, size / 2));
        glPushMatrix ();
        glTranslatef ((float)center[0], (float)center[1], (float)center[2]);
        vparams->drawTool()->setPolygonMode(0, false);
        vparams->drawTool()->drawCube(size, sofa::defaulttype::Vec4f(0.5, 0.5, 0.5, 1.0));
        glPopMatrix ();

        vparams->drawTool()->setPolygonMode(0, true);
    }
    for (int i = 0; i < 8; i++)
    {
        if (childVec[i])
            childVec[i]->draw(vparams);
    }
#endif /* SOFA_NO_OPENGL */
}

void TriangleOctree::insert (double _x, double _y, double _z,
        double inc, int t)
{
    if (inc >= size)
    {
        objects.push_back (t);
    }
    else
    {
        double size2 = size / 2;
        int dx = (_x >= (x + size2)) ? 1 : 0;
        int dy = (_y >= (y + size2)) ? 1 : 0;
        int dz = (_z >= (z + size2)) ? 1 : 0;

        int i = dx * 4 + dy * 2 + dz;
        if (!childVec[i])
        {
            is_leaf = false;
            childVec[i] =
                new TriangleOctree (tm, x + dx * size2, y + dy * size2,
                        z + dz * size2, size2);
        }

        childVec[i]->insert (_x, _y, _z, inc, t);
    }
}

inline
unsigned int choose_next (double x, double y, double z,
        unsigned int a, unsigned int b,
        unsigned int c)
{
    if (x < y)
    {
        if (x < z)
            return a;
        else
            return c;
    }
    else
    {
        if (y < z)
            return b;
        else
            return c;
    }
}

int TriangleOctree::nearestTriangle (int minIndex,
        const defaulttype::Vector3 & origin,
        const defaulttype::Vector3 & direction, traceResult &result)
{
    static RayTriangleIntersection intersectionSolver;
    defaulttype::Vector3 P;
    const TriangleOctreeRoot::VecCoord& pos = *tm->octreePos;
    //Triangle t1 (tm, minIndex);
    TriangleOctreeRoot::Tri t1 = (*tm->octreeTriangles)[minIndex];
    double minDist;
    SReal t, u, v;
    //if (intersectionSolver.NewComputation (&t1, origin, direction, t, u, v))
    if (intersectionSolver.NewComputation (pos[t1[0]], pos[t1[1]], pos[t1[2]], origin, direction, t, u, v))
    {
        result.u = u;
        result.v = v;
        result.t = minDist = t;
        result.tid = minIndex;
    }
    else
    {
        minDist = 10e8;
        minIndex = -1;
    }
    for (unsigned int i = 0; i < objects.size (); i++)
    {
        //Triangle t2 (tm, objects[i]);
        TriangleOctreeRoot::Tri t2 = (*tm->octreeTriangles)[objects[i]];
        if (!intersectionSolver.
            NewComputation (pos[t2[0]], pos[t2[1]], pos[t2[2]], origin, direction, t, u, v))
            continue;

        if (t < minDist)
        {
            result.u = u;
            result.v = v;
            result.t = minDist = t;
            result.tid = minIndex = objects[i];
        }
    }

    if (minIndex < 0) return minIndex;

#if 0
    /*u=p2 v=p3 p1=1-u+v*/
    int pointR=-1;
    //Triangle t3(tm,minIndex);
    TriangleOctreeRoot::Tri t3 = (*tm->octreeTriangles)[minIndex];
    if(result.u>0.99)
        pointR=t3[1];
    if(result.v>0.99)
        pointR=t3[2];
    if((result.u+result.v)<0.01)
        pointR=t3[0];
    if(pointR!=-1)
    {
        double cosAng=dot(direction,t3.n());

        for(unsigned int i=0; i<tm->pTri[pointR].size(); i++)
        {
            Triangle t2(tm,tm->pTri[pointR][i]);
            double cosAng2=dot(direction,t2.n());
            if(cosAng2>cosAng)
            {
                cosAng=cosAng2;
                minIndex=tm->pTri[pointR][i];
            }
        }
    }
#endif
    result.tid = minIndex;
    return minIndex;
}

int TriangleOctree::trace (const defaulttype::Vector3 & origin,
        const defaulttype::Vector3 & direction, double tx0,
        double ty0, double tz0, double tx1,
        double ty1, double tz1, unsigned int a,
        unsigned int b,defaulttype::Vector3 &origin1,defaulttype::Vector3 &direction1,traceResult &result)
{
    if (tx1 < 0.0 || ty1 < 0.0 || tz1 < 0.0)
        return -1;

    if (is_leaf)
    {
        if (objects.size ())
        {
            return nearestTriangle (objects[0], origin1, direction1,result);
        }
        else
        {
            return -1;
        }
    }

    const double &x0 = x;
    const double &y0 = y;
    const double &z0 = z;
    const double &x1 = x + size;
    const double &y1 = y + size;
    const double &z1 = z + size;
    const double INF = 1e9;

    double txm=0, tym=0, tzm=0;

    switch (b)
    {
    case 0:
        txm = 0.5 * (tx0 + tx1);
        tym = 0.5 * (ty0 + ty1);
        tzm = 0.5 * (tz0 + tz1);
        break;
    case 1:
        txm = 0.5 * (tx0 + tx1);
        tym = 0.5 * (ty0 + ty1);
        tzm = origin[2] < 0.5 * (z0 + z1) ? +INF : -INF;
        break;
    case 2:
        txm = 0.5 * (tx0 + tx1);
        tym = origin[1] < 0.5 * (y0 + y1) ? +INF : -INF;
        tzm = 0.5 * (tz0 + tz1);
        break;
    case 3:
        txm = 0.5 * (tx0 + tx1);
        tym = origin[1] < 0.5 * (y0 + y1) ? +INF : -INF;
        tzm = origin[2] < 0.5 * (z0 + z1) ? +INF : -INF;
        break;
    case 4:
        txm = origin[0] < 0.5 * (x0 + x1) ? +INF : -INF;
        tym = 0.5 * (ty0 + ty1);
        tzm = 0.5 * (tz0 + tz1);
        break;
    case 5:
        txm = (origin[0] < (0.5 * (x0 + x1)) )? +INF : -INF;
        tym = 0.5 * (ty0 + ty1);
        tzm = (origin[2] < (0.5 * (z0 + z1) ))? +INF : -INF;
        break;
    case 6:
        txm = origin[0] < 0.5 * (x0 + x1) ? +INF : -INF;
        tym = origin[1] < 0.5 * (y0 + y1) ? +INF : -INF;
        tzm = 0.5 * (tz0 + tz1);
        break;
    case 7:
        txm = origin[0] < 0.5 * (x0 + x1) ? +INF : -INF;
        tym = origin[1] < 0.5 * (y0 + y1) ? +INF : -INF;
        tzm = origin[2] < 0.5 * (z0 + z1) ? +INF : -INF;
        break;
    default:
        assert (!"Internal failure.");
    }
    unsigned int current_octant = 0;

    if (tx0 > ty0)
    {
        if (tx0 > tz0)
        {
            // max(tx0, ty0, tz0) is tx0. Entry plane is YZ.
            if (tym < tx0)
                current_octant |= 2;
            if (tzm < tx0)
                current_octant |= 1;
        }
        else
        {
            // max(tx0, ty0, tz0) is tz0. Entry plane is XY.
            if (txm < tz0)
                current_octant |= 4;
            if (tym < tz0)
                current_octant |= 2;
        }
    }
    else
    {
        if (ty0 > tz0)
        {
            // max(tx0, ty0, tz0) is ty0. Entry plane is XZ.
            if (txm < ty0)
                current_octant |= 4;
            if (tzm < ty0)
                current_octant |= 1;
        }
        else
        {
            // max(tx0, ty0, tz0) is tz0. Entry plane is XY.
            if (txm < tz0)
                current_octant |= 4;
            if (tym < tz0)
                current_octant |= 2;
        }
    }

    // This special state indicates algorithm termination.
    const unsigned int END = 8;

    while (true)
    {
        int idxMin=-1;
        switch (current_octant)
        {
        case 0:
            if (childVec[a])
            {
                idxMin =
                    childVec[a]->trace (origin, direction, tx0, ty0, tz0,
                            txm, tym, tzm, a, b,origin1,direction1,result);
                if (idxMin != -1)
                    return nearestTriangle (idxMin, origin1, direction1,result);
            }
            current_octant = choose_next (txm, tym, tzm, 4, 2, 1);
            break;

        case 1:
            if (childVec[1 ^ a])
            {
                idxMin =
                    childVec[1 ^ a]->trace (origin, direction, tx0, ty0,
                            tzm, txm, tym, tz1, a, b,origin1,direction1,result);
                if (idxMin != -1)
                    return nearestTriangle (idxMin, origin1, direction1,result);
            }
            current_octant = choose_next (txm, tym, tz1, 5, 3, END);
            break;

        case 2:
            if (childVec[2 ^ a])
            {
                idxMin =
                    childVec[2 ^ a]->trace (origin, direction, tx0, tym,
                            tz0, txm, ty1, tzm, a, b,origin1,direction1,result);
                if (idxMin != -1)
                    return nearestTriangle (idxMin, origin1, direction1,result);
            }
            current_octant = choose_next (txm, ty1, tzm, 6, END, 3);
            break;

        case 3:
            if (childVec[3 ^ a])
            {
                idxMin =
                    childVec[3 ^ a]->trace (origin, direction, tx0, tym,
                            tzm, txm, ty1, tz1, a, b,origin1,direction1,result);
                if (idxMin != -1)
                    return nearestTriangle (idxMin, origin1, direction1,result);
            }
            current_octant = choose_next (txm, ty1, tz1, 7, END, END);
            break;

        case 4:
            if (childVec[4 ^ a])
            {
                idxMin =
                    childVec[4 ^ a]->trace (origin, direction, txm, ty0,
                            tz0, tx1, tym, tzm, a, b,origin1,direction1,result);
                if (idxMin != -1)
                    return nearestTriangle (idxMin, origin1, direction1,result);
            }
            current_octant = choose_next (tx1, tym, tzm, END, 6, 5);
            break;

        case 5:
            if (childVec[5 ^ a])
            {
                idxMin =
                    childVec[5 ^ a]->trace (origin, direction, txm, ty0,
                            tzm, tx1, tym, tz1, a, b,origin1,direction1,result);
                if (idxMin != -1)
                    return nearestTriangle (idxMin, origin1, direction1,result);
            }
            current_octant = choose_next (tx1, tym, tz1, END, 7, END);
            break;

        case 6:
            if (childVec[6 ^ a])
            {
                idxMin =
                    childVec[6 ^ a]->trace (origin, direction, txm, tym,
                            tz0, tx1, ty1, tzm, a, b,origin1,direction1,result);
                if (idxMin != -1)
                    return nearestTriangle (idxMin, origin1, direction1,result);
            }
            current_octant = choose_next (tx1, ty1, tzm, END, END, 7);
            break;

        case 7:
            if (childVec[7 ^ a])
            {
                idxMin =
                    childVec[7 ^ a]->trace (origin, direction, txm, tym,
                            tzm, tx1, ty1, tz1, a, b,origin1,direction1,result);
                if (idxMin != -1)
                    return nearestTriangle (idxMin, origin1, direction1,result);
            }

        case END:
            if(idxMin==-1&&objects.size())
                return nearestTriangle (objects[0], origin1, direction1,result);
            return idxMin;
        }
    }
}


int TriangleOctree::trace (defaulttype::Vector3 origin, defaulttype::Vector3 direction,traceResult &result)
{
    unsigned int a = 0;
    unsigned int b = 0;
    //const double EPSILON = 1e-8;
    const double EPSILON = 0;
    defaulttype::Vector3 origin1=origin;
    defaulttype::Vector3 direction1=direction;

    if (direction[0] == 0.0)
    {
        direction[0]=EPSILON;
        b |= 4;
    }
    else if (direction[0] < 0.0)
    {
        origin[0] = -origin[0];
        direction[0] = -direction[0];
        a |= 4;
    }

    if (direction[1] == 0.0)
    {
        direction[1]=EPSILON;
        b |= 2;
    }
    else if (direction[1] < 0.0)
    {
        origin[1] = -origin[1];
        direction[1] = -direction[1];
        a |= 2;
    }

    if (direction[2] == 0.0)
    {
        direction[2]=EPSILON;
        b |= 1;
    }
    else if (direction[2] < 0.0)
    {
        origin[2] = -origin[2];
        direction[2] = -direction[2];
        a |= 1;
    }
    double tx0 = (-CUBE_SIZE - origin[0]) / direction[0];
    double tx1 = (CUBE_SIZE - origin[0]) / direction[0];
    double ty0 = (-CUBE_SIZE - origin[1]) / direction[1];
    double ty1 = (CUBE_SIZE - origin[1]) / direction[1];
    double tz0 = (-CUBE_SIZE - origin[2]) / direction[2];
    double tz1 = (CUBE_SIZE - origin[2]) / direction[2];
    if (bb_max3 (tx0, ty0, tz0) < bb_min3 (tx1, ty1, tz1))
        return trace (origin, direction, tx0, ty0, tz0, tx1, ty1, tz1, a, b,origin1,direction1,result);
    return -1;

}


void TriangleOctree::allTriangles (const defaulttype::Vector3 & /*origin*/,
        const defaulttype::Vector3 & /*direction*/,
        std::set<int>& results)
{
    for (unsigned int i = 0; i < objects.size (); i++)
        results.insert(objects[i]);
}

void TriangleOctree::allTriangles (const defaulttype::Vector3 & origin,
        const defaulttype::Vector3 & direction,
        helper::vector<traceResult>& results)
{
    static RayTriangleIntersection intersectionSolver;
    defaulttype::Vector3 P;
    const TriangleOctreeRoot::VecCoord& pos = *tm->octreePos;
    SReal t, u, v;
    for (unsigned int i = 0; i < objects.size (); i++)
    {
        TriangleOctreeRoot::Tri t2 = (*tm->octreeTriangles)[objects[i]];
        if (intersectionSolver.
            NewComputation (pos[t2[0]], pos[t2[1]], pos[t2[2]], origin, direction, t, u, v))
        {
            traceResult result;
            result.u = u;
            result.v = v;
            result.t = t;
            result.tid = objects[i];
            results.push_back(result);
        }
    }
}

template<class Res>
void TriangleOctree::traceAll (const defaulttype::Vector3 & origin,
        const defaulttype::Vector3 & direction, double tx0,
        double ty0, double tz0, double tx1,
        double ty1, double tz1, unsigned int a,
        unsigned int b,defaulttype::Vector3 &origin1,defaulttype::Vector3 &direction1, Res& results)
{
    if (tx1 < 0.0 || ty1 < 0.0 || tz1 < 0.0)
        return;

    if (is_leaf)
    {
        allTriangles (origin1, direction1, results);
        return;
    }

    const double &x0 = x;
    const double &y0 = y;
    const double &z0 = z;
    const double &x1 = x + size;
    const double &y1 = y + size;
    const double &z1 = z + size;
    const double INF = 1e9;

    double txm=0, tym=0, tzm=0;

    switch (b)
    {
    case 0:
        txm = 0.5 * (tx0 + tx1);
        tym = 0.5 * (ty0 + ty1);
        tzm = 0.5 * (tz0 + tz1);
        break;
    case 1:
        txm = 0.5 * (tx0 + tx1);
        tym = 0.5 * (ty0 + ty1);
        tzm = origin[2] < 0.5 * (z0 + z1) ? +INF : -INF;
        break;
    case 2:
        txm = 0.5 * (tx0 + tx1);
        tym = origin[1] < 0.5 * (y0 + y1) ? +INF : -INF;
        tzm = 0.5 * (tz0 + tz1);
        break;
    case 3:
        txm = 0.5 * (tx0 + tx1);
        tym = origin[1] < 0.5 * (y0 + y1) ? +INF : -INF;
        tzm = origin[2] < 0.5 * (z0 + z1) ? +INF : -INF;
        break;
    case 4:
        txm = origin[0] < 0.5 * (x0 + x1) ? +INF : -INF;
        tym = 0.5 * (ty0 + ty1);
        tzm = 0.5 * (tz0 + tz1);
        break;
    case 5:
        txm = (origin[0] < (0.5 * (x0 + x1)) )? +INF : -INF;
        tym = 0.5 * (ty0 + ty1);
        tzm = (origin[2] < (0.5 * (z0 + z1) ))? +INF : -INF;
        break;
    case 6:
        txm = origin[0] < 0.5 * (x0 + x1) ? +INF : -INF;
        tym = origin[1] < 0.5 * (y0 + y1) ? +INF : -INF;
        tzm = 0.5 * (tz0 + tz1);
        break;
    case 7:
        txm = origin[0] < 0.5 * (x0 + x1) ? +INF : -INF;
        tym = origin[1] < 0.5 * (y0 + y1) ? +INF : -INF;
        tzm = origin[2] < 0.5 * (z0 + z1) ? +INF : -INF;
        break;
    default:
        assert (!"Internal failure.");
    }
    unsigned int current_octant = 0;

    if (tx0 > ty0)
    {
        if (tx0 > tz0)
        {
            // max(tx0, ty0, tz0) is tx0. Entry plane is YZ.
            if (tym < tx0)
                current_octant |= 2;
            if (tzm < tx0)
                current_octant |= 1;
        }
        else
        {
            // max(tx0, ty0, tz0) is tz0. Entry plane is XY.
            if (txm < tz0)
                current_octant |= 4;
            if (tym < tz0)
                current_octant |= 2;
        }
    }
    else
    {
        if (ty0 > tz0)
        {
            // max(tx0, ty0, tz0) is ty0. Entry plane is XZ.
            if (txm < ty0)
                current_octant |= 4;
            if (tzm < ty0)
                current_octant |= 1;
        }
        else
        {
            // max(tx0, ty0, tz0) is tz0. Entry plane is XY.
            if (txm < tz0)
                current_octant |= 4;
            if (tym < tz0)
                current_octant |= 2;
        }
    }

    // This special state indicates algorithm termination.
    const unsigned int END = 8;

    while (true)
    {
        switch (current_octant)
        {
        case 0:
            if (childVec[a])
            {
                childVec[a]->traceAll (origin, direction, tx0, ty0, tz0,
                        txm, tym, tzm, a, b,origin1,direction1,results);
            }
            current_octant = choose_next (txm, tym, tzm, 4, 2, 1);
            break;

        case 1:
            if (childVec[1 ^ a])
            {
                childVec[1 ^ a]->traceAll (origin, direction, tx0, ty0,
                        tzm, txm, tym, tz1, a, b,origin1,direction1,results);
            }
            current_octant = choose_next (txm, tym, tz1, 5, 3, END);
            break;

        case 2:
            if (childVec[2 ^ a])
            {
                childVec[2 ^ a]->traceAll (origin, direction, tx0, tym,
                        tz0, txm, ty1, tzm, a, b,origin1,direction1,results);
            }
            current_octant = choose_next (txm, ty1, tzm, 6, END, 3);
            break;

        case 3:
            if (childVec[3 ^ a])
            {
                childVec[3 ^ a]->traceAll (origin, direction, tx0, tym,
                        tzm, txm, ty1, tz1, a, b,origin1,direction1,results);
            }
            current_octant = choose_next (txm, ty1, tz1, 7, END, END);
            break;

        case 4:
            if (childVec[4 ^ a])
            {
                childVec[4 ^ a]->traceAll (origin, direction, txm, ty0,
                        tz0, tx1, tym, tzm, a, b,origin1,direction1,results);
            }
            current_octant = choose_next (tx1, tym, tzm, END, 6, 5);
            break;

        case 5:
            if (childVec[5 ^ a])
            {
                childVec[5 ^ a]->traceAll (origin, direction, txm, ty0,
                        tzm, tx1, tym, tz1, a, b,origin1,direction1,results);
            }
            current_octant = choose_next (tx1, tym, tz1, END, 7, END);
            break;

        case 6:
            if (childVec[6 ^ a])
            {
                childVec[6 ^ a]->traceAll (origin, direction, txm, tym,
                        tz0, tx1, ty1, tzm, a, b,origin1,direction1,results);
            }
            current_octant = choose_next (tx1, ty1, tzm, END, END, 7);
            break;

        case 7:
            if (childVec[7 ^ a])
            {
                childVec[7 ^ a]->traceAll (origin, direction, txm, tym,
                        tzm, tx1, ty1, tz1, a, b,origin1,direction1,results);
            }

        case END:
            allTriangles (origin1, direction1, results);
            return;
        }
    }
}

void TriangleOctree::traceAll(defaulttype::Vector3 origin, defaulttype::Vector3 direction, helper::vector<traceResult>& results)
{
    traceAllStart(origin, direction, results);
}

void TriangleOctree::traceAllCandidates(defaulttype::Vector3 origin, defaulttype::Vector3 direction, std::set<int>& results)
{
    traceAllStart(origin, direction, results);
}

template<class Res>
void TriangleOctree::traceAllStart(defaulttype::Vector3 origin, defaulttype::Vector3 direction, Res& results)
{
    unsigned int a = 0;
    unsigned int b = 0;
    //const double EPSILON = 1e-8;
    const double EPSILON = 0;
    defaulttype::Vector3 origin1=origin;
    defaulttype::Vector3 direction1=direction;

    if (direction[0] == 0.0)
    {
        direction[0]=EPSILON;
        b |= 4;
    }
    else if (direction[0] < 0.0)
    {
        origin[0] = -origin[0];
        direction[0] = -direction[0];
        a |= 4;
    }

    if (direction[1] == 0.0)
    {
        direction[1]=EPSILON;
        b |= 2;
    }
    else if (direction[1] < 0.0)
    {
        origin[1] = -origin[1];
        direction[1] = -direction[1];
        a |= 2;
    }

    if (direction[2] == 0.0)
    {
        direction[2]=EPSILON;
        b |= 1;
    }
    else if (direction[2] < 0.0)
    {
        origin[2] = -origin[2];
        direction[2] = -direction[2];
        a |= 1;
    }
    double tx0 = (-CUBE_SIZE - origin[0]) / direction[0];
    double tx1 = (CUBE_SIZE - origin[0]) / direction[0];
    double ty0 = (-CUBE_SIZE - origin[1]) / direction[1];
    double ty1 = (CUBE_SIZE - origin[1]) / direction[1];
    double tz0 = (-CUBE_SIZE - origin[2]) / direction[2];
    double tz1 = (CUBE_SIZE - origin[2]) / direction[2];
    if (bb_max3 (tx0, ty0, tz0) < bb_min3 (tx1, ty1, tz1))
        traceAll (origin, direction, tx0, ty0, tz0, tx1, ty1, tz1, a, b,origin1,direction1,results);
}


void TriangleOctree::bbAllTriangles(const defaulttype::Vector3 & bbmin,
        const defaulttype::Vector3 & bbmax,
        std::set<int>& results)
{
    const TriangleOctreeRoot::VecCoord& pos = *tm->octreePos;
    const TriangleOctreeRoot::SeqTriangles& tri = *tm->octreeTriangles;
    for (unsigned int i = 0; i < objects.size (); i++)
    {
        int t = objects[i];
        defaulttype::Vector3 tmin = pos[tri[t][0]];
        defaulttype::Vector3 tmax = tmin;
        for (int j=1; j<3; ++j)
        {
            defaulttype::Vector3 p = pos[tri[t][j]];
            for (int c=0; c<3; ++c)
                    if (p[c] < tmin[c]) tmin[c] = p[c]; else if (p[c] > tmax[c]) tmax[c] = p[c];
        }
        if ( tmin[0] <= bbmax[0] && tmax[0] >= bbmin[0] &&
                tmin[1] <= bbmax[1] && tmax[1] >= bbmin[1] &&
                tmin[2] <= bbmax[2] && tmax[2] >= bbmin[2])
        {
            results.insert(t);
        }
    }
}

template<class Res>
void TriangleOctree::bbAll (const defaulttype::Vector3 & bbmin, const defaulttype::Vector3 & bbmax, Res& results)
{
    defaulttype::Vector3 c(x+size/2,y+size/2,z+size/2);
    bbAllTriangles (bbmin, bbmax, results);
    if (is_leaf)
    {
        return;
    }
    int dx0 = (bbmin[0] > c[0]) ? 1 : 0;    int dx1 = (bbmax[0] >= c[0]) ? 1 : 0;
    int dy0 = (bbmin[1] > c[1]) ? 1 : 0;    int dy1 = (bbmax[1] >= c[1]) ? 1 : 0;
    int dz0 = (bbmin[2] > c[2]) ? 1 : 0;    int dz1 = (bbmax[2] >= c[2]) ? 1 : 0;
    for (int dx = dx0; dx <= dx1; ++dx)
        for (int dy = dy0; dy <= dy1; ++dy)
            for (int dz = dz0; dz <= dz1; ++dz)
            {
                int i = dx * 4 + dy * 2 + dz;
                if (childVec[i])
                {
                    childVec[i]->bbAll (bbmin, bbmax, results);
                }
            }
}

void TriangleOctree::bboxAllCandidates(defaulttype::Vector3 bbmin, defaulttype::Vector3 bbmax, std::set<int>& results)
{
    bbAll(bbmin, bbmax, results);
}

TriangleOctreeRoot::TriangleOctreeRoot()
{
    octreeRoot = NULL;
    octreeTriangles = NULL;
    octreePos = NULL;
    cubeSize = CUBE_SIZE;
}

TriangleOctreeRoot::~TriangleOctreeRoot()
{
    if (octreeRoot)
        delete octreeRoot;
}

void TriangleOctreeRoot::buildOctree()
{
    if (!this->octreeTriangles || !this->octreePos) return;
    if (octreeRoot) delete octreeRoot;
    octreeRoot = new TriangleOctree(this);

    // for each triangle add it to the octree
    for (size_t i = 0; i < octreeTriangles->size(); i++)
    {
        fillOctree (i);
    }
}

int TriangleOctreeRoot::fillOctree (int tId, int /*d*/, defaulttype::Vector3 /*v*/)
{
    defaulttype::Vector3 center;
    defaulttype::Vector3 corner (-cubeSize, -cubeSize, -cubeSize);

    double bb[6];
    double bbsize;
    calcTriangleAABB(tId, bb, bbsize);

    // computes the depth of the bounding box in a octree
    int d1 = (int)((log10( (double) CUBE_SIZE * 2/ bbsize ) / log10( (double)2) ));
    // computes the size of the octree box that can store the bounding box
    int divs = (1 << (d1));
    double inc = (double) (2 * CUBE_SIZE) / divs;
    if (bb[0] >= -CUBE_SIZE && bb[2] >= -CUBE_SIZE && bb[4] >= -CUBE_SIZE
        && bb[1] <= CUBE_SIZE && bb[3] <= CUBE_SIZE && bb[5] <= CUBE_SIZE)
        for (double x1 =
                (((int)((bb[0] + CUBE_SIZE) / inc)) * inc - CUBE_SIZE);
                x1 <= bb[1]; x1 += inc)
        {

            for (double y1 =
                    ((int)((bb[2] + CUBE_SIZE) / inc)) * inc - CUBE_SIZE;
                    y1 <= bb[3]; y1 += inc)
            {


                for (double z1 =
                        ((int)((bb[4] + CUBE_SIZE) / inc)) * inc - CUBE_SIZE;
                        z1 <= bb[5]; z1 += inc)
                {
                    octreeRoot->insert (x1, y1, z1, inc, tId);

                }
            }
        }
    return 0;

}

void TriangleOctreeRoot::calcTriangleAABB(int tId, double* bb, double& size)
{
    Tri t = (*octreeTriangles)[tId];
    Coord p1 = (*octreePos)[t[0]];
    Coord p2 = (*octreePos)[t[1]];
    Coord p3 = (*octreePos)[t[2]];
    for (int i = 0; i < 3; i++)
    {
        bb[i * 2] = bb_min3 (p1[i], p2[i], p3[i]);
        bb[(i * 2) + 1] = bb_max3 (p1[i], p2[i], p3[i]);
    }
    size = bb_max3 (fabs (bb[1] - bb[0]), fabs (bb[3] - bb[2]),
            fabs (bb[5] - bb[4]));
}

} // namespace collision

} // namespace component

} // namespace sofa
