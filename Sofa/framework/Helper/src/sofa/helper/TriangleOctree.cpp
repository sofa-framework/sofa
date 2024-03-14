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
#include <sofa/helper/TriangleOctree.h>

#include <sofa/helper/visual/DrawTool.h>

namespace sofa::helper
{

TriangleOctree::~TriangleOctree()
{
    for(int i=0; i<8; i++)
    {
        if(childVec[i])
        {
            delete childVec[i];
            childVec[i]=nullptr;
        }
    }
}

void TriangleOctree::draw (sofa::helper::visual::DrawTool* drawTool)
{
    type::Vec3 center;
    if ( objects.size ())
    {
        center =
            (type::Vec3 (x, y, z) + type::Vec3 (size / 2, size / 2, size / 2));
        drawTool->pushMatrix();
        drawTool->translate((float)center[0], (float)center[1], (float)center[2]);
        drawTool->setPolygonMode(0, false);
        drawTool->drawCube(size, sofa::type::RGBAColor(0.5f, 0.5f, 0.5f, 1.0f));
        drawTool->popMatrix();

        drawTool->setPolygonMode(0, true);
    }
    for (int i = 0; i < 8; i++)
    {
        if (childVec[i])
            childVec[i]->draw(drawTool);
    }
}

void TriangleOctree::insert (SReal _x, SReal _y, SReal _z,
        SReal inc, int t)
{
    if (inc >= size)
    {
        objects.push_back (t);
    }
    else
    {
        const SReal size2 = size / 2;
        const int dx = (_x >= (x + size2)) ? 1 : 0;
        const int dy = (_y >= (y + size2)) ? 1 : 0;
        const int dz = (_z >= (z + size2)) ? 1 : 0;

        const int i = dx * 4 + dy * 2 + dz;
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
unsigned int choose_next (SReal x, SReal y, SReal z,
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
        const type::Vec3 & origin,
        const type::Vec3 & direction, traceResult &result)
{
    const TriangleOctreeRoot::VecCoord& pos = *tm->octreePos;
    //Triangle t1 (tm, minIndex);
    TriangleOctreeRoot::Tri t1 = (*tm->octreeTriangles)[minIndex];
    SReal minDist;
    SReal t, u, v;
    if (sofa::geometry::Triangle::rayIntersection(pos[t1[0]], pos[t1[1]], pos[t1[2]], origin, direction, t, u, v))
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
        if (!sofa::geometry::Triangle::rayIntersection(pos[t2[0]], pos[t2[1]], pos[t2[2]], origin, direction, t, u, v))
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
        SReal cosAng=dot(direction,t3.n());

        for(unsigned int i=0; i<tm->pTri[pointR].size(); i++)
        {
            Triangle t2(tm,tm->pTri[pointR][i]);
            SReal cosAng2=dot(direction,t2.n());
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

int TriangleOctree::trace (const type::Vec3 & origin,
        const type::Vec3 & direction, SReal tx0,
        SReal ty0, SReal tz0, SReal tx1,
        SReal ty1, SReal tz1, unsigned int a,
        unsigned int b,type::Vec3 &origin1,type::Vec3 &direction1,traceResult &result)
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

    const SReal &x0 = x;
    const SReal &y0 = y;
    const SReal &z0 = z;
    const SReal &x1 = x + size;
    const SReal &y1 = y + size;
    const SReal &z1 = z + size;
    const SReal INF = 1e9;

    SReal txm=0, tym=0, tzm=0;

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
            [[fallthrough]];
        case END:
            if(idxMin==-1&&objects.size())
                return nearestTriangle (objects[0], origin1, direction1,result);
            return idxMin;
        }
    }
}


int TriangleOctree::trace (type::Vec3 origin, type::Vec3 direction,traceResult &result)
{
    unsigned int a = 0;
    unsigned int b = 0;
    //const SReal EPSILON = 1e-8;
    const SReal EPSILON = 0;
    type::Vec3 origin1=origin;
    type::Vec3 direction1=direction;

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
    SReal tx0 = (-CUBE_SIZE - origin[0]) / direction[0];
    SReal tx1 = (CUBE_SIZE - origin[0]) / direction[0];
    SReal ty0 = (-CUBE_SIZE - origin[1]) / direction[1];
    SReal ty1 = (CUBE_SIZE - origin[1]) / direction[1];
    SReal tz0 = (-CUBE_SIZE - origin[2]) / direction[2];
    SReal tz1 = (CUBE_SIZE - origin[2]) / direction[2];
    if (std::max({ tx0, ty0, tz0 }) < std::min({ tx1, ty1, tz1 }))
        return trace (origin, direction, tx0, ty0, tz0, tx1, ty1, tz1, a, b,origin1,direction1,result);
    return -1;

}


void TriangleOctree::allTriangles (const type::Vec3 & /*origin*/,
        const type::Vec3 & /*direction*/,
        std::set<int>& results)
{
    for (unsigned int i = 0; i < objects.size (); i++)
        results.insert(objects[i]);
}

void TriangleOctree::allTriangles (const type::Vec3 & origin,
        const type::Vec3 & direction,
        type::vector<traceResult>& results)
{
    const TriangleOctreeRoot::VecCoord& pos = *tm->octreePos;
    SReal t, u, v;
    for (unsigned int i = 0; i < objects.size (); i++)
    {
        TriangleOctreeRoot::Tri t2 = (*tm->octreeTriangles)[objects[i]];
        if (sofa::geometry::Triangle::rayIntersection(pos[t2[0]], pos[t2[1]], pos[t2[2]], origin, direction, t, u, v))
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
void TriangleOctree::traceAll (const type::Vec3 & origin,
        const type::Vec3 & direction, SReal tx0,
        SReal ty0, SReal tz0, SReal tx1,
        SReal ty1, SReal tz1, unsigned int a,
        unsigned int b,type::Vec3 &origin1,type::Vec3 &direction1, Res& results)
{
    if (tx1 < 0.0 || ty1 < 0.0 || tz1 < 0.0)
        return;

    if (is_leaf)
    {
        allTriangles (origin1, direction1, results);
        return;
    }

    const SReal &x0 = x;
    const SReal &y0 = y;
    const SReal &z0 = z;
    const SReal &x1 = x + size;
    const SReal &y1 = y + size;
    const SReal &z1 = z + size;
    const SReal INF = 1e9;

    SReal txm=0, tym=0, tzm=0;

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
            [[fallthrough]];
        case END:
            allTriangles (origin1, direction1, results);
            return;
        }
    }
}

void TriangleOctree::traceAll(type::Vec3 origin, type::Vec3 direction, type::vector<traceResult>& results)
{
    traceAllStart(origin, direction, results);
}

void TriangleOctree::traceAllCandidates(type::Vec3 origin, type::Vec3 direction, std::set<int>& results)
{
    traceAllStart(origin, direction, results);
}

template<class Res>
void TriangleOctree::traceAllStart(type::Vec3 origin, type::Vec3 direction, Res& results)
{
    unsigned int a = 0;
    unsigned int b = 0;
    //const SReal EPSILON = 1e-8;
    const SReal EPSILON = 0;
    type::Vec3 origin1=origin;
    type::Vec3 direction1=direction;

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
    SReal tx0 = (-CUBE_SIZE - origin[0]) / direction[0];
    SReal tx1 = (CUBE_SIZE - origin[0]) / direction[0];
    SReal ty0 = (-CUBE_SIZE - origin[1]) / direction[1];
    SReal ty1 = (CUBE_SIZE - origin[1]) / direction[1];
    SReal tz0 = (-CUBE_SIZE - origin[2]) / direction[2];
    SReal tz1 = (CUBE_SIZE - origin[2]) / direction[2];
    if (std::max({ tx0, ty0, tz0 }) < std::min({ tx1, ty1, tz1 }))
        traceAll (origin, direction, tx0, ty0, tz0, tx1, ty1, tz1, a, b,origin1,direction1,results);
}


void TriangleOctree::bbAllTriangles(const type::Vec3 & bbmin,
        const type::Vec3 & bbmax,
        std::set<int>& results)
{
    const TriangleOctreeRoot::VecCoord& pos = *tm->octreePos;
    const TriangleOctreeRoot::SeqTriangles& tri = *tm->octreeTriangles;
    for (unsigned int i = 0; i < objects.size (); i++)
    {
        int t = objects[i];
        type::Vec3 tmin = pos[tri[t][0]];
        type::Vec3 tmax = tmin;
        for (int j=1; j<3; ++j)
        {
            type::Vec3 p = pos[tri[t][j]];
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
void TriangleOctree::bbAll (const type::Vec3 & bbmin, const type::Vec3 & bbmax, Res& results)
{
    type::Vec3 c(x+size/2,y+size/2,z+size/2);
    bbAllTriangles (bbmin, bbmax, results);
    if (is_leaf)
    {
        return;
    }
    const int dx0 = (bbmin[0] > c[0]) ? 1 : 0;
    const int dx1 = (bbmax[0] >= c[0]) ? 1 : 0;
    const int dy0 = (bbmin[1] > c[1]) ? 1 : 0;
    const int dy1 = (bbmax[1] >= c[1]) ? 1 : 0;
    const int dz0 = (bbmin[2] > c[2]) ? 1 : 0;
    const int dz1 = (bbmax[2] >= c[2]) ? 1 : 0;
    for (int dx = dx0; dx <= dx1; ++dx)
        for (int dy = dy0; dy <= dy1; ++dy)
            for (int dz = dz0; dz <= dz1; ++dz)
            {
                const int i = dx * 4 + dy * 2 + dz;
                if (childVec[i])
                {
                    childVec[i]->bbAll (bbmin, bbmax, results);
                }
            }
}

void TriangleOctree::bboxAllCandidates(type::Vec3 bbmin, type::Vec3 bbmax, std::set<int>& results)
{
    bbAll(bbmin, bbmax, results);
}

TriangleOctreeRoot::TriangleOctreeRoot()
{
    octreeRoot = nullptr;
    octreeTriangles = nullptr;
    octreePos = nullptr;
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

int TriangleOctreeRoot::fillOctree (int tId, int /*d*/, type::Vec3 /*v*/)
{
    SReal bb[6];
    SReal bbsize;
    calcTriangleAABB(tId, bb, bbsize);

    // computes the depth of the bounding box in a octree
    const int d1 = (int)((log10( (SReal) CUBE_SIZE * 2/ bbsize ) / log10( (SReal)2) ));
    // computes the size of the octree box that can store the bounding box
    const int divs = (1 << (d1));
    const SReal inc = (SReal) (2 * CUBE_SIZE) / divs;
    if (bb[0] >= -CUBE_SIZE && bb[2] >= -CUBE_SIZE && bb[4] >= -CUBE_SIZE
        && bb[1] <= CUBE_SIZE && bb[3] <= CUBE_SIZE && bb[5] <= CUBE_SIZE)
        for (SReal x1 =
                (((int)((bb[0] + CUBE_SIZE) / inc)) * inc - CUBE_SIZE);
                x1 <= bb[1]; x1 += inc)
        {

            for (SReal y1 =
                    ((int)((bb[2] + CUBE_SIZE) / inc)) * inc - CUBE_SIZE;
                    y1 <= bb[3]; y1 += inc)
            {


                for (SReal z1 =
                        ((int)((bb[4] + CUBE_SIZE) / inc)) * inc - CUBE_SIZE;
                        z1 <= bb[5]; z1 += inc)
                {
                    octreeRoot->insert (x1, y1, z1, inc, tId);

                }
            }
        }
    return 0;

}

void TriangleOctreeRoot::calcTriangleAABB(int tId, SReal* bb, SReal& size)
{
    Tri t = (*octreeTriangles)[tId];
    Coord p1 = (*octreePos)[t[0]];
    Coord p2 = (*octreePos)[t[1]];
    Coord p3 = (*octreePos)[t[2]];
    for (int i = 0; i < 3; i++)
    {
        bb[i * 2] = std::min({ p1[i], p2[i], p3[i] });
        bb[(i * 2) + 1] = std::max({ p1[i], p2[i], p3[i] });
    }
    size = std::max({ fabs(bb[1] - bb[0]), fabs(bb[3] - bb[2]),
            fabs(bb[5] - bb[4]) });
}

} // namespace sofa::helper
