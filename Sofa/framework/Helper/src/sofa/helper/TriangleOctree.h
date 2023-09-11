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
#pragma once

#include <sofa/helper/config.h>

#include <sofa/type/Vec.h>
#include <sofa/type/vector.h>
#include <sofa/topology/Triangle.h>
#include <set>

// fwd declaration for depreciation
namespace sofa::core::visual
{
    class VisualParams;
}

namespace sofa::helper
{

namespace visual
{
    class DrawTool;
}

class TriangleOctree;

class SOFA_HELPER_API TriangleOctreeRoot
{
public:
    /*THIS STATIC CUBE SIZE MUST BE CHANGED, it represents the size of the occtree cube*/
    static constexpr int CUBE_SIZE = 800;

    typedef sofa::topology::Triangle Tri;
    typedef sofa::type::vector<sofa::topology::Triangle> SeqTriangles;
    typedef sofa::type::Vec3 Coord;
    typedef sofa::type::vector<sofa::type::Vec3> VecCoord;
    /// the triangles used as input to construct the octree
    const SeqTriangles* octreeTriangles;
    /// the positions of vertices used as input to construct the octree
    const VecCoord* octreePos;
    /// the first node of the octree
    TriangleOctree* octreeRoot;
    /// the size of the octree cube
    int cubeSize;

    TriangleOctreeRoot();
    ~TriangleOctreeRoot();

    void buildOctree();
    void buildOctree(const SeqTriangles* triangles, const VecCoord* pos)
    {
        this->octreeTriangles = triangles;
        this->octreePos = pos;
        buildOctree();
    }

protected:
    /// used to add a triangle  to the octree
    int fillOctree(int t, int d = 0, type::Vec3 v = { 0_sreal, 0_sreal, 0_sreal });
    /// used to compute the Bounding Box for each triangle
    void calcTriangleAABB(int t, SReal* bb, SReal& size);
};

class SOFA_HELPER_API TriangleOctree
{
public:
    static constexpr int CUBE_SIZE = TriangleOctreeRoot::CUBE_SIZE;

    class traceResult
    {
    public:
        traceResult():tid(-1),t(0),u(0),v(0) {}
        int tid;
        SReal t,u,v;
        bool operator == (const traceResult& r) const { return tid == r.tid && t == r.t && u == r.u && v == r.v; }
        bool operator != (const traceResult& r) const { return tid != r.tid || t != r.t || u != r.u || v != r.v; }
        bool operator < (const traceResult& r) const { return t < r.t; }
        bool operator <= (const traceResult& r) const { return t <= r.t; }
        bool operator > (const traceResult& r) const { return t > r.t; }
        bool operator >= (const traceResult& r) const { return t >= r.t; }
    };

    SReal x, y, z;
    bool visited;

    SReal size;
    bool val;
    bool is_leaf;
    bool internal;
    TriangleOctreeRoot *tm;
    type::vector< int >objects;
    TriangleOctree *childVec[8];

    ~TriangleOctree ();
    /*the default cube has xmin=-CUBE_SIZE xmax=CUBE_SIZE, ymin=-CUBE_SIZE, ymax=CUBE_SIZE, zmin=-CUBE_SIZE,zmax=CUBE_SIZE*/
    TriangleOctree (TriangleOctreeRoot * _tm, SReal _x = (SReal)-CUBE_SIZE, SReal _y = (SReal)-CUBE_SIZE, SReal _z =(SReal) -CUBE_SIZE, SReal _size = 2 * CUBE_SIZE)
        : x (_x), y (_y), z (_z), size (_size)
        , tm(_tm)
    {
        is_leaf = true;
        internal = false;
        for (int i = 0; i < 8; i++)
            childVec[i] = nullptr;
    }

    void draw(sofa::helper::visual::DrawTool* drawtool);

    /// Find the nearest triangle intersecting the given ray, or -1 of not found
    int trace (type::Vec3 origin, type::Vec3 direction, traceResult &result);

    /// Find all triangles intersecting the given ray
    void traceAll (type::Vec3 origin, type::Vec3 direction, type::vector<traceResult>& results);

    /// Find all triangles intersecting the given ray
    void traceAllCandidates(type::Vec3 origin, type::Vec3 direction, std::set<int>& results);

    /// Find all triangles intersecting the given ray
    void bboxAllCandidates(type::Vec3 bbmin, type::Vec3 bbmax, std::set<int>& results);

    friend class TriangleOctreeRoot;

protected:
    int trace (const type::Vec3 & origin, const type::Vec3 & direction,
            SReal tx0, SReal ty0, SReal tz0, SReal tx1, SReal ty1,
            SReal tz1, unsigned int a, unsigned int b,type::Vec3 &origin1,type::Vec3 &direction1, traceResult &result);

    template<class Res>
    void traceAllStart (type::Vec3 origin, type::Vec3 direction, Res& results);

    template<class Res>
    void traceAll (const type::Vec3 & origin, const type::Vec3 & direction,
            SReal tx0, SReal ty0, SReal tz0, SReal tx1, SReal ty1,
            SReal tz1, unsigned int a, unsigned int b,type::Vec3 &origin1,type::Vec3 &direction1, Res& results);

    template<class Res>
    void bbAll (const type::Vec3 & bbmin, const type::Vec3 & bbmax, Res& results);

    int nearestTriangle (int minIndex, const type::Vec3 & origin,
            const type::Vec3 & direction,traceResult &result);

    void allTriangles (const type::Vec3 & origin,
            const type::Vec3 & direction, type::vector<traceResult>& results);

    void allTriangles (const type::Vec3 & origin,
            const type::Vec3 & direction, std::set<int>& results);

    void bbAllTriangles (const type::Vec3 & bbmin,
            const type::Vec3 & bbmax, std::set<int>& results);

    void insert (SReal _x, SReal _y, SReal _z, SReal _inc, int t);

};

} // namespace sofa::helper
