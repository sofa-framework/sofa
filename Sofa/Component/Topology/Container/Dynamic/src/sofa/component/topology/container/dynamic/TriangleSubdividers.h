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
#include <sofa/component/topology/container/dynamic/config.h>
#include <sofa/component/topology/container/dynamic/PointModifiers.h>

#include <sofa/type/vector.h>
#include <sofa/core/topology/BaseTopology.h>

namespace sofa::component::topology::container::dynamic
{

using Triangle = core::topology::BaseMeshTopology::Triangle;
using Edge = sofa::core::topology::BaseMeshTopology::Edge;
using TriangleID = core::topology::BaseMeshTopology::TriangleID;
using EdgeID = core::topology::BaseMeshTopology::EdgeID;
using PointID = core::topology::BaseMeshTopology::PointID;

/**
* This class store all the info to create a new triangle in the mesh taking into account estimated unique id
* triangle structure with vertex indices
* This structure also store all the ancestors and coefficient to efficently add this triangle with the area ratio into the current mesh.
*/
class TriangleToAdd
{
public:
    TriangleToAdd(TriangleID uniqueID, sofa::core::topology::BaseMeshTopology::Triangle _triangle,
        const type::vector<TriangleID>& ancestors, const type::vector<SReal>& coefs)
        : m_uniqueID(uniqueID)
        , m_triangle(_triangle)
        , m_ancestors(ancestors)
        , m_coefs(coefs)
    {}

    virtual ~TriangleToAdd() {}

    TriangleID m_uniqueID; ///< unique new id of the future triangle
    Triangle m_triangle; ///< Triangle topological structure
    type::vector<TriangleID> m_ancestors; ///< Triangle indices ancestors of this new Triangle
    type::vector<SReal> m_coefs; ///< Coefficient to apply with the ancestors info to compute new triangle area

    sofa::type::fixed_array<sofa::type::Vec3, 3> m_triCoords;
    bool isUp = false;
};


/**
* This class store all info to be able to split a Triangle into a given configuration.
* Depending on the configuration, 2 or 3 new triangle will be computed as well as new points.
* All info to create the new elements are computed and will be generated using a TriangleSetModifier component.
*/
class TriangleSubdivider
{
public:
    TriangleSubdivider(TriangleID triangleId, const Triangle& triangle);

    virtual bool subdivide(const sofa::type::fixed_array<sofa::type::Vec3, 3>& triCoords)
    {
        SOFA_UNUSED(triCoords);
        return false;
    }

protected:
    PointID localVertexId(PointID vertexIndex);

    sofa::type::Vec3 computePointCoordinates(const PointToAdd* PTA, const sofa::type::fixed_array<sofa::type::Vec3, 3>& triCoords);

protected:
    TriangleID m_triangleId; ///< Index of this triangle in the Topology Container
    sofa::core::topology::BaseMeshTopology::Triangle m_triangle; ///< Triangle fixed array of the 3 vertex indices

    type::vector<PointToAdd*> m_points; ///< Vector of new point to be added while subdividing this Triangle
    type::vector<TriangleToAdd*> m_trianglesToAdd; ///< Vector of triangle to be added while subdividing this Triangle
};


/**
* Specialisation of TriangleSubdivider to compute configuration where 1 Node is added in inside of the Triangle
* In this case 3 new Triangles and 1 Point will be created.
*/
class TriangleSubdivider_1Node : public TriangleSubdivider
{
public:
    TriangleSubdivider_1Node(TriangleID triangleId, const Triangle& triangle) 
        : TriangleSubdivider(triangleId, triangle)
    {}

    bool subdivide(const sofa::type::fixed_array<sofa::type::Vec3, 3>& triCoords) override;
};


/**
* Specialisation of TriangleSubdivider to compute configuration where 1 Node is added on one edge of the Triangle
* In this case 2 new Triangles and 1 Point will be created.
*/
class TriangleSubdivider_1Edge : public TriangleSubdivider
{
public:
    TriangleSubdivider_1Edge(TriangleID triangleId, const Triangle& triangle)
        : TriangleSubdivider(triangleId, triangle)
    {}

    bool subdivide(const sofa::type::fixed_array<sofa::type::Vec3, 3>& triCoords) override;
};


/**
* Specialisation of TriangleSubdivider to compute configuration where 2 Nodes are added on 2 different edges of the Triangle
* In this case 3 new Triangles and 2 Points will be created.
*/
class TriangleSubdivider_2Edge : public TriangleSubdivider
{
public:
    TriangleSubdivider_2Edge(TriangleID triangleId, const Triangle& triangle)
        : TriangleSubdivider(triangleId, triangle)
    {}

    bool subdivide(const sofa::type::fixed_array<sofa::type::Vec3, 3>& triCoords) override;
};


/**
* Specialisation of TriangleSubdivider to compute configuration where 3 Nodes are added on 3 different edges of the Triangle
* In this case 4 new Triangles and 3 Points will be created.
*/
class TriangleSubdivider_3Edge : public TriangleSubdivider
{
public:
    TriangleSubdivider_3Edge(TriangleID triangleId, const Triangle& triangle)
        : TriangleSubdivider(triangleId, triangle)
    {}

    bool subdivide(const sofa::type::fixed_array<sofa::type::Vec3, 3>& triCoords) override;
};


/**
* Specialisation of TriangleSubdivider to compute configuration where 2 Nodes are added, 1 on edge and 1 inside of the Triangle
* In this case 4 new Triangles and 2 Points will be created.
*/
class TriangleSubdivider_2Node : public TriangleSubdivider
{
public:
    TriangleSubdivider_2Node(TriangleID triangleId, const Triangle& triangle)
        : TriangleSubdivider(triangleId, triangle)
    {}

    bool subdivide(const sofa::type::fixed_array<sofa::type::Vec3, 3>& triCoords) override;
};



} //namespace sofa::component::topology::container::dynamic
