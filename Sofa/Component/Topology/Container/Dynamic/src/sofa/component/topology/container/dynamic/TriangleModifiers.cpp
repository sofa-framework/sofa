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
#include <sofa/component/topology/container/dynamic/TriangleModifiers.h>

namespace sofa::component::topology::container::dynamic
{


TriangleSubdivider::TriangleSubdivider(TriangleID triangleId, const sofa::core::topology::BaseMeshTopology::Triangle& triangle)
    : m_triangleId(triangleId)
    , m_triangle(triangle)
{

}


PointID TriangleSubdivider::localVertexId(PointID vertexIndex)
{
    for (unsigned int i = 0; i < 3; ++i)
    {
        if (m_triangle[i] == vertexIndex)
            return i;
    }

    return InvalidID;
}


sofa::type::Vec3 TriangleSubdivider::computePointCoordinates(const PointToAdd* PTA, const sofa::type::fixed_array<sofa::type::Vec3, 3>& triCoords)
{
    sofa::type::Vec3 pG = sofa::type::Vec3(0, 0, 0);
    for (unsigned int i = 0; i < PTA->m_ancestors.size(); ++i)
    {
        PointID localId = localVertexId(PTA->m_ancestors[i]);
        pG = pG + triCoords[localId] * PTA->m_coefs[i];
    }

    return pG;
}


bool TriangleSubdivider_1Node::subdivide(const sofa::type::fixed_array<sofa::type::Vec3, 3>& triCoords)
{
    if (m_points.size() != 1)
    {
        msg_error("TriangleSubdivider_1Node") << "More than 1 point to add to subdivide triangle id: " << m_triangleId;
        return false;
    }

    const PointToAdd* PTA = m_points[0];

    // compute point coordinates using barycoefs
    sofa::type::Vec3 pG = computePointCoordinates(PTA, triCoords);

    type::vector<TriangleID> ancestors;
    ancestors.push_back(m_triangleId);

    SReal areaFull = geometry::Triangle::area(triCoords[0], triCoords[1], triCoords[2]);

    for (unsigned int i = 0; i < 3; i++)
    {
        Triangle newTri = Triangle(m_triangle[i], m_triangle[(i + 1) % 3], PTA->m_idPoint);

        type::vector<SReal> coefs;
        SReal areaNewTri = geometry::Triangle::area(triCoords[i], triCoords[(i + 1) % 3], pG);
        coefs.push_back(areaNewTri / areaFull);

        auto TTA = new TriangleToAdd(1000000 * m_triangleId + i, newTri, ancestors, coefs);
        TTA->m_triCoords = { triCoords[i] , triCoords[(i + 1) % 3] , pG };
        m_trianglesToAdd.push_back(TTA);
    }

    return true;
}


bool TriangleSubdivider_1Edge::subdivide(const sofa::type::fixed_array<sofa::type::Vec3, 3>& triCoords)
{
    if (m_points.size() != 1)
    {
        msg_error("TriangleSubdivider_1Node") << "More than 1 point to add to subdivide triangle id: " << m_triangleId;
        return false;
    }

    const PointToAdd* PTA = m_points[0];
    EdgeID localEdgeId = InvalidID;
    for (unsigned int i = 0; i < 3; i++)
    {
        if (m_triangle[i] != PTA->m_ancestors[0] && m_triangle[i] != PTA->m_ancestors[1])
        {
            localEdgeId = i;
            break;
        }
    }

    if (localEdgeId == InvalidID)
    {
        msg_error("TriangleSubdivider_1Node") << "Point to add is not part of one of the edge of this triangle: " << m_triangleId;
        return false;
    }

    Triangle newTri0 = Triangle(m_triangle[(localEdgeId + 1) % 3], PTA->m_idPoint, m_triangle[localEdgeId]);
    Triangle newTri1 = Triangle(PTA->m_idPoint, m_triangle[(localEdgeId + 2) % 3], m_triangle[localEdgeId]);

    sofa::type::Vec3 pG = computePointCoordinates(PTA, triCoords);
    SReal areaFull = geometry::Triangle::area(triCoords[0], triCoords[1], triCoords[2]);
    SReal areaTri0 = geometry::Triangle::area(triCoords[(localEdgeId + 1) % 3], pG, triCoords[localEdgeId]);

    type::vector<TriangleID> ancestors;
    ancestors.push_back(m_triangleId);
    type::vector<SReal> coefsTri0, coefsTri1;
    coefsTri0.push_back(areaTri0 / areaFull);
    coefsTri1.push_back(1 - coefsTri0[0]);

    auto TTA0 = new TriangleToAdd(1000000 * m_triangleId, newTri0, ancestors, coefsTri0);
    auto TTA1 = new TriangleToAdd(1000000 * m_triangleId + 1, newTri1, ancestors, coefsTri1);
    TTA0->m_triCoords = { triCoords[(localEdgeId + 1) % 3] , pG , triCoords[localEdgeId] };
    TTA1->m_triCoords = { pG , triCoords[(localEdgeId + 2) % 3] , triCoords[localEdgeId] };

    m_trianglesToAdd.push_back(TTA0);
    m_trianglesToAdd.push_back(TTA1);

    return true;
}


bool TriangleSubdivider_2Edge::subdivide(const sofa::type::fixed_array<sofa::type::Vec3, 3>& triCoords)
{
    if (m_points.size() != 2)
    {
        msg_error("TriangleSubdivider_2Node") << "There are no 2 points to add to subdivide triangle id: " << m_triangleId;
        return false;
    }

    const PointToAdd* PTA0 = m_points[0];
    const PointToAdd* PTA1 = m_points[1];

    // get commun point
    const Edge theEdge0 = Edge(PTA0->m_ancestors[0], PTA0->m_ancestors[1]);
    const Edge theEdge1 = Edge(PTA1->m_ancestors[0], PTA1->m_ancestors[1]);

    PointID communLocalID = InvalidID;
    PointID communID = InvalidID;
    Edge theLocalEdge0;
    Edge theLocalEdge1;
    for (unsigned int i = 0; i < 3; i++)
    {
        auto pointId = m_triangle[i];
        unsigned int cpt = 0;

        if (pointId == theEdge0[0])
        {
            theLocalEdge0[0] = i;
            cpt++;
        }

        if (pointId == theEdge0[1])
        {
            theLocalEdge0[1] = i;
            cpt++;
        }


        if (pointId == theEdge1[0])
        {
            theLocalEdge1[0] = i;
            cpt++;
        }

        if (pointId == theEdge1[1])
        {
            theLocalEdge1[1] = i;
            cpt++;
        }

        if (cpt == 2)
        {
            communLocalID = i;
            communID = pointId;
        }
    }

    PointID nextID = m_triangle[(communLocalID + 1) % 3];
    PointID otherEdgeId0 = (theEdge0[0] == communID) ? theEdge0[1] : theEdge0[0];
    PointID otherEdgeId1 = (theEdge1[0] == communID) ? theEdge1[1] : theEdge1[0];

    bool directOriented = (otherEdgeId1 == nextID) ? true : false;

    sofa::type::fixed_array<PointID, 4> baseQuadriID;
    sofa::type::fixed_array<sofa::type::Vec3, 4> quadPoints;

    sofa::type::Vec3 p0 = triCoords[theLocalEdge0[0]] * PTA0->m_coefs[0] + triCoords[theLocalEdge0[1]] * PTA0->m_coefs[1];
    sofa::type::Vec3 p1 = triCoords[theLocalEdge1[0]] * PTA1->m_coefs[0] + triCoords[theLocalEdge1[1]] * PTA1->m_coefs[1];

    sofa::type::fixed_array<Triangle, 3> newTris;
    sofa::type::fixed_array<SReal, 3> newAreas;
    if (directOriented)
    {
        newTris[0] = Triangle(communID, PTA1->m_idPoint, PTA0->m_idPoint); // top triangle
        baseQuadriID = { PTA1->m_idPoint, nextID, m_triangle[(communLocalID + 2) % 3], PTA0->m_idPoint };
        quadPoints = { p1 , triCoords[(communLocalID + 1) % 3], triCoords[(communLocalID + 2) % 3], p0 };
    }
    else
    {
        newTris[0] = Triangle(communID, PTA0->m_idPoint, PTA1->m_idPoint); // top triangle
        baseQuadriID = { PTA0->m_idPoint, nextID, m_triangle[(communLocalID + 2) % 3], PTA1->m_idPoint };
        quadPoints = { p0 , triCoords[(communLocalID + 1) % 3], triCoords[(communLocalID + 2) % 3], p1 };
    }

    // compute diagonals
    auto diag0 = (quadPoints[0] - quadPoints[2]).norm2();
    auto diag1 = (quadPoints[1] - quadPoints[3]).norm2();

    SReal areaFull = geometry::Triangle::area(triCoords[0], triCoords[1], triCoords[2]);
    newAreas[0] = geometry::Triangle::area(triCoords[communLocalID], p1, p0);

    sofa::type::fixed_array<sofa::type::fixed_array<sofa::type::Vec3, 3>, 3> allTriCoords;
    allTriCoords[0] = { triCoords[communLocalID], p1, p0 };

    if (diag0 < diag1)
    {
        newTris[1] = Triangle(baseQuadriID[0], baseQuadriID[1], baseQuadriID[2]);
        newTris[2] = Triangle(baseQuadriID[2], baseQuadriID[3], baseQuadriID[0]);
        newAreas[1] = geometry::Triangle::area(quadPoints[0], quadPoints[1], quadPoints[2]);
        newAreas[2] = geometry::Triangle::area(quadPoints[2], quadPoints[3], quadPoints[0]);
        allTriCoords[1] = { quadPoints[0], quadPoints[1], quadPoints[2] };
        allTriCoords[2] = { quadPoints[2], quadPoints[3], quadPoints[0] };
    }
    else
    {
        newTris[1] = Triangle(baseQuadriID[0], baseQuadriID[1], baseQuadriID[3]);
        newTris[2] = Triangle(baseQuadriID[1], baseQuadriID[2], baseQuadriID[3]);
        newAreas[1] = geometry::Triangle::area(quadPoints[0], quadPoints[1], quadPoints[3]);
        newAreas[2] = geometry::Triangle::area(quadPoints[1], quadPoints[2], quadPoints[3]);
        allTriCoords[1] = { quadPoints[0], quadPoints[1], quadPoints[3] };
        allTriCoords[2] = { quadPoints[1], quadPoints[2], quadPoints[3] };
    }

    type::vector<TriangleID> ancestors;
    ancestors.push_back(m_triangleId);
    for (unsigned int i = 0; i < 3; i++)
    {
        type::vector<SReal> coefs;
        coefs.push_back(newAreas[i] / areaFull);
        auto TTA = new TriangleToAdd(1000000 * m_triangleId + i, newTris[i], ancestors, coefs);
        TTA->m_triCoords = allTriCoords[i];
        m_trianglesToAdd.push_back(TTA);
    }

    return true;
}


bool TriangleSubdivider_3Edge::subdivide(const sofa::type::fixed_array<sofa::type::Vec3, 3>& triCoords)
{
    if (m_points.size() != 3)
    {
        msg_error("TriangleSubdivider_3Edge") << "There are no 2 points to add to subdivide triangle id: " << m_triangleId;
        return false;
    }

    sofa::type::fixed_array<PointID, 3> newIDs;
    sofa::type::fixed_array<sofa::type::Vec3, 3> newPoints;
    for (unsigned int i = 0; i < 3; i++)
    {
        const PointID uniqID = getUniqueId(m_triangle[(i + 1) % 3], m_triangle[(i + 2) % 3]);
        bool found = false;
        for (auto PTA : m_points)
        {
            if (PTA->m_uniqueID == uniqID)
            {
                newIDs[i] = PTA->m_idPoint;
                if (m_triangle[(i + 1) % 3] == PTA->m_ancestors[0])
                    newPoints[i] = triCoords[(i + 1) % 3] * PTA->m_coefs[0] + triCoords[(i + 2) % 3] * PTA->m_coefs[1];
                else
                    newPoints[i] = triCoords[(i + 1) % 3] * PTA->m_coefs[1] + triCoords[(i + 2) % 3] * PTA->m_coefs[0];

                found = true;
                break;
            }
        }

        if (!found)
        {
            msg_error("TriangleSubdivider_3Node") << "Unique ID not found in the list of future point to be added: " << uniqID;
            return false;
        }
    }

    sofa::type::fixed_array<Triangle, 4> newTris;
    sofa::type::fixed_array<SReal, 4> newAreas;
    sofa::type::fixed_array<sofa::type::fixed_array<sofa::type::Vec3, 3>, 4> allTriCoords;
    newTris[0] = Triangle(newIDs[1], newIDs[0], m_triangle[2]);
    newTris[1] = Triangle(newIDs[2], newIDs[1], m_triangle[0]);
    newTris[2] = Triangle(newIDs[0], newIDs[2], m_triangle[1]);
    newTris[3] = Triangle(newIDs[0], newIDs[1], newIDs[2]);

    SReal areaFull = geometry::Triangle::area(triCoords[0], triCoords[1], triCoords[2]);
    newAreas[0] = geometry::Triangle::area(newPoints[1], newPoints[0], triCoords[2]);
    newAreas[1] = geometry::Triangle::area(newPoints[2], newPoints[1], triCoords[0]);
    newAreas[2] = geometry::Triangle::area(newPoints[0], newPoints[2], triCoords[1]);
    newAreas[3] = geometry::Triangle::area(newPoints[0], newPoints[1], newPoints[2]);
    allTriCoords[0] = { newPoints[1], newPoints[0], triCoords[2] };
    allTriCoords[1] = { newPoints[2], newPoints[1], triCoords[0] };
    allTriCoords[2] = { newPoints[0], newPoints[2], triCoords[1] };
    allTriCoords[3] = { newPoints[0], newPoints[1], newPoints[2] };

    type::vector<TriangleID> ancestors;
    ancestors.push_back(m_triangleId);
    for (unsigned int i = 0; i < 4; i++)
    {
        type::vector<SReal> coefs;
        coefs.push_back(newAreas[i] / areaFull);
        auto TTA = new TriangleToAdd(1000000 * m_triangleId + i, newTris[i], ancestors, coefs);
        TTA->m_triCoords = allTriCoords[i];

        m_trianglesToAdd.push_back(TTA);
    }

    return true;
}


bool TriangleSubdivider_2Node::subdivide(const sofa::type::fixed_array<sofa::type::Vec3, 3>& triCoords)
{
    if (m_points.size() != 2)
    {
        msg_error("TriangleSubdivider_2Node") << "There are no 2 points to add to subdivide triangle id: " << m_triangleId;
        return false;
    }

    // Find intersected Edge
    PointID localEdgeId = InvalidID;
    PointID ptOnEdgeId = InvalidID;
    PointID ptInTriId = InvalidID;
    sofa::type::Vec3 ptInTri, ptOnEdge;

    const PointToAdd* PTA0 = m_points[0];
    const PointToAdd* PTA1 = m_points[1];

    for (unsigned int i = 0; i < 3; i++)
    {
        const PointID uniqID = getUniqueId(m_triangle[(i + 1) % 3], m_triangle[(i + 2) % 3]);

        if (PTA0->m_uniqueID == uniqID)
        {
            localEdgeId = i;
            ptOnEdgeId = PTA0->m_idPoint;
            ptInTriId = PTA1->m_idPoint;
            ptInTri = computePointCoordinates(PTA1, triCoords);
            ptOnEdge = computePointCoordinates(PTA0, triCoords);
            break;
        }
        else if (PTA1->m_uniqueID == uniqID)
        {
            localEdgeId = i;
            ptOnEdgeId = PTA1->m_idPoint;
            ptInTriId = PTA0->m_idPoint;
            ptInTri = computePointCoordinates(PTA0, triCoords);
            ptOnEdge = computePointCoordinates(PTA1, triCoords);
            break;
        }
    }

    if (localEdgeId == InvalidID)
    {
        msg_error("TriangleSubdivider_2Node") << "Unique ID on edge not found in the list of future point to be added.";
        return false;
    }

    sofa::type::fixed_array<Triangle, 4> newTris;
    sofa::type::fixed_array<SReal, 4> newAreas;
    sofa::type::fixed_array<sofa::type::fixed_array<sofa::type::Vec3, 3>, 4> allTriCoords;
    newTris[0] = Triangle(m_triangle[localEdgeId], m_triangle[(localEdgeId + 1) % 3], ptInTriId);
    newTris[1] = Triangle(m_triangle[(localEdgeId + 2) % 3], m_triangle[localEdgeId], ptInTriId);
    newTris[2] = Triangle(m_triangle[(localEdgeId + 1) % 3], ptOnEdgeId, ptInTriId);
    newTris[3] = Triangle(ptOnEdgeId, m_triangle[(localEdgeId + 2) % 3], ptInTriId);

    SReal areaFull = geometry::Triangle::area(triCoords[0], triCoords[1], triCoords[2]);
    newAreas[0] = geometry::Triangle::area(triCoords[localEdgeId], triCoords[(localEdgeId + 1) % 3], ptInTri);
    newAreas[1] = geometry::Triangle::area(triCoords[(localEdgeId + 2) % 3], triCoords[localEdgeId], ptInTri);
    newAreas[2] = geometry::Triangle::area(triCoords[(localEdgeId + 1) % 3], ptOnEdge, ptInTri);
    newAreas[3] = geometry::Triangle::area(ptOnEdge, triCoords[(localEdgeId + 2) % 3], ptInTri);
    allTriCoords[0] = { triCoords[localEdgeId], triCoords[(localEdgeId + 1) % 3], ptInTri };
    allTriCoords[1] = { triCoords[(localEdgeId + 2) % 3], triCoords[localEdgeId], ptInTri };
    allTriCoords[2] = { triCoords[(localEdgeId + 1) % 3], ptOnEdge, ptInTri };
    allTriCoords[3] = { ptOnEdge, triCoords[(localEdgeId + 2) % 3], ptInTri };

    type::vector<TriangleID> ancestors;
    ancestors.push_back(m_triangleId);
    for (unsigned int i = 0; i < 4; i++)
    {
        type::vector<SReal> coefs;
        coefs.push_back(newAreas[i] / areaFull);
        auto TTA = new TriangleToAdd(1000000 * m_triangleId + i, newTris[i], ancestors, coefs);
        TTA->m_triCoords = allTriCoords[i];

        m_trianglesToAdd.push_back(TTA);
    }

    return true;
}

} //namespace sofa::component::topology::container::dynamic
