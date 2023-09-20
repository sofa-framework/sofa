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
#include <sofa/component/topology/utility/TopologyChecker.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>


#include <sofa/simulation/Simulation.h>

namespace sofa::component::topology::utility
{

using namespace defaulttype;
using namespace sofa::core::topology;


int TopologyCheckerClass = core::RegisterObject("Read topological Changes and process them.")
        .add< TopologyChecker >();


TopologyChecker::TopologyChecker()
    : d_eachStep(initData(&d_eachStep, false, "eachStep", "Check topology at each step"))
    , l_topology(initLink("topology", "link to the topology container"))
    , m_topology(nullptr)
{
    this->f_listening.setValue(true);
}


TopologyChecker::~TopologyChecker()
{

}


void TopologyChecker::init()
{
    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (m_topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
}

void TopologyChecker::reinit()
{
    checkTopology();
}


bool TopologyChecker::checkTopology()
{
    bool result = false;
    if (m_topology->getTopologyType() == geometry::ElementType::HEXAHEDRON)
        result = checkHexahedronTopology();
    if (m_topology->getTopologyType() == geometry::ElementType::TETRAHEDRON)
        result = checkTetrahedronTopology();
    else if (m_topology->getTopologyType() == geometry::ElementType::QUAD)
        result = checkQuadTopology();
    else if (m_topology->getTopologyType() == geometry::ElementType::TRIANGLE)
        result = checkTriangleTopology();
    else if (m_topology->getTopologyType() == geometry::ElementType::EDGE)
        result = checkEdgeTopology();

    return result;
}



bool TopologyChecker::checkEdgeTopology()
{
    // check edge container consistency
    if (checkEdgeContainer() == false)
        return false;

    // check cross elements consistency
    return checkEdgeToVertexCrossContainer();
}


bool TopologyChecker::checkEdgeContainer()
{
    const sofa::Size nbE = m_topology->getNbEdges();
    const sofa::core::topology::BaseMeshTopology::SeqEdges& my_edges = m_topology->getEdges();

    if (nbE != my_edges.size())
    {
        msg_error() << "CheckEdgeTopology failed: not the good number of edges, getNbEdges returns " << nbE << " whereas edge array size is: " << my_edges.size();
        return false;
    }

    // check edge buffer
    bool res = true;
    for (sofa::Index i = 0; i < nbE; ++i)
    {
        const auto& edge = my_edges[i];
        if (edge[0] == edge[1])
        {
            msg_error() << "CheckEdgeTopology failed: edge " << i << " has 2 identical vertices: " << edge;
            res = false;
        }
        if (std::any_of(edge.begin(), edge.end(), [](const auto v){ return v == sofa::InvalidID;}))
        {
            msg_error() << "CheckEdgeTopology failed: edge " << i << " has an invalid vertex id: " << edge;
            res = false;
        }
    }

    return res;
}

bool TopologyChecker::checkEdgeToVertexCrossContainer()
{
    const sofa::core::topology::BaseMeshTopology::SeqEdges& my_edges = m_topology->getEdges();
    std::set<int> edgeSet;

    bool res = true;
    const sofa::Size nbP = m_topology->getNbPoints();
    for (sofa::Index i = 0; i < nbP; ++i)
    {
        const auto& EdgesAV = m_topology->getEdgesAroundVertex(i);
        for (size_t j = 0; j < EdgesAV.size(); ++j)
        {
            if (EdgesAV[j] != sofa::InvalidID)
            {
                const Topology::Edge& edge = my_edges[EdgesAV[j]];
                if (!(edge[0] == i || edge[1] == i))
                {
                    msg_error() << "CheckEdgeTopology failed: edge " << EdgesAV[j] << ": [" << edge << "] not around vertex: " << i;
                    res = false;
                }

                // count number of edge
                edgeSet.insert(EdgesAV[j]);
            }
            else
            {
                msg_error() << "CheckEdgeTopology failed: edge " << j << " around vertex " << i << " is an invalid id";
                res = false;
            }
        }
    }

    if (edgeSet.size() != my_edges.size())
    {
        msg_error() << "CheckEdgeTopology failed: found " << edgeSet.size() << " edges in m_edgesAroundVertex out of " << my_edges.size();
        res = false;
    }

    return res;
}




bool TopologyChecker::checkTriangleTopology()
{
    // check edge container consistency
    if (checkTriangleContainer() == false)
        return false;

    // check cross elements consistency
    bool res = checkTriangleToEdgeCrossContainer();
    
    res = res && checkTriangleToVertexCrossContainer();

    return res && checkEdgeTopology();
}


bool TopologyChecker::checkTriangleContainer()
{
    const sofa::Size nbT = m_topology->getNbTriangles();
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& my_triangles = m_topology->getTriangles();

    if (nbT != my_triangles.size())
    {
        msg_error() << "CheckTriangleTopology failed: not the good number of triangles, getNbTriangles returns " << nbT << " whereas triangle array size is: " << my_triangles.size();
        return false;
    }

    // check triangle buffer
    bool res = true;
    for (sofa::Index i = 0; i < nbT; ++i)
    {
        const auto& triangle = my_triangles[i];
        if (triangle[0] == triangle[1] || triangle[0] == triangle[2] || triangle[1] == triangle[2])
        {
            msg_error() << "CheckTriangleTopology failed: triangle " << i << " has 2 identical vertices: " << triangle;
            res = false;
        }
        if (std::any_of(triangle.begin(), triangle.end(), [](const auto v){ return v == sofa::InvalidID;}))
        {
            msg_error() << "CheckEdgeTopology failed: triangle " << i << " has an invalid vertex id: " << triangle;
            res = false;
        }
    }

    return res;
}


bool TopologyChecker::checkTriangleToEdgeCrossContainer()
{
    bool ret = true;
    const sofa::Size nbT = m_topology->getNbTriangles();
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& my_triangles = m_topology->getTriangles();

    // check triangles buffer
    std::set <int> triangleSet;

    const sofa::Size nbE = m_topology->getNbEdges();
    const sofa::core::topology::BaseMeshTopology::SeqEdges& my_edges = m_topology->getEdges();
    
    // check edges in triangles
    for (sofa::Index i = 0; i < nbT; ++i)
    {
        const Topology::Triangle& triangle = my_triangles[i];
        const auto& eInTri = m_topology->getEdgesInTriangle(i);

        for (unsigned int j = 0; j < 3; j++)
        {
            if (eInTri[j] == sofa::InvalidID)
            {
                msg_error() << "CheckTriangleTopology failed: EdgesInTriangle of triangle: " << i << ": " << triangle << " has invalid ID: " << eInTri;
                ret = false;
                continue;
            }

            const Topology::Edge& edge = my_edges[eInTri[j]];
            int cptFound = 0;
            for (unsigned int k = 0; k < 3; k++)
                if (edge[0] == triangle[k] || edge[1] == triangle[k])
                    cptFound++;

            if (cptFound != 2)
            {
                msg_error() << "CheckTriangleTopology failed: edge: " << eInTri[j] << ": [" << edge << "] not found in triangle: " << i << ": " << triangle;
                ret = false;
            }
        }
    }


    // check triangles around edges
    // check m_trianglesAroundEdge using checked m_edgesInTriangle
    triangleSet.clear();
    for (sofa::Index edgeId = 0; edgeId < nbE; ++edgeId)
    {
        const BaseMeshTopology::TrianglesAroundEdge& tes = m_topology->getTrianglesAroundEdge(edgeId);
        for (auto triId : tes)
        {
            const BaseMeshTopology::EdgesInTriangle& eInTri = m_topology->getEdgesInTriangle(triId);
            const bool check_triangle_edge_shell = (eInTri[0] == edgeId)
                || (eInTri[1] == edgeId)
                || (eInTri[2] == edgeId);
            if (!check_triangle_edge_shell)
            {
                msg_error() << "CheckTriangleTopology failed: triangle: " << triId << " with edges: [" << eInTri << "] not found around edge: " << edgeId;
                ret = false;
            }

            triangleSet.insert(triId);
        }
    }

    if (triangleSet.size() != my_triangles.size())
    {
        msg_error() << "CheckTriangleTopology failed: found " << triangleSet.size() << " triangles in m_trianglesAroundEdge out of " << my_triangles.size();
        ret = false;
    }

    return ret;
}


bool TopologyChecker::checkTriangleToVertexCrossContainer()
{
    bool ret = true;
    // check cross element
    const sofa::Size nbP = m_topology->getNbPoints();
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& my_triangles = m_topology->getTriangles();

    // check triangles around vertex
    std::set <int> triangleSet;
    for (sofa::Index i = 0; i < nbP; ++i)
    {
        const auto& triAV = m_topology->getTrianglesAroundVertex(i);
        for (size_t j = 0; j < triAV.size(); ++j)
        {
            if (triAV[j] != sofa::InvalidID)
            {
                const Topology::Triangle& triangle = my_triangles[triAV[j]];
                const bool check_triangle_vertex_shell = (triangle[0] == i) || (triangle[1] == i) || (triangle[2] == i);
                if (!check_triangle_vertex_shell)
                {
                    msg_error() << "CheckTriangleTopology failed: triangle " << triAV[j] << ": [" << triangle << "] not around vertex: " << i;
                    ret = false;
                }

                triangleSet.insert(triAV[j]);
            }
            else
            {
                msg_error() << "CheckTriangleTopology failed: triangle " << j << " around vertex " << i << " is an invalid id";
                ret = false;
            }
        }
    }

    if (triangleSet.size() != my_triangles.size())
    {
        msg_error() << "CheckTriangleTopology failed: found " << triangleSet.size() << " triangles in trianglesAroundVertex out of " << my_triangles.size();
        ret = false;
    }

    return ret;
}




bool TopologyChecker::checkQuadTopology()
{
    // check edge container consistency
    if (checkQuadContainer() == false)
        return false;

    // check cross elements consistency
    bool res = checkQuadToEdgeCrossContainer();

    res = res && checkQuadToVertexCrossContainer();

    return res && checkEdgeTopology();
}


bool TopologyChecker::checkQuadContainer()
{
    const sofa::Size nbQ = m_topology->getNbQuads();
    const sofa::core::topology::BaseMeshTopology::SeqQuads& my_quads = m_topology->getQuads();

    if (nbQ != my_quads.size())
    {
        msg_error() << "checkQuadTopology failed: not the good number of quads, getNbQuads returns " << nbQ << " whereas quad array size is: " << my_quads.size();
        return false;
    }

    // check quad buffer
    bool res = true;
    for (sofa::Index i = 0; i < nbQ; ++i)
    {
        const auto& quad = my_quads[i];
        for (int j = 0; j < 3; ++j)
        {
            for (int k = j + 1; k < 4; ++k)
            {
                if (quad[j] == quad[k])
                {
                    msg_error() << "checkQuadTopology failed: quad " << i << " has 2 identical vertices: " << quad;
                    res = false;
                }
            }
        }
        if (std::any_of(quad.begin(), quad.end(), [](const auto v){ return v == sofa::InvalidID;}))
        {
            msg_error() << "CheckEdgeTopology failed: quad " << i << " has an invalid vertex id: " << quad;
            res = false;
        }
    }

    return res;
}


bool TopologyChecker::checkQuadToEdgeCrossContainer()
{
    bool ret = true;
    const sofa::Size nbQ = m_topology->getNbQuads();
    const sofa::core::topology::BaseMeshTopology::SeqQuads& my_quads = m_topology->getQuads();

    std::set <int> quadSet;

    const sofa::Size nbE = m_topology->getNbEdges();
    const sofa::core::topology::BaseMeshTopology::SeqEdges& my_edges = m_topology->getEdges();
    
    // check edges in quads
    for (sofa::Index i = 0; i < nbQ; ++i)
    {
        const Topology::Quad& quad = my_quads[i];
        const auto& eInQ = m_topology->getEdgesInQuad(i);

        for (auto eId : eInQ)
        {
            if (eId == sofa::InvalidID)
            {
                msg_error() << "CheckQuadTopology failed: EdgesInQuad of quad: " << i << ": " << quad << " has invalid ID: " << eInQ;
                ret = false;
                continue;
            }

            const Topology::Edge& edge = my_edges[eId];
            int cptFound = 0;
            for (unsigned int k = 0; k < 4; k++)
                if (edge[0] == quad[k] || edge[1] == quad[k])
                    cptFound++;

            if (cptFound != 2)
            {
                msg_error() << "CheckQuadTopology failed: edge: " << eId << ": [" << edge << "] not found in quad: " << i << ": " << quad;
                ret = false;
            }
        }
    }

    // check quads around edges
    // check m_quadsAroundEdge using checked m_edgesInQuad
    quadSet.clear();
    for (sofa::Index edgeId = 0; edgeId < nbE; ++edgeId)
    {
        const BaseMeshTopology::QuadsAroundEdge& qAE = m_topology->getQuadsAroundEdge(edgeId);
        for (auto qId : qAE)
        {
            const BaseMeshTopology::EdgesInQuad& eInQ = m_topology->getEdgesInQuad(qId);
            const bool check_quad_edge_shell = (eInQ[0] == edgeId)
                || (eInQ[1] == edgeId)
                || (eInQ[2] == edgeId)
                || (eInQ[3] == edgeId);
            if (!check_quad_edge_shell)
            {
                msg_error() << "CheckQuadTopology failed: quad: " << qId << " with edges: [" << eInQ << "] not found around edge: " << edgeId;
                ret = false;
            }

            quadSet.insert(qId);
        }
    }

    if (quadSet.size() != my_quads.size())
    {
        msg_error() << "CheckQuadTopology failed: found " << quadSet.size() << " quads in m_quadsAroundEdge out of " << my_quads.size();
        ret = false;
    }

    return ret;
}


bool TopologyChecker::checkQuadToVertexCrossContainer()
{
    bool ret = true;
    // check cross element
    const sofa::Size nbP = m_topology->getNbPoints();
    const sofa::core::topology::BaseMeshTopology::SeqQuads& my_quads = m_topology->getQuads();

    // check quads around vertex
    std::set <int> quadSet;
    for (sofa::Index i = 0; i < nbP; ++i)
    {
        const auto& quadAV = m_topology->getQuadsAroundVertex(i);
        for (size_t j = 0; j < quadAV.size(); ++j)
        {
            if (quadAV[j] != sofa::InvalidID)
            {
                const Topology::Quad& quad = my_quads[quadAV[j]];
                const bool check_quad_vertex_shell = (quad[0] == i) || (quad[1] == i) || (quad[2] == i) || (quad[3] == i);
                if (!check_quad_vertex_shell)
                {
                    msg_error() << "CheckQuadTopology failed: quad " << quadAV[j] << ": [" << quad << "] not around vertex: " << i;
                    ret = false;
                }

                quadSet.insert(quadAV[j]);
            }
            else
            {
                msg_error() << "CheckQuadTopology failed: quad " << j << " around vertex " << i << " is an invalid id";
                ret = false;
            }
        }
    }

    if (quadSet.size() != my_quads.size())
    {
        msg_error() << "CheckQuadTopology failed: found " << quadSet.size() << " quads in quadsAroundVertex out of " << my_quads.size();
        ret = false;
    }

    return ret;
}




bool TopologyChecker::checkTetrahedronTopology()
{
    // check edge container consistency
    if (checkTetrahedronContainer() == false)
        return false;

    // check cross elements consistency
    bool res = checkTetrahedronToTriangleCrossContainer();

    res = res && checkTetrahedronToEdgeCrossContainer();

    res = res && checkTetrahedronToVertexCrossContainer();

    return res && checkTriangleTopology();
}


bool TopologyChecker::checkTetrahedronContainer()
{
    const sofa::Size nbT = m_topology->getNbTetrahedra();
    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& my_tetrahedra = m_topology->getTetrahedra();

    if (nbT != my_tetrahedra.size())
    {
        msg_error() << "checkTetrahedronTopology failed: not the good number of tetrahedra, getNbTetrahedra returns " << nbT << " whereas triangle array size is: " << my_tetrahedra.size();
        return false;
    }

    // check tetrahedron buffer
    bool res = true;
    for (sofa::Index i = 0; i < nbT; ++i)
    {
        const auto& tetra = my_tetrahedra[i];
        for (int j = 0; j < 3; ++j)
        {
            for (int k = j + 1; k < 4; ++k)
            {
                if (tetra[j] == tetra[k])
                {
                    msg_error() << "checkTetrahedronTopology failed: tetrahedron " << i << " has 2 identical vertices: " << tetra;
                    res = false;
                }
            }
        }
        if (std::any_of(tetra.begin(), tetra.end(), [](const auto v){ return v == sofa::InvalidID;}))
        {
            msg_error() << "CheckEdgeTopology failed: tetrahedron " << i << " has an invalid vertex id: " << tetra;
            res = false;
        }
    }

    return res;
}


bool TopologyChecker::checkTetrahedronToTriangleCrossContainer()
{
    bool ret = true;
    const sofa::Size nbT = m_topology->getNbTetrahedra();
    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& my_tetrahedra = m_topology->getTetrahedra();

    std::set <int> tetrahedronSet;

    const sofa::Size nbTri = m_topology->getNbTriangles();
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& my_triangles = m_topology->getTriangles();
    
    // check first m_trianglesInTetrahedron
    for (sofa::Index tetraId = 0; tetraId < nbT; ++tetraId)
    {
        const Topology::Tetrahedron& tetrahedron = my_tetrahedra[tetraId];
        const auto& triInTetra = m_topology->getTrianglesInTetrahedron(tetraId);

        for (unsigned int j = 0; j < 4; j++)
        {
            if (triInTetra[j] == sofa::InvalidID)
            {
                msg_error() << "checkTetrahedronTopology failed: TrianglesInTetrahedron of tetrahedron: " << tetraId << ": " << tetrahedron << " has invalid ID: " << triInTetra;
                ret = false;
                continue;
            }

            const Topology::Triangle& triangle = my_triangles[triInTetra[j]];
            int cptFound = 0;
            for (unsigned int k = 0; k < 4; k++)
                if (triangle[0] == tetrahedron[k] || triangle[1] == tetrahedron[k] || triangle[2] == tetrahedron[k])
                    cptFound++;

            if (cptFound != 3)
            {
                msg_error() << "checkTetrahedronTopology failed: triangle: " << triInTetra[j] << ": [" << triangle << "] not found in tetrahedron: " << tetraId << ": " << tetrahedron;
                ret = false;
            }
        }
    }

    // check tetrahedra around triangles
    // check m_tetrahedraAroundTriangle using checked m_trianglesInTetrahedron
    tetrahedronSet.clear();
    for (sofa::Index triId = 0; triId < nbTri; ++triId)
    {
        const BaseMeshTopology::TetrahedraAroundTriangle& tes = m_topology->getTetrahedraAroundTriangle(triId);
        for (auto tetraId : tes)
        {
            const BaseMeshTopology::TrianglesInTetrahedron& triInTetra = m_topology->getTrianglesInTetrahedron(tetraId);
            const bool check_tetra_triangle_shell = (triInTetra[0] == triId)
                || (triInTetra[1] == triId)
                || (triInTetra[2] == triId)
                || (triInTetra[3] == triId);
            if (!check_tetra_triangle_shell)
            {
                msg_error() << "checkTetrahedronTopology failed: tetrahedron: " << tetraId << " with triangle: [" << triInTetra << "] not found around triangle: " << tetraId;
                ret = false;
            }

            tetrahedronSet.insert(tetraId);
        }
    }

    if (tetrahedronSet.size() != my_tetrahedra.size())
    {
        msg_error() << "checkTetrahedronTopology failed: found " << tetrahedronSet.size() << " tetrahedra in m_tetrahedraAroundTriangle out of " << my_tetrahedra.size();
        ret = false;
    }

    return ret;
}


bool TopologyChecker::checkTetrahedronToEdgeCrossContainer()
{
    bool ret = true;
    const sofa::Size nbT = m_topology->getNbTetrahedra();
    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& my_tetrahedra = m_topology->getTetrahedra();

    std::set <int> tetrahedronSet;

    const sofa::Size nbE = m_topology->getNbEdges();
    const sofa::core::topology::BaseMeshTopology::SeqEdges& my_edges = m_topology->getEdges();
    // check edges in tetrahedra
    for (sofa::Index i = 0; i < nbT; ++i)
    {
        const Topology::Tetrahedron& tetrahedron = my_tetrahedra[i];
        const auto& eInTetra = m_topology->getEdgesInTetrahedron(i);

        for (unsigned int j = 0; j < 6; j++)
        {
            if (eInTetra[j] != sofa::InvalidID)
            {
                const Topology::Edge& edge = my_edges[eInTetra[j]];
                int cptFound = 0;
                for (unsigned int k = 0; k < 4; k++)
                    if (edge[0] == tetrahedron[k] || edge[1] == tetrahedron[k])
                        cptFound++;

                if (cptFound != 2)
                {
                    msg_error() << "checkTetrahedronTopology failed: edge: " << eInTetra[j] << ": [" << edge << "] not found in tetrahedron: " << i << ": " << tetrahedron;
                    ret = false;
                }
            }
            else
            {
                msg_error() << "checkTetrahedronTopology failed: edge " << j << " in tetrahedron " << i << " (" << tetrahedron << ") is an invalid id";
                ret = false;
            }

        }
    }

    // check tetrahedra around edges
    // check m_tetrahedraAroundEdge using checked m_edgesInTetrahedron
    tetrahedronSet.clear();
    for (sofa::Index edgeId = 0; edgeId < nbE; ++edgeId)
    {
        const BaseMeshTopology::TetrahedraAroundEdge& tes = m_topology->getTetrahedraAroundEdge(edgeId);
        for (auto tetraId : tes)
        {
            const BaseMeshTopology::EdgesInTetrahedron& eInTetra = m_topology->getEdgesInTetrahedron(tetraId);
            const bool check_tetra_edge_shell = (eInTetra[0] == edgeId)
                || (eInTetra[1] == edgeId)
                || (eInTetra[2] == edgeId)
                || (eInTetra[3] == edgeId)
                || (eInTetra[4] == edgeId)
                || (eInTetra[5] == edgeId);
            if (!check_tetra_edge_shell)
            {
                msg_error() << "checkTetrahedronTopology failed: tetrahedron: " << tetraId << " with edges: [" << eInTetra << "] not found around edge: " << edgeId;
                ret = false;
            }

            tetrahedronSet.insert(tetraId);
        }
    }

    if (tetrahedronSet.size() != my_tetrahedra.size())
    {
        msg_error() << "CheckTriangleTopology failed: found " << tetrahedronSet.size() << " tetrahedra in m_tetrahedraAroundTriangle out of " << my_tetrahedra.size();
        ret = false;
    }

    return ret;
}


bool TopologyChecker::checkTetrahedronToVertexCrossContainer()
{
    bool ret = true;
    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& my_tetrahedra = m_topology->getTetrahedra();

    // check cross element
    const sofa::Size nbP = m_topology->getNbPoints();

    // check tetrahedra around vertex
    std::set <int> tetrahedronSet;
    for (sofa::Index pId = 0; pId < nbP; ++pId)
    {
        const auto& tetraAV = m_topology->getTetrahedraAroundVertex(pId);
        for (auto tetraId : tetraAV)
        {
            if (tetraId != sofa::InvalidID)
            {
                const Topology::Tetrahedron& tetra = my_tetrahedra[tetraId];
                const bool check_tetra_vertex_shell = (tetra[0] == pId)
                    || (tetra[1] == pId)
                    || (tetra[2] == pId)
                    || (tetra[3] == pId);
                if (!check_tetra_vertex_shell)
                {
                    msg_error() << "checkTetrahedronTopology failed: Tetrahedron " << tetraId << ": [" << tetra << "] not around vertex: " << pId;
                    ret = false;
                }

                tetrahedronSet.insert(tetraId);
            }
            else
            {
                msg_error() << "checkTetrahedronTopology failed: Tetrahedron around vertex " << pId << " is an invalid id";
                ret = false;
            }
        }
    }

    if (tetrahedronSet.size() != my_tetrahedra.size())
    {
        msg_error() << "checkTetrahedronTopology failed: found " << tetrahedronSet.size() << " tetrahedra in tetrahedraAroundVertex out of " << my_tetrahedra.size();
        ret = false;
    }

    return ret;
}




bool TopologyChecker::checkHexahedronTopology()
{
    // check edge container consistency
    if (checkHexahedronContainer() == false)
        return false;

    // check cross elements consistency
    bool res = checkHexahedronToQuadCrossContainer();

    res = res && checkHexahedronToEdgeCrossContainer();

    res = res && checkHexahedronToVertexCrossContainer();

    return res && checkQuadTopology();
}


bool TopologyChecker::checkHexahedronContainer()
{
    const sofa::Size nbH = m_topology->getNbHexahedra();
    const sofa::core::topology::BaseMeshTopology::SeqHexahedra& my_hexahedra = m_topology->getHexahedra();

    if (nbH != my_hexahedra.size())
    {
        msg_error() << "checkHexahedronTopology failed: not the good number of hexahedra, getNbHexahedra returns " << nbH << " whereas hexahedra array size is: " << my_hexahedra.size();
        return false;
    }

    // check hexahedron buffer
    bool res = true;
    for (sofa::Index i = 0; i < nbH; ++i)
    {
        const auto& hexahedron = my_hexahedra[i];
        for (int j = 0; j < 7; ++j)
        {
            for (int k = j + 1; k < 8; ++k)
            {
                if (hexahedron[j] == hexahedron[k])
                {
                    msg_error() << "checkHexahedronTopology failed: hexahedron " << i << " has 2 identical vertices: " << hexahedron;
                    res = false;
                }
            }
        }
        if (std::any_of(hexahedron.begin(), hexahedron.end(), [](const auto v){ return v == sofa::InvalidID;}))
        {
            msg_error() << "CheckEdgeTopology failed: hexahedron " << i << " has an invalid vertex id: " << hexahedron;
            res = false;
        }
    }

    return res;
}


bool TopologyChecker::checkHexahedronToQuadCrossContainer()
{
    bool ret = true;
    const sofa::Size nbH = m_topology->getNbHexahedra();
    const sofa::core::topology::BaseMeshTopology::SeqHexahedra& my_hexahedra = m_topology->getHexahedra();

    std::set <int> hexahedronSet;

    const sofa::Size nbQ = m_topology->getNbQuads();
    const sofa::core::topology::BaseMeshTopology::SeqQuads& my_quads = m_topology->getQuads();
    
    // check first m_quadsInHexahedron
    for (sofa::Index hexaId = 0; hexaId < nbH; ++hexaId)
    {
        const Topology::Hexahedron& hexahedron = my_hexahedra[hexaId];
        const auto& qInHexa = m_topology->getQuadsInHexahedron(hexaId);

        for (auto qId : qInHexa)
        {
            if (qId == sofa::InvalidID)
            {
                msg_error() << "checkHexahedronTopology failed: QuadsInHexahedron of hexahedron: " << hexaId << ": " << hexahedron << " has invalid ID: " << qInHexa;
                ret = false;
                continue;
            }

            const Topology::Quad& quad = my_quads[qId];
            int cptFound = 0;
            for (unsigned int k = 0; k < 8; k++)
                if (quad[0] == hexahedron[k] || quad[1] == hexahedron[k] || quad[2] == hexahedron[k] || quad[3] == hexahedron[k])
                    cptFound++;

            if (cptFound != 4)
            {
                msg_error() << "checkHexahedronTopology failed: quad: " << qId << ": [" << quad << "] not found in hexahedron: " << hexaId << ": " << hexahedron;
                ret = false;
            }
        }
    }

    // check hexahedra around triangles
    // check m_hexahedraAroundTriangle using checked m_trianglesInHexahedron
    hexahedronSet.clear();
    for (sofa::Index qId = 0; qId < nbQ; ++qId)
    {
        const BaseMeshTopology::HexahedraAroundQuad& hAq = m_topology->getHexahedraAroundQuad(qId);
        for (auto hexaId : hAq)
        {
            const BaseMeshTopology::QuadsInHexahedron& qInHexa = m_topology->getQuadsInHexahedron(hexaId);
            bool check_hexa_quad_shell = false;
            for (const auto quadID : qInHexa)
            {
                if (quadID == qId) {
                    check_hexa_quad_shell = true;
                    break;
                }
            }

            if (!check_hexa_quad_shell)
            {
                msg_error() << "checkHexahedronTopology failed: hexahedron: " << hexaId << " with quad: [" << qInHexa << "] not found around quad: " << qId;
                ret = false;
            }

            hexahedronSet.insert(hexaId);
        }
    }

    if (hexahedronSet.size() != my_hexahedra.size())
    {
        msg_error() << "checkHexahedronTopology failed: found " << hexahedronSet.size() << " hexahedra in m_hexahedraAroundQuad out of " << my_hexahedra.size();
        ret = false;
    }

    return ret;
}


bool TopologyChecker::checkHexahedronToEdgeCrossContainer()
{
    bool ret = true;
    const sofa::Size nbH = m_topology->getNbHexahedra();
    const sofa::core::topology::BaseMeshTopology::SeqHexahedra& my_hexahedra = m_topology->getHexahedra();

    std::set <int> hexahedronSet;

    const sofa::Size nbE = m_topology->getNbEdges();
    const sofa::core::topology::BaseMeshTopology::SeqEdges& my_edges = m_topology->getEdges();
    // check edges in hexahedra
    for (sofa::Index i = 0; i < nbH; ++i)
    {
        const Topology::Hexahedron& hexahedron = my_hexahedra[i];
        const auto& eInHexa = m_topology->getEdgesInHexahedron(i);

        for (unsigned int j = 0; j < 6; j++)
        {
            if (eInHexa[j] != sofa::InvalidID)
            {
                const Topology::Edge& edge = my_edges[eInHexa[j]];
                int cptFound = 0;
                for (unsigned int k = 0; k < 8; k++)
                    if (edge[0] == hexahedron[k] || edge[1] == hexahedron[k])
                        cptFound++;

                if (cptFound != 2)
                {
                    msg_error() << "checkHexahedronTopology failed: edge: " << eInHexa[j] << ": [" << edge << "] not found in hexahedron: " << i << ": " << hexahedron;
                    ret = false;
                }
            }
            else
            {
                msg_error() << "checkHexahedronTopology failed: edge " << j << " in hexahedron " << i << " has an invalid id" ;
                ret = false;
            }
        }
    }

    // check hexahedra around edges
    // check m_hexahedraAroundEdge using checked m_edgesInHexahedron
    hexahedronSet.clear();
    for (sofa::Index edgeId = 0; edgeId < nbE; ++edgeId)
    {
        const BaseMeshTopology::HexahedraAroundEdge& hAe = m_topology->getHexahedraAroundEdge(edgeId);
        for (auto hexaId : hAe)
        {
            const BaseMeshTopology::EdgesInHexahedron& eInHexa = m_topology->getEdgesInHexahedron(hexaId);

            bool check_hexa_edge_shell = false;
            for (const auto eInID : eInHexa)
            {
                if (eInID == edgeId) {
                    check_hexa_edge_shell = true;
                    break;
                }
            }

            if (!check_hexa_edge_shell)
            {
                msg_error() << "checkHexahedronTopology failed: hexahedron: " << hexaId << " with edges: [" << eInHexa << "] not found around edge: " << edgeId;
                ret = false;
            }

            hexahedronSet.insert(hexaId);
        }
    }

    if (hexahedronSet.size() != my_hexahedra.size())
    {
        msg_error() << "checkHexahedronTopology failed: found " << hexahedronSet.size() << " hexahedra in m_hexahedraAroundEdge out of " << my_hexahedra.size();
        ret = false;
    }

    return ret;
}


bool TopologyChecker::checkHexahedronToVertexCrossContainer()
{
    bool ret = true;
    const sofa::core::topology::BaseMeshTopology::SeqHexahedra& my_hexahedra = m_topology->getHexahedra();

    // check cross element
    const sofa::Size nbP = m_topology->getNbPoints();

    // check hexahedra around vertex
    std::set <int> hexahedronSet;
    for (sofa::Index pId = 0; pId < nbP; ++pId)
    {
        const auto& hexaAV = m_topology->getHexahedraAroundVertex(pId);
        for (auto hexaId : hexaAV)
        {
            if (hexaId != sofa::InvalidID)
            {
                const Topology::Hexahedron& hexa = my_hexahedra[hexaId];
                bool check_hexa_vertex_shell = false;
                for (int j = 0; j < 8; ++j)
                {
                    if (hexa[j] == pId) {
                        check_hexa_vertex_shell = true;
                        break;
                    }
                }

                if (!check_hexa_vertex_shell)
                {
                    msg_error() << "checkHexahedronTopology failed: Hexahedron " << hexaId << ": [" << hexa << "] not around vertex: " << pId;
                    ret = false;
                }

                hexahedronSet.insert(hexaId);
            }
            else
            {
                msg_error() << "checkHexahedronTopology failed: hexahedron around vertex " << pId << " is an invalid id";
                ret = false;
            }
        }
    }

    if (hexahedronSet.size() != my_hexahedra.size())
    {
        msg_error() << "checkHexahedronTopology failed: found " << hexahedronSet.size() << " hexahedra in hexahedraAroundVertex out of " << my_hexahedra.size();
        ret = false;
    }

    return ret;
}


void TopologyChecker::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (const sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {
        if (ev->getKey() == 'T')
        {
            const bool res = checkTopology();
            if (!res)
                msg_error() << "checkTopology Error!!";
        }
    }

    if (simulation::AnimateEndEvent::checkEventType(event) && d_eachStep.getValue())
    {
        checkTopology();
    }
}


void TopologyChecker::draw(const core::visual::VisualParams* vparams)
{
    if (!m_topology)
        return;

    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

}

} // namespace sofa::component::topology::utility

