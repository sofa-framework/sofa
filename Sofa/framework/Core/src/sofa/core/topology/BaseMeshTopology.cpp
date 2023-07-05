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
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/io/MeshTopologyLoader.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/objectmodel/BaseNode.h>

namespace sofa::core::topology
{

using namespace sofa::defaulttype;
using type::vector;
using type::fixed_array;

BaseMeshTopology::BaseMeshTopology()
    : fileTopology(initData(&fileTopology,"filename","Filename of the mesh"))
{
    addAlias(&fileTopology,"fileTopology");
}

/// Returns the set of edges adjacent to a given vertex.
const BaseMeshTopology::EdgesAroundVertex& BaseMeshTopology::getEdgesAroundVertex(PointID)
{
    if (getNbEdges()) msg_error() << "getEdgesAroundVertex unsupported.";
    return InvalidSet;
}

/// Returns the set of edges adjacent to a given triangle.
const BaseMeshTopology::EdgesInTriangle& BaseMeshTopology::getEdgesInTriangle(TriangleID)
{
    if (getNbEdges()) msg_error() << "getEdgesInTriangle unsupported.";    
    return InvalidEdgesInTriangles;
}

/// Returns the set of edges adjacent to a given quad.
const BaseMeshTopology::EdgesInQuad& BaseMeshTopology::getEdgesInQuad(QuadID)
{
    if (getNbEdges()) msg_error() << "getEdgesInQuad unsupported.";
    return InvalidEdgesInQuad;
}

/// Returns the set of edges adjacent to a given tetrahedron.
const BaseMeshTopology::EdgesInTetrahedron& BaseMeshTopology::getEdgesInTetrahedron(TetraID)
{
    if (getNbEdges()) msg_error() << "getEdgesInTetrahedron unsupported.";
    return InvalidEdgesInTetrahedron;
}


/// Returns the set of edges adjacent to a given hexahedron.
const BaseMeshTopology::EdgesInHexahedron& BaseMeshTopology::getEdgesInHexahedron(HexaID)
{
    if (getNbEdges()) msg_error() << "getEdgesInHexahedron unsupported.";
    return InvalidEdgesInHexahedron;
}

/// Returns the set of triangle adjacent to a given vertex.
const BaseMeshTopology::TrianglesAroundVertex& BaseMeshTopology::getTrianglesAroundVertex(PointID)
{
    if (getNbTriangles()) msg_error() << "getTrianglesAroundVertex unsupported.";
    return InvalidSet;
}

/// Returns the set of triangle adjacent to a given edge.
const BaseMeshTopology::TrianglesAroundEdge& BaseMeshTopology::getTrianglesAroundEdge(EdgeID)
{
    if (getNbTriangles()) msg_error() << "getEdgesAroundVertex unsupported.";
    return InvalidSet;
}

/// Returns the set of triangle adjacent to a given tetrahedron.
const BaseMeshTopology::TrianglesInTetrahedron& BaseMeshTopology::getTrianglesInTetrahedron(TetraID)
{
    if (getNbTriangles()) msg_error() << "getTrianglesInTetrahedron unsupported.";
    return InvalidTrianglesInTetrahedron;
}

/// Returns the set of quad adjacent to a given vertex.
const BaseMeshTopology::QuadsAroundVertex& BaseMeshTopology::getQuadsAroundVertex(PointID)
{
    if (getNbQuads()) msg_error() << "getQuadsAroundVertex unsupported.";
    return InvalidSet;
}

/// Returns the set of quad adjacent to a given edge.
const BaseMeshTopology::QuadsAroundEdge& BaseMeshTopology::getQuadsAroundEdge(EdgeID)
{
    if (getNbQuads()) msg_error() << "getQuadsAroundEdge unsupported.";
    return InvalidSet;
}

/// Returns the set of quad adjacent to a given hexahedron.
const BaseMeshTopology::QuadsInHexahedron& BaseMeshTopology::getQuadsInHexahedron(HexaID)
{
    if (getNbQuads()) msg_error() << "getQuadsInHexahedron unsupported.";
    return InvalidQuadsInHexahedron;
}

/// Returns the set of tetrahedra adjacent to a given vertex.
const BaseMeshTopology::TetrahedraAroundVertex& BaseMeshTopology::getTetrahedraAroundVertex(PointID)
{
    if (getNbTetrahedra()) msg_error() << "getTetrahedraAroundVertex unsupported.";
    return InvalidSet;
}

/// Returns the set of tetrahedra adjacent to a given edge.
const BaseMeshTopology::TetrahedraAroundEdge& BaseMeshTopology::getTetrahedraAroundEdge(EdgeID)
{
    if (getNbTetrahedra()) msg_error() << "getTetrahedraAroundEdge unsupported.";
    return InvalidSet;
}

/// Returns the set of tetrahedra adjacent to a given triangle.
const BaseMeshTopology::TetrahedraAroundTriangle& BaseMeshTopology::getTetrahedraAroundTriangle(TriangleID)
{
    if (getNbTetrahedra()) msg_error() << "getTetrahedraAroundTriangle unsupported.";
    return InvalidSet;
}

/// Returns the set of hexahedra adjacent to a given vertex.
const BaseMeshTopology::HexahedraAroundVertex& BaseMeshTopology::getHexahedraAroundVertex(PointID)
{
    if (getNbHexahedra()) msg_error() << "getHexahedraAroundVertex unsupported.";
    return InvalidSet;
}

/// Returns the set of hexahedra adjacent to a given edge.
const BaseMeshTopology::HexahedraAroundEdge& BaseMeshTopology::getHexahedraAroundEdge(EdgeID)
{
    if (getNbHexahedra()) msg_error() << "getHexahedraAroundEdge unsupported.";
    return InvalidSet;
}

/// Returns the set of hexahedra adjacent to a given quad.
const BaseMeshTopology::HexahedraAroundQuad& BaseMeshTopology::getHexahedraAroundQuad(QuadID)
{
    if (getNbHexahedra()) msg_error() << "getHexahedraAroundQuad unsupported.";
    return InvalidSet;
}


/// Returns the set of vertices adjacent to a given vertex (i.e. sharing an edge)
const BaseMeshTopology::VerticesAroundVertex BaseMeshTopology::getVerticesAroundVertex(PointID i)
{
    const SeqEdges& edges = getEdges();
    const EdgesAroundVertex& shell = getEdgesAroundVertex(i);
    VerticesAroundVertex adjacentVertices;

    for (Size j = 0; j<shell.size(); j++)
    {
        Edge theEdge = edges[ shell[j] ];
        if ( theEdge[0] == i )
            adjacentVertices.push_back ( theEdge[1] );
        else
            adjacentVertices.push_back ( theEdge[0] );
    }

    return adjacentVertices;
}


/// Returns the set of element indices adjacent to a given element (i.e. sharing a link)
const vector<BaseMeshTopology::Index> BaseMeshTopology::getElementAroundElement(Index)
{
    return InvalidSet;
}


/// Returns the set of element indices adjacent to a given list of elements (i.e. sharing a link)
const vector<BaseMeshTopology::Index> BaseMeshTopology::getElementAroundElements(vector<Index>)
{
    return InvalidSet;
}

/// Returns the set of element indices connected to an input one (i.e. which can be reached by topological links)
const vector<BaseMeshTopology::Index> BaseMeshTopology::getConnectedElement(Index)
{
    return InvalidSet;
}


/// Returns the set of triangles on the border of the triangulation
const sofa::type::vector<BaseMeshTopology::TriangleID>& BaseMeshTopology::getTrianglesOnBorder()
{
    msg_error() << "getTrianglesOnBorder unsupported.";
    return InvalidSet;
}

/// Returns the set of edges on the border of the triangulation
const sofa::type::vector<BaseMeshTopology::EdgeID>& BaseMeshTopology::getEdgesOnBorder()
{
    msg_error() << "getEdgesOnBorder unsupported.";
    return InvalidSet;
}

/// Returns the set of points on the border of the triangulation
const sofa::type::vector<BaseMeshTopology::PointID>& BaseMeshTopology::getPointsOnBorder()
{
    msg_error() << "getPointsOnBorder unsupported.";
    return InvalidSet;
}


void BaseMeshTopology::init()
{
    if (!fileTopology.getValue().empty())
    {
        this->load(fileTopology.getFullPath().c_str());
    }
}

class DefaultMeshTopologyLoader : public helper::io::MeshTopologyLoader
{
public:
    BaseMeshTopology* dest;
    DefaultMeshTopologyLoader(BaseMeshTopology* dest) : dest(dest) {}
    void addPoint(SReal px, SReal py, SReal pz) override
    {
        dest->addPoint(px,py,pz);
    }
    void addLine(Index p1, Index p2) override
    {
        dest->addEdge(p1,p2);
    }
    void addTriangle(Index p1, Index p2, Index p3) override
    {
        dest->addTriangle(p1,p2,p3);
    }
    void addQuad(Index p1, Index p2, Index p3, Index p4) override
    {
        dest->addQuad(p1,p2,p3,p4);
    }
    void addTetra(Index p1, Index p2, Index p3, Index p4) override
    {
        dest->addTetra(p1,p2,p3,p4);
    }
    void addCube(Index p1, Index p2, Index p3, Index p4, Index p5, Index p6, Index p7, Index p8) override
    {
        dest->addHexa(p1,p2,p3,p4,p5,p6,p7,p8);
    }
};

bool BaseMeshTopology::load(const char* filename)
{
    clear();
    std::string meshFilename(filename);
    if (!sofa::helper::system::DataRepository.findFile (meshFilename))
    {
        msg_error() << "Mesh \"" << filename << "\" not found";
        return false;
    }
    this->fileTopology.setValue( filename );
    DefaultMeshTopologyLoader loader(this);
    if (!loader.load(meshFilename.c_str()))
    {
        msg_error() << "Unable to load Mesh \"" << filename << "\"";
        return false;
    }
    return true;
}

// for procedural creation
void BaseMeshTopology::clear()
{
    msg_error() << "clear() not supported.";
}

void BaseMeshTopology::addPoint(SReal, SReal, SReal)
{
    msg_error() << "addPoint() not supported.";
}

void BaseMeshTopology::addEdge(Index, Index)
{
    msg_error() << "addEdge() not supported.";
}

void BaseMeshTopology::addTriangle(Index, Index, Index)
{
    msg_error() << "addTriangle() not supported.";
}

void BaseMeshTopology::addQuad(Index, Index, Index, Index)
{
    msg_error() << "addQuad() not supported.";
}

void BaseMeshTopology::addTetra(Index, Index, Index, Index)
{
    msg_error() << "addTetra() not supported.";
}

void BaseMeshTopology::addHexa(Index, Index, Index, Index, Index, Index, Index, Index)
{
    msg_error() << "addHexa() not supported.";
}

void BaseMeshTopology::reOrientateTriangle(TriangleID /*id*/)
{
    msg_error() << "reOrientateTriangle() not supported.";
}

std::list<const TopologyChange *>::const_iterator BaseMeshTopology::beginChange() const
{
    msg_error() << "beginChange() not supported.";
    const std::list<const TopologyChange *>::const_iterator l;
    return l;
}

std::list<const TopologyChange *>::const_iterator BaseMeshTopology::endChange() const
{
    msg_error() << "endChange() not supported.";
    const std::list<const TopologyChange *>::const_iterator l;
    return l;
}

std::list<const TopologyChange *>::const_iterator BaseMeshTopology::beginStateChange() const
{
    msg_error() << "beginStateChange() not supported.";
    const std::list<const TopologyChange *>::const_iterator l;
    return l;
}

std::list<const TopologyChange *>::const_iterator BaseMeshTopology::endStateChange() const
{
    msg_error() << "endStateChange() not supported.";
    const std::list<const TopologyChange *>::const_iterator l;
    return l;
}


Topology::EdgeID BaseMeshTopology::getEdgeIndex(PointID, PointID)
{
    msg_error() << "getEdgeIndex() not supported.";
    return InvalidID;
}

Topology::TriangleID BaseMeshTopology::getTriangleIndex(PointID, PointID, PointID)
{
    msg_error() << "getTriangleIndex() not supported.";
    return InvalidID;
}

Topology::QuadID BaseMeshTopology::getQuadIndex(PointID, PointID, PointID, PointID)
{
    msg_error() << "getQuadIndex() not supported.";
    return InvalidID;
}

Topology::TetrahedronID BaseMeshTopology::getTetrahedronIndex(PointID, PointID, PointID, PointID)
{
    msg_error() << "getTetrahedronIndex() not supported.";
    return InvalidID;
}

Topology::HexahedronID BaseMeshTopology::getHexahedronIndex(PointID, PointID, PointID, PointID, PointID, PointID, PointID, PointID)
{
    msg_error() << "getHexahedronIndex() not supported.";
    return InvalidID;
}

int BaseMeshTopology::getVertexIndexInTriangle(const Triangle &, PointID) const
{
    msg_error() << "getVertexIndexInTriangle() not supported.";
    return 0;
}

int BaseMeshTopology::getEdgeIndexInTriangle(const EdgesInTriangle &, EdgeID) const
{
    msg_error() << "getEdgeIndexInTriangle() not supported.";
    return 0;
}

int BaseMeshTopology::getVertexIndexInQuad(const Quad &, PointID) const
{
    msg_error() << "getVertexIndexInQuad() not supported.";
    return 0;
}

int BaseMeshTopology::getEdgeIndexInQuad(const EdgesInQuad &, EdgeID) const
{
    msg_error() << "getEdgeIndexInQuad() not supported.";
    return 0;
}

int BaseMeshTopology::getVertexIndexInTetrahedron(const Tetra &, PointID) const
{
    msg_error() << "getVertexIndexInTetrahedron() not supported.";
    return 0;
}

int BaseMeshTopology::getEdgeIndexInTetrahedron(const EdgesInTetrahedron &, EdgeID) const
{
    msg_error() << "getEdgeIndexInTetrahedron() not supported.";
    return 0;
}

int BaseMeshTopology::getTriangleIndexInTetrahedron(const TrianglesInTetrahedron &, TriangleID) const
{
    msg_error() << "getTriangleIndexInTetrahedron() not supported.";
    return 0;
}

int BaseMeshTopology::getVertexIndexInHexahedron(const Hexa &, PointID) const
{
    msg_error() << "getVertexIndexInHexahedron() not supported.";
    return 0;
}

int BaseMeshTopology::getEdgeIndexInHexahedron(const EdgesInHexahedron &, EdgeID) const
{
    msg_error() << "getEdgeIndexInHexahedron() not supported.";
    return 0;
}

int BaseMeshTopology::getQuadIndexInHexahedron(const QuadsInHexahedron &, QuadID) const
{
    msg_error() << "getQuadIndexInHexahedron() not supported.";
    return 0;
}

BaseMeshTopology::Edge BaseMeshTopology::getLocalEdgesInTetrahedron (const PointID) const
{
    msg_error() << "getLocalEdgesInTetrahedron() not supported.";
    return InvalidEdge;
}

BaseMeshTopology::Triangle BaseMeshTopology::getLocalTrianglesInTetrahedron (const PointID) const
{
    msg_error() << "getLocalTrianglesInTetrahedron() not supported.";
    return InvalidTriangle;
}

BaseMeshTopology::Edge BaseMeshTopology::getLocalEdgesInHexahedron (const PointID) const
{
    msg_error() << "getLocalEdgesInHexahedron() not supported.";
    return InvalidEdge;
}

BaseMeshTopology::Quad BaseMeshTopology::getLocalQuadsInHexahedron (const PointID)  const
{
    msg_error() << "getLocalQuadsInHexahedron() not supported.";
    return InvalidQuad;
}

bool BaseMeshTopology::insertInNode( objectmodel::BaseNode* node )
{
    node->addMeshTopology(this);
    Inherit1::insertInNode(node);
    return true;
}

bool BaseMeshTopology::removeInNode( objectmodel::BaseNode* node )
{
    node->removeMeshTopology(this);    
    Inherit1::removeInNode(node);
    return true;
}
} // namespace sofa::core::topology
