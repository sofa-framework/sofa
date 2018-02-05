/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/io/MeshTopologyLoader.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/objectmodel/BaseNode.h>

namespace sofa
{

namespace core
{

namespace topology
{

using namespace sofa::defaulttype;
using helper::vector;
using helper::fixed_array;

BaseMeshTopology::BaseMeshTopology()
    : fileTopology(initData(&fileTopology,"fileTopology","Filename of the mesh"))
{
    addAlias(&fileTopology,"filename");
}

/// Returns the set of edges adjacent to a given vertex.
const BaseMeshTopology::EdgesAroundVertex& BaseMeshTopology::getEdgesAroundVertex(PointID)
{
    if (getNbEdges()) serr<<"getEdgesAroundVertex unsupported."<<sendl;
    static EdgesAroundVertex empty;
    return empty;
}

/// Returns the set of edges adjacent to a given triangle.
const BaseMeshTopology::EdgesInTriangle& BaseMeshTopology::getEdgesInTriangle(TriangleID)
{
    if (getNbEdges()) serr<<"getEdgesInTriangle unsupported."<<sendl;
    static EdgesInTriangle empty;
    empty.assign(InvalidID);
    return empty;
}

/// Returns the set of edges adjacent to a given quad.
const BaseMeshTopology::EdgesInQuad& BaseMeshTopology::getEdgesInQuad(QuadID)
{
    if (getNbEdges()) serr<<"getEdgesInQuad unsupported."<<sendl;
    static EdgesInQuad empty;
    empty.assign(InvalidID);
    return empty;
}

/// Returns the set of edges adjacent to a given tetrahedron.
const BaseMeshTopology::EdgesInTetrahedron& BaseMeshTopology::getEdgesInTetrahedron(TetraID)
{
    if (getNbEdges()) serr<<"getEdgesInTetrahedron unsupported."<<sendl;
    static EdgesInTetrahedron empty;
    empty.assign(InvalidID);
    return empty;
}


/// Returns the set of edges adjacent to a given hexahedron.
const BaseMeshTopology::EdgesInHexahedron& BaseMeshTopology::getEdgesInHexahedron(HexaID)
{
    if (getNbEdges()) serr<<"getEdgesInHexahedron unsupported."<<sendl;
    static EdgesInHexahedron empty;
    empty.assign(InvalidID);
    return empty;
}

/// Returns the set of triangle adjacent to a given vertex.
const BaseMeshTopology::TrianglesAroundVertex& BaseMeshTopology::getTrianglesAroundVertex(PointID)
{
    if (getNbTriangles()) serr<<"getTrianglesAroundVertex unsupported."<<sendl;
    static TrianglesAroundVertex empty;
    return empty;
}

/// Returns the set of triangle adjacent to a given edge.
const BaseMeshTopology::TrianglesAroundEdge& BaseMeshTopology::getTrianglesAroundEdge(EdgeID)
{
    if (getNbTriangles()) serr<<"getEdgesAroundVertex unsupported."<<sendl;
    static TrianglesAroundEdge empty;
    return empty;
}

/// Returns the set of triangle adjacent to a given tetrahedron.
const BaseMeshTopology::TrianglesInTetrahedron& BaseMeshTopology::getTrianglesInTetrahedron(TetraID)
{
    if (getNbTriangles()) serr<<"getTrianglesInTetrahedron unsupported."<<sendl;
    static TrianglesInTetrahedron empty;
    empty.assign(InvalidID);
    return empty;
}

/// Returns the set of quad adjacent to a given vertex.
const BaseMeshTopology::QuadsAroundVertex& BaseMeshTopology::getQuadsAroundVertex(PointID)
{
    if (getNbQuads()) serr<<"getQuadsAroundVertex unsupported."<<sendl;
    static QuadsAroundVertex empty;
    return empty;
}

/// Returns the set of quad adjacent to a given edge.
const BaseMeshTopology::QuadsAroundEdge& BaseMeshTopology::getQuadsAroundEdge(EdgeID)
{
    if (getNbQuads()) serr<<"getQuadsAroundEdge unsupported."<<sendl;
    static QuadsAroundEdge empty;
    return empty;
}

/// Returns the set of quad adjacent to a given hexahedron.
const BaseMeshTopology::QuadsInHexahedron& BaseMeshTopology::getQuadsInHexahedron(HexaID)
{
    if (getNbQuads()) serr<<"getQuadsInHexahedron unsupported."<<sendl;
    static QuadsInHexahedron empty;
    empty.assign(InvalidID);
    return empty;
}

/// Returns the set of tetrahedra adjacent to a given vertex.
const BaseMeshTopology::TetrahedraAroundVertex& BaseMeshTopology::getTetrahedraAroundVertex(PointID)
{
    if (getNbTetrahedra()) serr<<"getTetrahedraAroundVertex unsupported."<<sendl;
    static TetrahedraAroundVertex empty;
    return empty;
}

/// Returns the set of tetrahedra adjacent to a given edge.
const BaseMeshTopology::TetrahedraAroundEdge& BaseMeshTopology::getTetrahedraAroundEdge(EdgeID)
{
    if (getNbTetrahedra()) serr<<"getTetrahedraAroundEdge unsupported."<<sendl;
    static TetrahedraAroundEdge empty;
    return empty;
}

/// Returns the set of tetrahedra adjacent to a given triangle.
const BaseMeshTopology::TetrahedraAroundTriangle& BaseMeshTopology::getTetrahedraAroundTriangle(TriangleID)
{
    if (getNbTetrahedra()) serr<<"getTetrahedraAroundTriangle unsupported."<<sendl;
    static TetrahedraAroundTriangle empty;
    return empty;
}

/// Returns the set of hexahedra adjacent to a given vertex.
const BaseMeshTopology::HexahedraAroundVertex& BaseMeshTopology::getHexahedraAroundVertex(PointID)
{
    if (getNbHexahedra()) serr<<"getHexahedraAroundVertex unsupported."<<sendl;
    static HexahedraAroundVertex empty;
    return empty;
}

/// Returns the set of hexahedra adjacent to a given edge.
const BaseMeshTopology::HexahedraAroundEdge& BaseMeshTopology::getHexahedraAroundEdge(EdgeID)
{
    if (getNbHexahedra()) serr<<"getHexahedraAroundEdge unsupported."<<sendl;
    static HexahedraAroundEdge empty;
    return empty;
}

/// Returns the set of hexahedra adjacent to a given quad.
const BaseMeshTopology::HexahedraAroundQuad& BaseMeshTopology::getHexahedraAroundQuad(QuadID)
{
    if (getNbHexahedra()) serr<<"getHexahedraAroundQuad unsupported."<<sendl;
    static HexahedraAroundQuad empty;
    return empty;
}


/// Returns the set of vertices adjacent to a given vertex (i.e. sharing an edge)
const BaseMeshTopology::VerticesAroundVertex BaseMeshTopology::getVerticesAroundVertex(PointID i)
{
    const SeqEdges& edges = getEdges();
    const EdgesAroundVertex& shell = getEdgesAroundVertex(i);
    VerticesAroundVertex adjacentVertices;

    for (std::size_t j = 0; j<shell.size(); j++)
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
const vector<BaseMeshTopology::index_type> BaseMeshTopology::getElementAroundElement(index_type)
{
    static vector<index_type> empty;
    return empty;
}


/// Returns the set of element indices adjacent to a given list of elements (i.e. sharing a link)
const vector<BaseMeshTopology::index_type> BaseMeshTopology::getElementAroundElements(vector<index_type>)
{
    static vector<index_type> empty;
    return empty;
}

/// Returns the set of element indices connected to an input one (i.e. which can be reached by topological links)
const vector<BaseMeshTopology::index_type> BaseMeshTopology::getConnectedElement(index_type)
{
    static vector<index_type> empty;
    return empty;
}


/// Returns the set of triangles on the border of the triangulation
const sofa::helper::vector <BaseMeshTopology::TriangleID>& BaseMeshTopology::getTrianglesOnBorder()
{
    serr<<"getTrianglesOnBorder unsupported."<<sendl;
    static sofa::helper::vector <BaseMeshTopology::TriangleID> empty;
    return empty;
}

/// Returns the set of edges on the border of the triangulation
const sofa::helper::vector <BaseMeshTopology::EdgeID>& BaseMeshTopology::getEdgesOnBorder()
{
    serr<<"getEdgesOnBorder unsupported."<<sendl;
    static sofa::helper::vector <BaseMeshTopology::EdgeID> empty;
    return empty;
}

/// Returns the set of points on the border of the triangulation
const sofa::helper::vector <BaseMeshTopology::PointID>& BaseMeshTopology::getPointsOnBorder()
{
    serr<<"getPointsOnBorder unsupported."<<sendl;
    static sofa::helper::vector <BaseMeshTopology::PointID> empty;
    return empty;
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
    virtual void addPoint(SReal px, SReal py, SReal pz)
    {
        dest->addPoint(px,py,pz);
    }
    virtual void addLine(int p1, int p2)
    {
        dest->addEdge(p1,p2);
    }
    virtual void addTriangle(int p1, int p2, int p3)
    {
        dest->addTriangle(p1,p2,p3);
    }
    virtual void addQuad(int p1, int p2, int p3, int p4)
    {
        dest->addQuad(p1,p2,p3,p4);
    }
    virtual void addTetra(int p1, int p2, int p3, int p4)
    {
        dest->addTetra(p1,p2,p3,p4);
    }
    virtual void addCube(int p1, int p2, int p3, int p4, int p5, int p6, int p7, int p8)
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
        serr << "Mesh \""<< filename <<"\" not found"<< sendl;
        return false;
    }
    this->fileTopology.setValue( filename );
    DefaultMeshTopologyLoader loader(this);
    if (!loader.load(meshFilename.c_str()))
    {
        serr << "Unable to load Mesh \""<<filename << "\"" << sendl;
        return false;
    }
    return true;
}

// for procedural creation
void BaseMeshTopology::clear()
{
    serr<<"clear() not supported." << sendl;
}

void BaseMeshTopology::addPoint(SReal, SReal, SReal)
{
    serr<<"addPoint() not supported." << sendl;
}

void BaseMeshTopology::addEdge(int, int)
{
    serr<<"addEdge() not supported." << sendl;
}

void BaseMeshTopology::addTriangle(int, int, int)
{
    serr<<"addTriangle() not supported." << sendl;
}

void BaseMeshTopology::addQuad(int, int, int, int)
{
    serr<<"addQuad() not supported." << sendl;
}

void BaseMeshTopology::addTetra(int, int, int, int)
{
    serr<<"addTetra() not supported." << sendl;
}

void BaseMeshTopology::addHexa(int, int, int, int, int, int, int, int)
{
    serr<<"addHexa() not supported." << sendl;
}

void BaseMeshTopology::reOrientateTriangle(TriangleID /*id*/)
{
    serr<<"reOrientateTriangle() not supported." << sendl;
}

std::list<const TopologyChange *>::const_iterator BaseMeshTopology::beginChange() const
{
    serr << "beginChange() not supported." << sendl;
    std::list<const TopologyChange *>::const_iterator l;
    return l;
}

std::list<const TopologyChange *>::const_iterator BaseMeshTopology::endChange() const
{
    serr<<"endChange() not supported." << sendl;
    std::list<const TopologyChange *>::const_iterator l;
    return l;
}

std::list<const TopologyChange *>::const_iterator BaseMeshTopology::beginStateChange() const
{
    serr<<"beginStateChange() not supported." << sendl;
    std::list<const TopologyChange *>::const_iterator l;
    return l;
}

std::list<const TopologyChange *>::const_iterator BaseMeshTopology::endStateChange() const
{
    serr<<"endStateChange() not supported." << sendl;
    std::list<const TopologyChange *>::const_iterator l;
    return l;
}


std::list<TopologyEngine *>::const_iterator BaseMeshTopology::beginTopologyEngine() const
{
    serr<<"beginTopologyEngine() not supported." << sendl;
    std::list<TopologyEngine *>::const_iterator l;
    return l;
}


std::list<TopologyEngine *>::const_iterator BaseMeshTopology::endTopologyEngine() const
{
    serr<<"endTopologyEngine() not supported." << sendl;
    std::list<TopologyEngine *>::const_iterator l;
    return l;
}

void BaseMeshTopology::addTopologyEngine(TopologyEngine* _topologyEngine)
{
    serr<<"addTopologyEngine() not supported." << sendl;
    (void)_topologyEngine;
}

int BaseMeshTopology::getEdgeIndex(PointID, PointID)
{
    serr<<"getEdgeIndex() not supported." << sendl;
    return 0;
}

int BaseMeshTopology::getTriangleIndex(PointID, PointID, PointID)
{
    serr<<"getTriangleIndex() not supported." << sendl;
    return 0;
}

int BaseMeshTopology::getQuadIndex(PointID, PointID, PointID, PointID)
{
    serr<<"getQuadIndex() not supported." << sendl;
    return 0;
}

int BaseMeshTopology::getTetrahedronIndex(PointID, PointID, PointID, PointID)
{
    serr<<"getTetrahedronIndex() not supported." << sendl;
    return 0;
}

int BaseMeshTopology::getHexahedronIndex(PointID, PointID, PointID, PointID, PointID, PointID, PointID, PointID)
{
    serr<<"getHexahedronIndex() not supported." << sendl;
    return 0;
}

int BaseMeshTopology::getVertexIndexInTriangle(const Triangle &, PointID) const
{
    serr<<"getVertexIndexInTriangle() not supported." << sendl;
    return 0;
}

int BaseMeshTopology::getEdgeIndexInTriangle(const EdgesInTriangle &, EdgeID) const
{
    serr<<"getEdgeIndexInTriangle() not supported." << sendl;
    return 0;
}

int BaseMeshTopology::getVertexIndexInQuad(const Quad &, PointID) const
{
    serr<<"getVertexIndexInQuad() not supported." << sendl;
    return 0;
}

int BaseMeshTopology::getEdgeIndexInQuad(const EdgesInQuad &, EdgeID) const
{
    serr<<"getEdgeIndexInQuad() not supported." << sendl;
    return 0;
}

int BaseMeshTopology::getVertexIndexInTetrahedron(const Tetra &, PointID) const
{
    serr<<"getVertexIndexInTetrahedron() not supported." << sendl;
    return 0;
}

int BaseMeshTopology::getEdgeIndexInTetrahedron(const EdgesInTetrahedron &, EdgeID) const
{
    serr<<"getEdgeIndexInTetrahedron() not supported." << sendl;
    return 0;
}

int BaseMeshTopology::getTriangleIndexInTetrahedron(const TrianglesInTetrahedron &, TriangleID) const
{
    serr<<"getTriangleIndexInTetrahedron() not supported." << sendl;
    return 0;
}

int BaseMeshTopology::getVertexIndexInHexahedron(const Hexa &, PointID) const
{
    serr<<"getVertexIndexInHexahedron() not supported." << sendl;
    return 0;
}

int BaseMeshTopology::getEdgeIndexInHexahedron(const EdgesInHexahedron &, EdgeID) const
{
    serr<<"getEdgeIndexInHexahedron() not supported." << sendl;
    return 0;
}

int BaseMeshTopology::getQuadIndexInHexahedron(const QuadsInHexahedron &, QuadID) const
{
    serr<<"getQuadIndexInHexahedron() not supported." << sendl;
    return 0;
}

BaseMeshTopology::Edge BaseMeshTopology::getLocalEdgesInTetrahedron (const PointID) const
{
    static BaseMeshTopology::Edge empty;
    serr<<"getLocalEdgesInTetrahedron() not supported." << sendl;
    return empty;
}

BaseMeshTopology::Triangle BaseMeshTopology::getLocalTrianglesInTetrahedron (const PointID) const
{
    static BaseMeshTopology::Triangle empty;
    serr<<"getLocalTrianglesInTetrahedron() not supported." << sendl;
    return empty;
}

BaseMeshTopology::Edge BaseMeshTopology::getLocalEdgesInHexahedron (const PointID) const
{
    static BaseMeshTopology::Edge empty;
    serr<<"getLocalEdgesInHexahedron() not supported." << sendl;
    return empty;
}

BaseMeshTopology::Quad BaseMeshTopology::getLocalQuadsInHexahedron (const PointID)  const
{
    static BaseMeshTopology::Quad empty;
    serr<<"getLocalQuadsInHexahedron() not supported." << sendl;
    return empty;
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



} // namespace topology

} // namespace core

} // namespace sofa
