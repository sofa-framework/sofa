/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/helper/io/MeshTopologyLoader.h>
#include <sofa/helper/system/FileRepository.h>

namespace sofa
{

namespace core
{

namespace componentmodel
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
    if (getNbEdges()) std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgesAroundVertex unsupported."<<std::endl;
    static EdgesAroundVertex empty;
    return empty;
}

/// Returns the set of edges adjacent to a given triangle.
const BaseMeshTopology::EdgesInTriangle& BaseMeshTopology::getEdgesInTriangle(TriangleID)
{
    if (getNbEdges()) std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgesInTriangle unsupported."<<std::endl;
    static EdgesInTriangle empty;
    empty.assign(InvalidID);
    return empty;
}

/// Returns the set of edges adjacent to a given quad.
const BaseMeshTopology::EdgesInQuad& BaseMeshTopology::getEdgesInQuad(QuadID)
{
    if (getNbEdges()) std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgesInQuad unsupported."<<std::endl;
    static EdgesInQuad empty;
    empty.assign(InvalidID);
    return empty;
}

/// Returns the set of edges adjacent to a given tetrahedron.
const BaseMeshTopology::EdgesInTetrahedron& BaseMeshTopology::getEdgesInTetrahedron(TetraID)
{
    if (getNbEdges()) std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgesInTetrahedron unsupported."<<std::endl;
    static EdgesInTetrahedron empty;
    empty.assign(InvalidID);
    return empty;
}


/// Returns the set of edges adjacent to a given hexahedron.
const BaseMeshTopology::EdgesInHexahedron& BaseMeshTopology::getEdgesInHexahedron(HexaID)
{
    if (getNbEdges()) std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgesInHexahedron unsupported."<<std::endl;
    static EdgesInHexahedron empty;
    empty.assign(InvalidID);
    return empty;
}

/// Returns the set of triangle adjacent to a given vertex.
const BaseMeshTopology::TrianglesAroundVertex& BaseMeshTopology::getTrianglesAroundVertex(PointID)
{
    if (getNbTriangles()) std::cerr << "WARNING: "<<this->getClassName()<<"::getTrianglesAroundVertex unsupported."<<std::endl;
    static TrianglesAroundVertex empty;
    return empty;
}

/// Returns the set of triangle adjacent to a given edge.
const BaseMeshTopology::TrianglesAroundEdge& BaseMeshTopology::getTrianglesAroundEdge(EdgeID)
{
    if (getNbTriangles()) std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgesAroundVertex unsupported."<<std::endl;
    static TrianglesAroundEdge empty;
    return empty;
}

/// Returns the set of triangle adjacent to a given tetrahedron.
const BaseMeshTopology::TrianglesInTetrahedron& BaseMeshTopology::getTrianglesInTetrahedron(TetraID)
{
    if (getNbTriangles()) std::cerr << "WARNING: "<<this->getClassName()<<"::getTrianglesInTetrahedron unsupported."<<std::endl;
    static TrianglesInTetrahedron empty;
    empty.assign(InvalidID);
    return empty;
}

/// Returns the set of quad adjacent to a given vertex.
const BaseMeshTopology::QuadsAroundVertex& BaseMeshTopology::getQuadsAroundVertex(PointID)
{
    if (getNbQuads()) std::cerr << "WARNING: "<<this->getClassName()<<"::getQuadsAroundVertex unsupported."<<std::endl;
    static QuadsAroundVertex empty;
    return empty;
}

/// Returns the set of quad adjacent to a given edge.
const BaseMeshTopology::QuadsAroundEdge& BaseMeshTopology::getQuadsAroundEdge(EdgeID)
{
    if (getNbQuads()) std::cerr << "WARNING: "<<this->getClassName()<<"::getQuadsAroundEdge unsupported."<<std::endl;
    static QuadsAroundEdge empty;
    return empty;
}

/// Returns the set of quad adjacent to a given hexahedron.
const BaseMeshTopology::QuadsInHexahedron& BaseMeshTopology::getQuadsInHexahedron(HexaID)
{
    if (getNbQuads()) std::cerr << "WARNING: "<<this->getClassName()<<"::getQuadsInHexahedron unsupported."<<std::endl;
    static QuadsInHexahedron empty;
    empty.assign(InvalidID);
    return empty;
}

/// Returns the set of tetrahedra adjacent to a given vertex.
const BaseMeshTopology::TetrahedraAroundVertex& BaseMeshTopology::getTetrahedraAroundVertex(PointID)
{
    if (getNbTetrahedra()) std::cerr << "WARNING: "<<this->getClassName()<<"::getTetrahedraAroundVertex unsupported."<<std::endl;
    static TetrahedraAroundVertex empty;
    return empty;
}

/// Returns the set of tetrahedra adjacent to a given edge.
const BaseMeshTopology::TetrahedraAroundEdge& BaseMeshTopology::getTetrahedraAroundEdge(EdgeID)
{
    if (getNbTetrahedra()) std::cerr << "WARNING: "<<this->getClassName()<<"::getTetrahedraAroundEdge unsupported."<<std::endl;
    static TetrahedraAroundEdge empty;
    return empty;
}

/// Returns the set of tetrahedra adjacent to a given triangle.
const BaseMeshTopology::TetrahedraAroundTriangle& BaseMeshTopology::getTetrahedraAroundTriangle(TriangleID)
{
    if (getNbTetrahedra()) std::cerr << "WARNING: "<<this->getClassName()<<"::getTetrahedraAroundTriangle unsupported."<<std::endl;
    static TetrahedraAroundTriangle empty;
    return empty;
}

/// Returns the set of hexahedra adjacent to a given vertex.
const BaseMeshTopology::HexahedraAroundVertex& BaseMeshTopology::getHexahedraAroundVertex(PointID)
{
    if (getNbHexahedra()) std::cerr << "WARNING: "<<this->getClassName()<<"::getHexahedraAroundVertex unsupported."<<std::endl;
    static HexahedraAroundVertex empty;
    return empty;
}

/// Returns the set of hexahedra adjacent to a given edge.
const BaseMeshTopology::HexahedraAroundEdge& BaseMeshTopology::getHexahedraAroundEdge(EdgeID)
{
    if (getNbHexahedra()) std::cerr << "WARNING: "<<this->getClassName()<<"::getHexahedraAroundEdge unsupported."<<std::endl;
    static HexahedraAroundEdge empty;
    return empty;
}

/// Returns the set of hexahedra adjacent to a given quad.
const BaseMeshTopology::HexahedraAroundQuad& BaseMeshTopology::getHexahedraAroundQuad(QuadID)
{
    if (getNbHexahedra()) std::cerr << "WARNING: "<<this->getClassName()<<"::getHexahedraAroundQuad unsupported."<<std::endl;
    static HexahedraAroundQuad empty;
    return empty;
}


/// Returns the set of vertices adjacent to a given vertex (i.e. sharing an edge)
const BaseMeshTopology::VerticesAroundVertex BaseMeshTopology::getVerticesAroundVertex(PointID i)
{
    const SeqEdges& edges = getEdges();
    const EdgesAroundVertex& shell = getEdgesAroundVertex(i);
    VerticesAroundVertex adjacentVertices;

    for (unsigned int j = 0; j<shell.size(); j++)
    {
        Edge theEdge = edges[ shell[j] ];
        if ( theEdge[0] == i )
            adjacentVertices.push_back ( theEdge[1] );
        else
            adjacentVertices.push_back ( theEdge[0] );
    }

    return adjacentVertices;
}


/// Returns the set of triangles on the border of the triangulation
const sofa::helper::vector <BaseMeshTopology::TriangleID>& BaseMeshTopology::getTrianglesOnBorder()
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getTrianglesOnBorder unsupported."<<std::endl;
    static sofa::helper::vector <BaseMeshTopology::TriangleID> empty;
    return empty;
}

/// Returns the set of edges on the border of the triangulation
const sofa::helper::vector <BaseMeshTopology::EdgeID>& BaseMeshTopology::getEdgesOnBorder()
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgesOnBorder unsupported."<<std::endl;
    static sofa::helper::vector <BaseMeshTopology::EdgeID> empty;
    return empty;
}

/// Returns the set of points on the border of the triangulation
const sofa::helper::vector <BaseMeshTopology::PointID>& BaseMeshTopology::getPointsOnBorder()
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getPointsOnBorder unsupported."<<std::endl;
    static sofa::helper::vector <BaseMeshTopology::PointID> empty;
    return empty;
}


void BaseMeshTopology::init()
{
}
void BaseMeshTopology::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->core::componentmodel::topology::Topology::parse(arg);
    if (!fileTopology.getValue().empty())
    {
        this->load(fileTopology.getValue().c_str());
    }
}

class DefaultMeshTopologyLoader : public helper::io::MeshTopologyLoader
{
public:
    BaseMeshTopology* dest;
    DefaultMeshTopologyLoader(BaseMeshTopology* dest) : dest(dest) {}
    virtual void addPoint(double px, double py, double pz)
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
    std::cerr << "WARNING: "<<this->getClassName()<<"::clear() not supported." << std::endl;
}

void BaseMeshTopology::addPoint(double, double, double)
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::addPoint() not supported." << std::endl;
}

void BaseMeshTopology::addEdge(int, int)
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::addEdge() not supported." << std::endl;
}

void BaseMeshTopology::addTriangle(int, int, int)
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::addTriangle() not supported." << std::endl;
}

void BaseMeshTopology::addQuad(int, int, int, int)
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::addQuad() not supported." << std::endl;
}

void BaseMeshTopology::addTetra(int, int, int, int)
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::addTetra() not supported." << std::endl;
}

void BaseMeshTopology::addHexa(int, int, int, int, int, int, int, int)
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::addHexa() not supported." << std::endl;
}

std::list<const TopologyChange *>::const_iterator BaseMeshTopology::firstChange() const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::firstChange() not supported." ;
    std::cerr<< std::endl;
    std::list<const TopologyChange *>::const_iterator l;
    return l;
}

std::list<const TopologyChange *>::const_iterator BaseMeshTopology::lastChange() const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::lastChange() not supported." << std::endl;
    std::list<const TopologyChange *>::const_iterator l;
    return l;
}

std::list<const TopologyChange *>::const_iterator BaseMeshTopology::firstStateChange() const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::firstStateChange() not supported." << std::endl;
    std::list<const TopologyChange *>::const_iterator l;
    return l;
}

std::list<const TopologyChange *>::const_iterator BaseMeshTopology::lastStateChange() const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::lastStateChange() not supported." << std::endl;
    std::list<const TopologyChange *>::const_iterator l;
    return l;
}

int BaseMeshTopology::getEdgeIndex(PointID, PointID)
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgeIndex() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getTriangleIndex(PointID, PointID, PointID)
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getTriangleIndex() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getQuadIndex(PointID, PointID, PointID, PointID)
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getQuadIndex() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getTetrahedronIndex(PointID, PointID, PointID, PointID)
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getTetrahedronIndex() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getHexahedronIndex(PointID, PointID, PointID, PointID, PointID, PointID, PointID, PointID)
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getHexahedronIndex() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getVertexIndexInTriangle(const Triangle &, PointID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getVertexIndexInTriangle() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getEdgeIndexInTriangle(const EdgesInTriangle &, EdgeID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgeIndexInTriangle() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getVertexIndexInQuad(const Quad &, PointID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getVertexIndexInQuad() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getEdgeIndexInQuad(const EdgesInQuad &, EdgeID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgeIndexInQuad() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getVertexIndexInTetrahedron(const Tetra &, PointID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getVertexIndexInTetrahedron() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getEdgeIndexInTetrahedron(const EdgesInTetrahedron &, EdgeID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgeIndexInTetrahedron() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getTriangleIndexInTetrahedron(const TrianglesInTetrahedron &, TriangleID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getTriangleIndexInTetrahedron() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getVertexIndexInHexahedron(const Hexa &, PointID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getVertexIndexInHexahedron() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getEdgeIndexInHexahedron(const EdgesInHexahedron &, EdgeID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgeIndexInHexahedron() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getQuadIndexInHexahedron(const QuadsInHexahedron &, QuadID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getQuadIndexInHexahedron() not supported." << std::endl;
    return 0;
}

BaseMeshTopology::Edge BaseMeshTopology::getLocalEdgesInTetrahedron (const PointID) const
{
    static BaseMeshTopology::Edge empty;
    std::cerr << "WARNING: "<<this->getClassName()<<"::getLocalEdgesInTetrahedron() not supported." << std::endl;
    return empty;
}

BaseMeshTopology::Edge BaseMeshTopology::getLocalEdgesInHexahedron (const PointID) const
{
    static BaseMeshTopology::Edge empty;
    std::cerr << "WARNING: "<<this->getClassName()<<"::getLocalEdgesInHexahedron() not supported." << std::endl;
    return empty;
}


} // namespace topology

} // namespace componentmodel

} // namespace core

} // namespace sofa
