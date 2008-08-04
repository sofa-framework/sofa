/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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

/// Returns the set of edges adjacent to a given vertex.
const BaseMeshTopology::VertexEdges& BaseMeshTopology::getEdgeVertexShell(PointID)
{
    if (getNbEdges()) std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgeVertexShell unsupported."<<std::endl;
    static VertexEdges empty;
    return empty;
}

/// Returns the set of edges adjacent to a given triangle.
const BaseMeshTopology::TriangleEdges& BaseMeshTopology::getEdgeTriangleShell(TriangleID)
{
    if (getNbEdges()) std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgeTriangleShell unsupported."<<std::endl;
    static TriangleEdges empty;
    empty.assign(InvalidID);
    return empty;
}

/// Returns the set of edges adjacent to a given quad.
const BaseMeshTopology::QuadEdges& BaseMeshTopology::getEdgeQuadShell(QuadID)
{
    if (getNbEdges()) std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgeQuadShell unsupported."<<std::endl;
    static QuadEdges empty;
    empty.assign(InvalidID);
    return empty;
}

/// Returns the set of edges adjacent to a given tetrahedron.
const BaseMeshTopology::TetraEdges& BaseMeshTopology::getEdgeTetraShell(TetraID)
{
    if (getNbEdges()) std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgeTetraShell unsupported."<<std::endl;
    static TetraEdges empty;
    empty.assign(InvalidID);
    return empty;
}


/// Returns the set of edges adjacent to a given hexahedron.
const BaseMeshTopology::HexaEdges& BaseMeshTopology::getEdgeHexaShell(HexaID)
{
    if (getNbEdges()) std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgeHexaShell unsupported."<<std::endl;
    static HexaEdges empty;
    empty.assign(InvalidID);
    return empty;
}

/// Returns the set of triangle adjacent to a given vertex.
const BaseMeshTopology::VertexTriangles& BaseMeshTopology::getTriangleVertexShell(PointID)
{
    if (getNbTriangles()) std::cerr << "WARNING: "<<this->getClassName()<<"::getTriangleVertexShell unsupported."<<std::endl;
    static VertexTriangles empty;
    return empty;
}

/// Returns the set of triangle adjacent to a given edge.
const BaseMeshTopology::EdgeTriangles& BaseMeshTopology::getTriangleEdgeShell(EdgeID)
{
    if (getNbTriangles()) std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgeVertexShell unsupported."<<std::endl;
    static EdgeTriangles empty;
    return empty;
}

/// Returns the set of triangle adjacent to a given tetrahedron.
const BaseMeshTopology::TetraTriangles& BaseMeshTopology::getTriangleTetraShell(TetraID)
{
    if (getNbTriangles()) std::cerr << "WARNING: "<<this->getClassName()<<"::getTriangleTetraShell unsupported."<<std::endl;
    static TetraTriangles empty;
    empty.assign(InvalidID);
    return empty;
}

/// Returns the set of quad adjacent to a given vertex.
const BaseMeshTopology::VertexQuads& BaseMeshTopology::getQuadVertexShell(PointID)
{
    if (getNbQuads()) std::cerr << "WARNING: "<<this->getClassName()<<"::getQuadVertexShell unsupported."<<std::endl;
    static VertexQuads empty;
    return empty;
}

/// Returns the set of quad adjacent to a given edge.
const BaseMeshTopology::EdgeQuads& BaseMeshTopology::getQuadEdgeShell(EdgeID)
{
    if (getNbQuads()) std::cerr << "WARNING: "<<this->getClassName()<<"::getQuadEdgeShell unsupported."<<std::endl;
    static EdgeQuads empty;
    return empty;
}

/// Returns the set of quad adjacent to a given hexahedron.
const BaseMeshTopology::HexaQuads& BaseMeshTopology::getQuadHexaShell(HexaID)
{
    if (getNbQuads()) std::cerr << "WARNING: "<<this->getClassName()<<"::getQuadHexaShell unsupported."<<std::endl;
    static HexaQuads empty;
    empty.assign(InvalidID);
    return empty;
}

/// Returns the set of tetrahedra adjacent to a given vertex.
const BaseMeshTopology::VertexTetras& BaseMeshTopology::getTetraVertexShell(PointID)
{
    if (getNbTetras()) std::cerr << "WARNING: "<<this->getClassName()<<"::getTetraVertexShell unsupported."<<std::endl;
    static VertexTetras empty;
    return empty;
}

/// Returns the set of tetrahedra adjacent to a given edge.
const BaseMeshTopology::EdgeTetras& BaseMeshTopology::getTetraEdgeShell(EdgeID)
{
    if (getNbTetras()) std::cerr << "WARNING: "<<this->getClassName()<<"::getTetraEdgeShell unsupported."<<std::endl;
    static EdgeTetras empty;
    return empty;
}

/// Returns the set of tetrahedra adjacent to a given triangle.
const BaseMeshTopology::TriangleTetras& BaseMeshTopology::getTetraTriangleShell(TriangleID)
{
    if (getNbTetras()) std::cerr << "WARNING: "<<this->getClassName()<<"::getTetraTriangleShell unsupported."<<std::endl;
    static TriangleTetras empty;
    return empty;
}

/// Returns the set of hexahedra adjacent to a given vertex.
const BaseMeshTopology::VertexHexas& BaseMeshTopology::getHexaVertexShell(PointID)
{
    if (getNbHexas()) std::cerr << "WARNING: "<<this->getClassName()<<"::getHexaVertexShell unsupported."<<std::endl;
    static VertexHexas empty;
    return empty;
}

/// Returns the set of hexahedra adjacent to a given edge.
const BaseMeshTopology::EdgeHexas& BaseMeshTopology::getHexaEdgeShell(EdgeID)
{
    if (getNbHexas()) std::cerr << "WARNING: "<<this->getClassName()<<"::getHexaEdgeShell unsupported."<<std::endl;
    static EdgeHexas empty;
    return empty;
}

/// Returns the set of hexahedra adjacent to a given quad.
const BaseMeshTopology::QuadHexas& BaseMeshTopology::getHexaQuadShell(QuadID)
{
    if (getNbHexas()) std::cerr << "WARNING: "<<this->getClassName()<<"::getHexaQuadShell unsupported."<<std::endl;
    static QuadHexas empty;
    return empty;
}

// for procedural creation without file loader
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
    std::cerr << "WARNING: "<<this->getClassName()<<"::firstChange() not supported." << std::endl;
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

int BaseMeshTopology::getEdgeIndexInTriangle(const TriangleEdges &, EdgeID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgeIndexInTriangle() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getVertexIndexInQuad(Quad &, PointID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getVertexIndexInQuad() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getEdgeIndexInQuad(QuadEdges &, EdgeID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgeIndexInQuad() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getVertexIndexInTetrahedron(const Tetra &, PointID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getVertexIndexInTetrahedron() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getEdgeIndexInTetrahedron(const TetraEdges &, EdgeID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgeIndexInTetrahedron() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getTriangleIndexInTetrahedron(const TetraTriangles &, TriangleID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getTriangleIndexInTetrahedron() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getVertexIndexInHexahedron(Hexa &, PointID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getVertexIndexInHexahedron() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getEdgeIndexInHexahedron(const HexaEdges &, EdgeID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getEdgeIndexInHexahedron() not supported." << std::endl;
    return 0;
}

int BaseMeshTopology::getQuadIndexInHexahedron(const HexaQuads &, QuadID) const
{
    std::cerr << "WARNING: "<<this->getClassName()<<"::getQuadIndexInHexahedron() not supported." << std::endl;
    return 0;
}

BaseMeshTopology::Edge BaseMeshTopology::getLocalTetrahedronEdges (const unsigned int) const
{
    static BaseMeshTopology::Edge empty;
    std::cerr << "WARNING: "<<this->getClassName()<<"::getLocalTetrahedronEdges() not supported." << std::endl;
    return empty;
}

} // namespace topology

} // namespace componentmodel

} // namespace core

} // namespace sofa
