/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/topology/BaseMeshTopology.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;
using helper::vector;
using helper::fixed_array;

BaseMeshTopology::BaseMeshTopology()
{
}

/// Returns the set of edges adjacent to a given vertex.
const vector<BaseMeshTopology::EdgeID> &BaseMeshTopology::getEdgeVertexShell(PointID)
{
    static vector<EdgeID> empty;
    return empty;
}

/// Returns the set of edges adjacent to a given triangle.
const vector<BaseMeshTopology::EdgeID> &BaseMeshTopology::getEdgeTriangleShell(TriangleID)
{
    static vector<EdgeID> empty;
    return empty;
}

/// Returns the set of edges adjacent to a given quad.
const vector<BaseMeshTopology::EdgeID> &BaseMeshTopology::getEdgeQuadShell(QuadID)
{
    static vector<EdgeID> empty;
    return empty;
}

/// Returns the set of edges adjacent to a given tetrahedron.
const vector<BaseMeshTopology::EdgeID> &BaseMeshTopology::getEdgeTetraShell(TetraID)
{
    static vector<EdgeID> empty;
    return empty;
}

/// Returns the set of edges adjacent to a given hexahedron.
const vector<BaseMeshTopology::EdgeID> &BaseMeshTopology::getEdgeHexaShell(HexaID)
{
    static vector<EdgeID> empty;
    return empty;
}

/// Returns the set of triangle adjacent to a given vertex.
const vector<BaseMeshTopology::TriangleID> &BaseMeshTopology::getTriangleVertexShell(PointID)
{
    static vector<EdgeID> empty;
    return empty;
}

/// Returns the set of triangle adjacent to a given edge.
const vector<BaseMeshTopology::TriangleID> &BaseMeshTopology::getTriangleEdgeShell(EdgeID)
{
    static vector<EdgeID> empty;
    return empty;
}

/// Returns the set of triangle adjacent to a given tetrahedron.
const vector<BaseMeshTopology::TriangleID> &BaseMeshTopology::getTriangleTetraShell(TetraID)
{
    static vector<EdgeID> empty;
    return empty;
}

/// Returns the set of quad adjacent to a given vertex.
const vector<BaseMeshTopology::QuadID> &BaseMeshTopology::getQuadVertexShell(PointID)
{
    static vector<EdgeID> empty;
    return empty;
}

/// Returns the set of quad adjacent to a given edge.
const vector<BaseMeshTopology::QuadID> &BaseMeshTopology::getQuadEdgeShell(EdgeID)
{
    static vector<EdgeID> empty;
    return empty;
}

/// Returns the set of quad adjacent to a given hexahedron.
const vector<BaseMeshTopology::QuadID> &BaseMeshTopology::getQuadHexaShell(HexaID)
{
    static vector<EdgeID> empty;
    return empty;
}

/// Returns the set of tetrahedra adjacent to a given vertex.
const vector<BaseMeshTopology::TetraID> &BaseMeshTopology::getTetraVertexShell(PointID)
{
    static vector<EdgeID> empty;
    return empty;
}

/// Returns the set of tetrahedra adjacent to a given edge.
const vector<BaseMeshTopology::TetraID> &BaseMeshTopology::getTetraEdgeShell(EdgeID)
{
    static vector<EdgeID> empty;
    return empty;
}

/// Returns the set of tetrahedra adjacent to a given triangle.
const vector<BaseMeshTopology::TetraID> &BaseMeshTopology::getTetraTriangleShell(TriangleID)
{
    static vector<EdgeID> empty;
    return empty;
}

/// Returns the set of hexahedra adjacent to a given vertex.
const vector<BaseMeshTopology::HexaID> &BaseMeshTopology::getHexaVertexShell(PointID)
{
    static vector<EdgeID> empty;
    return empty;
}

/// Returns the set of hexahedra adjacent to a given edge.
const vector<BaseMeshTopology::HexaID> &BaseMeshTopology::getHexaEdgeShell(EdgeID)
{
    static vector<EdgeID> empty;
    return empty;
}

/// Returns the set of hexahedra adjacent to a given quad.
const vector<BaseMeshTopology::HexaID> &BaseMeshTopology::getHexaQuadShell(QuadID)
{
    static vector<EdgeID> empty;
    return empty;
}

// for procedural creation without file loader
void BaseMeshTopology::addPoint(double, double, double)
{
    std::cerr << "BaseMeshTopology::addPoint not supported." << std::endl;
}

void BaseMeshTopology::addEdge(int, int)
{
    std::cerr << "BaseMeshTopology::addEdge not supported." << std::endl;
}

void BaseMeshTopology::addTriangle(int, int, int)
{
    std::cerr << "BaseMeshTopology::addTriangle not supported." << std::endl;
}

void BaseMeshTopology::addQuad(int, int, int, int)
{
    std::cerr << "BaseMeshTopology::addQuad not supported." << std::endl;
}

void BaseMeshTopology::addTetra(int, int, int, int)
{
    std::cerr << "BaseMeshTopology::addTetra not supported." << std::endl;
}

void BaseMeshTopology::addHexa(int, int, int, int, int, int, int, int)
{
    std::cerr << "BaseMeshTopology::addHexa not supported." << std::endl;
}

} // namespace topology

} // namespace component

} // namespace sofa
