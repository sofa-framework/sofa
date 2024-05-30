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
#include <sofa/component/io/mesh/config.h>

#include <sofa/core/loader/MeshLoader.h>
namespace sofa::component::io::mesh
{


/** Procedurally creates a triangular grid.
  The coordinates range from (0,0,0) to (1,1,0). They can be translated, rotated and scaled using the corresponding attributes of the parent class.

  @author François Faure, 2012
*/
class SOFA_COMPONENT_IO_MESH_API GridMeshCreator : public sofa::core::loader::MeshLoader
{
public:

    SOFA_CLASS(GridMeshCreator,sofa::core::loader::MeshLoader);
    virtual std::string type() { return "This object is procedurally created"; }
    bool canLoad() override { return true; }
    bool doLoad() override; ///< create the grid
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data< type::Vec2i > resolution;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data< int > trianglePattern;



    Data< type::Vec2i > d_resolution;  ///< Number of vertices in each direction
    Data< int > d_trianglePattern;            ///< 0: no triangles, 1: alternate triangles, 2: upward triangles, 3: downward triangles.

protected:
    GridMeshCreator();

    void doClearBuffers() override;
    ///< index of a vertex, given its integer coordinates (between 0 and resolution) in the plane.
    unsigned vert( unsigned x, unsigned y) { return x + y * d_resolution.getValue()[0]; }

    // To avoid edge redundancy, we insert the edges to a set, an then dump the set. Edge (a,b) is considered equal to (b,a), so only one of them is inserted
    std::set<Edge> uniqueEdges;                                ///< edges without redundancy
    void insertUniqueEdge(unsigned a, unsigned b);             ///< insert an edge if it is not redundant
    void insertTriangle(unsigned a, unsigned b, unsigned c);   ///< insert a triangle (no reduncy checking !) and unique edges
    void insertQuad(unsigned a, unsigned b, unsigned c, unsigned d);   ///< insert a quad (no reduncy checking !) and unique edges

};




} //namespace sofa::component::io::mesh
