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
#ifndef SOFA_HELPER_IO_MESH_H
#define SOFA_HELPER_IO_MESH_H

#include <sofa/type/fixed_array.h>
#include <sofa/type/vector.h>
#include <sofa/type/Vec.h>
#include <sofa/helper/Factory.h>
#include <sofa/type/Material.h>
#include <sofa/type/PrimitiveGroup.h>
#include <sofa/topology/Topology.h>


namespace sofa::helper::io
{
    using namespace sofa::topology;

class SOFA_HELPER_API Mesh
{
public:

    std::string loaderType;

    typedef sofa::type::PrimitiveGroup PrimitiveGroup;
    typedef sofa::type::Material Material;
    typedef sofa::Index PointID;

    /* specify for each control point lying on an edge : the control point index, the index of the  edge,
    the 2 integers specifying the position within this edge (i.e. 11 for a quadratic edge, 13 within a quartic edge).. */
    typedef sofa::type::fixed_array<PointID, 4> HighOrderEdgePosition;
    /* specify for each control point lying on a triangle  : the control point index, the index of the  triangle,
    the 3 integers specifying the position within this triangle (i.e. 111 for a cubic triangle , 121 within a quartic triangle).. */
    typedef sofa::type::fixed_array<PointID, 5> HighOrderTrianglePosition;
    /* specify for each control point lying on a Quad  : the control point index, the index of the  quad,
    the 2 integers specifying the degree of the element in the x and y directions, the 2 integers specifying the position within this quad (i.e. 12 for a cubic triangle ).. */
    typedef sofa::type::fixed_array<PointID, 6> HighOrderQuadPosition;
    /* specify for each control point lying on a tetrahedron  : the control point index, the index of the  tetrahedron,
    the 3 integers specifying the position within this tetrahedron (i.e. 1111 for a quartic tetrahedron , 1211 within a quintic tetrahedron).. */
    typedef sofa::type::fixed_array<PointID, 6> HighOrderTetrahedronPosition;
    /* specify for each control point lying on a Hexahedron  : the control point index, the index of the  Hexahedron,
    the 3 integers specifying the degree of the element in the x, y and z directions, the 3 integers specifying the position within this hexahedron (i.e. 121  ).. */
    typedef sofa::type::fixed_array<PointID, 8> HighOrderHexahedronPosition;

    sofa::type::vector<type::Vec3> & getVertices() { return m_vertices; }
    const sofa::type::vector<type::Vec3> & getVertices() const { return m_vertices; }

    sofa::type::vector< Edge > & getEdges() { return m_edges; }
    const sofa::type::vector< Edge > & getEdges() const { return m_edges; }

    sofa::type::vector< Triangle > & getTriangles() { return m_triangles; }
    const sofa::type::vector< Triangle > & getTriangles() const { return m_triangles; }

    sofa::type::vector< Quad > & getQuads() { return m_quads; }
    const sofa::type::vector< Quad > & getQuads() const { return m_quads; }

    sofa::type::vector< Tetrahedron > & getTetrahedra() { return m_tetrahedra; }
    const sofa::type::vector< Tetrahedron > & getTetrahedra() const { return m_tetrahedra; }

    sofa::type::vector< Hexahedron > & getHexahedra() { return m_hexahedra; }
    const sofa::type::vector< Hexahedron > & getHexahedra() const { return m_hexahedra; }

    sofa::type::vector<type::Vec3> & getTexCoords() { return texCoords; }
    const sofa::type::vector<type::Vec3> & getTexCoords() const { return texCoords; }
    sofa::type::vector<type::Vec3> & getNormals() { return normals; }
    const sofa::type::vector<type::Vec3> & getNormals() const { return normals; }
    sofa::type::vector< type::vector < type::vector <PointID> > > & getFacets() { return facets; }
    const sofa::type::vector< type::vector < type::vector <PointID> > > & getFacets() const { return facets; }


    const sofa::type::vector< PrimitiveGroup > & getEdgesGroups() const { return m_edgesGroups; }
    const sofa::type::vector< PrimitiveGroup > & getTrianglesGroups() const { return m_trianglesGroups; }
    const sofa::type::vector< PrimitiveGroup > & getQuadsGroups() const { return m_quadsGroups; }
    const sofa::type::vector< PrimitiveGroup > & getPolygonsGroups() const { return m_polygonsGroups; }
    const sofa::type::vector< PrimitiveGroup > & getTetrahedraGroups() const { return m_tetrahedraGroups; }
    const sofa::type::vector< PrimitiveGroup > & getHexahedraGroups() const { return m_hexahedraGroups; }
    const sofa::type::vector< PrimitiveGroup > & getPrismsGroups() const { return m_prismsGroups; }
    const sofa::type::vector< PrimitiveGroup > & getPyramidsGroups() const { return m_pyramidsGroups; }

    const sofa::type::vector< HighOrderEdgePosition >& getHighOrderEdgePositions() const { return m_highOrderEdgePositions; }
    const sofa::type::vector< HighOrderTrianglePosition >& getHighOrderTrianglePositions() const { return m_highOrderTrianglePositions; }
    const sofa::type::vector< HighOrderQuadPosition >& getHighOrderQuadPositions() const { return m_highOrderQuadPositions; }


    const Material& getMaterial() const { return material; }

    const std::vector<Material>& getMaterials() { return materials; }
    const std::vector<PrimitiveGroup>& getGroups() { return groups; }

    std::string& getTextureName() { return textureName; }

    typedef Factory<std::string, Mesh, std::string> FactoryMesh;

    static Mesh* Create(const std::string &filename);
    static Mesh* Create(const std::string& loader, const std::string& filename);

    template<class Object>
    static Object* create(Object*, std::string arg)
    {
        return new Object(arg);
    }

protected:
    // Point coordinates in 3D.
    sofa::type::vector<type::Vec3> m_vertices;

    // Tab of 2D elements composition
    sofa::type::vector< Edge > m_edges; ///< Edges of the mesh loaded
    sofa::type::vector< Triangle > m_triangles; ///< Triangles of the mesh loaded
    sofa::type::vector< Quad > m_quads; ///< Quads of the mesh loaded
    type::vector< type::vector <sofa::Index> > m_polygons; ///< Polygons of the mesh loaded
    type::vector< HighOrderEdgePosition > m_highOrderEdgePositions; ///< High order edge points of the mesh loaded
    type::vector< HighOrderTrianglePosition > m_highOrderTrianglePositions; ///< High order triangle points of the mesh loaded
    type::vector< HighOrderQuadPosition > m_highOrderQuadPositions; ///< High order quad points of the mesh loaded

    // Tab of 3D elements composition
    sofa::type::vector< Tetrahedron > m_tetrahedra; ///< Tetrahedra of the mesh loaded
    sofa::type::vector< Hexahedron > m_hexahedra; ///< Hexahedra of the mesh loaded

    // Groups
    type::vector< PrimitiveGroup > m_edgesGroups; ///< Groups of Edges
    type::vector< PrimitiveGroup > m_trianglesGroups; ///< Groups of Triangles
    type::vector< PrimitiveGroup > m_quadsGroups; ///< Groups of Quads
    type::vector< PrimitiveGroup > m_polygonsGroups; ///< Groups of Polygons
    type::vector< PrimitiveGroup > m_tetrahedraGroups; ///< Groups of Tetrahedra
    type::vector< PrimitiveGroup > m_hexahedraGroups; ///< Groups of Hexahedra
    type::vector< PrimitiveGroup > m_prismsGroups; ///< Groups of Prisms
    type::vector< PrimitiveGroup > m_pyramidsGroups; ///< Groups of Pyramids


    sofa::type::vector<type::Vec3> texCoords; // for the moment, we suppose that texCoords is order 2 (2 texCoords for a vertex)
    sofa::type::vector<type::Vec3> normals;
    sofa::type::vector< sofa::type::vector< sofa::type::vector<PointID> > > facets;
    Material material;

    std::vector<Material> materials;
    std::vector<PrimitiveGroup> groups;

    std::string textureName;

};

} // namespace sofa::helper::io


#endif
