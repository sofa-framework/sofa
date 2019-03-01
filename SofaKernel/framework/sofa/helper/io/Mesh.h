/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/Factory.h>
#include <sofa/core/loader/PrimitiveGroup.h>
#include <sofa/core/loader/Material.h>
#include <sofa/core/topology/Topology.h>

namespace sofa
{

namespace helper
{

namespace io
{
    using namespace sofa::core::topology;

class SOFA_HELPER_API Mesh
{    
public:
    
    std::string loaderType;
    
public:
    typedef sofa::defaulttype::Vector3 Vector3;
    typedef sofa::core::loader::PrimitiveGroup PrimitiveGroup;
    typedef sofa::core::loader::Material Material;
    typedef Topology::PointID PointID;

    /* specify for each control point lying on an edge : the control point index, the index of the  edge,
    the 2 integers specifying the position within this edge (i.e. 11 for a quadratic edge, 13 within a quartic edge).. */
    typedef sofa::helper::fixed_array<PointID, 4> HighOrderEdgePosition;
    /* specify for each control point lying on a triangle  : the control point index, the index of the  triangle,
    the 3 integers specifying the position within this triangle (i.e. 111 for a cubic triangle , 121 within a quartic triangle).. */
    typedef sofa::helper::fixed_array<PointID, 5> HighOrderTrianglePosition;
    /* specify for each control point lying on a Quad  : the control point index, the index of the  quad,
    the 2 integers specifying the degree of the element in the x and y directions, the 2 integers specifying the position within this quad (i.e. 12 for a cubic triangle ).. */
    typedef sofa::helper::fixed_array<PointID, 6> HighOrderQuadPosition;
    /* specify for each control point lying on a tetrahedron  : the control point index, the index of the  tetrahedron,
    the 3 integers specifying the position within this tetrahedron (i.e. 1111 for a quartic tetrahedron , 1211 within a quintic tetrahedron).. */
    typedef sofa::helper::fixed_array<PointID, 6> HighOrderTetrahedronPosition;
    /* specify for each control point lying on a Hexahedron  : the control point index, the index of the  Hexahedron,
    the 3 integers specifying the degree of the element in the x, y and z directions, the 3 integers specifying the position within this hexahedron (i.e. 121  ).. */
    typedef sofa::helper::fixed_array<PointID, 8> HighOrderHexahedronPosition;

    sofa::helper::vector<Vector3> & getVertices() { return m_vertices; }
    const sofa::helper::vector<Vector3> & getVertices() const { return m_vertices; }

    sofa::helper::vector< Topology::Edge > & getEdges() { return m_edges; }
    const sofa::helper::vector< Topology::Edge > & getEdges() const { return m_edges; }

    sofa::helper::vector< Topology::Triangle > & getTriangles() { return m_triangles; }
    const sofa::helper::vector< Topology::Triangle > & getTriangles() const { return m_triangles; }

    sofa::helper::vector< Topology::Quad > & getQuads() { return m_quads; }
    const sofa::helper::vector< Topology::Quad > & getQuads() const { return m_quads; }

    sofa::helper::vector< Topology::Tetrahedron > & getTetrahedra() { return m_tetrahedra; }
    const sofa::helper::vector< Topology::Tetrahedron > & getTetrahedra() const { return m_tetrahedra; }

    sofa::helper::vector< Topology::Hexahedron > & getHexahedra() { return m_hexahedra; }
    const sofa::helper::vector< Topology::Hexahedron > & getHexahedra() const { return m_hexahedra; }

    sofa::helper::vector<Vector3> & getTexCoords() { return texCoords; }
    const sofa::helper::vector<Vector3> & getTexCoords() const { return texCoords; }
    sofa::helper::vector<Vector3> & getNormals() { return normals; }
    const sofa::helper::vector<Vector3> & getNormals() const { return normals; }
    sofa::helper::vector< vector < vector <int> > > & getFacets() { return facets; }
    const sofa::helper::vector< vector < vector <int> > > & getFacets() const { return facets; }


    const sofa::helper::vector< PrimitiveGroup > & getEdgesGroups() const { return m_edgesGroups; }
    const sofa::helper::vector< PrimitiveGroup > & getTrianglesGroups() const { return m_trianglesGroups; }
    const sofa::helper::vector< PrimitiveGroup > & getQuadsGroups() const { return m_quadsGroups; }
    const sofa::helper::vector< PrimitiveGroup > & getPolygonsGroups() const { return m_polygonsGroups; }
    const sofa::helper::vector< PrimitiveGroup > & getTetrahedraGroups() const { return m_tetrahedraGroups; }
    const sofa::helper::vector< PrimitiveGroup > & getHexahedraGroups() const { return m_hexahedraGroups; }
    const sofa::helper::vector< PrimitiveGroup > & getPentahedraGroups() const { return m_pentahedraGroups; }
    const sofa::helper::vector< PrimitiveGroup > & getPyramidsGroups() const { return m_pyramidsGroups; }
    
    const sofa::helper::vector< HighOrderEdgePosition >& getHighOrderEdgePositions() const { return m_highOrderEdgePositions; }
    const sofa::helper::vector< HighOrderTrianglePosition >& getHighOrderTrianglePositions() const { return m_highOrderTrianglePositions; }
    const sofa::helper::vector< HighOrderQuadPosition >& getHighOrderQuadPositions() const { return m_highOrderQuadPositions; }
    

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
    sofa::helper::vector<Vector3> m_vertices;

    // Tab of 2D elements composition
    sofa::helper::vector< Topology::Edge > m_edges; ///< Edges of the mesh loaded
    sofa::helper::vector< Topology::Triangle > m_triangles; ///< Triangles of the mesh loaded
    sofa::helper::vector< Topology::Quad > m_quads; ///< Quads of the mesh loaded
    helper::vector< helper::vector <unsigned int> > m_polygons; ///< Polygons of the mesh loaded
    helper::vector< HighOrderEdgePosition > m_highOrderEdgePositions; ///< High order edge points of the mesh loaded
    helper::vector< HighOrderTrianglePosition > m_highOrderTrianglePositions; ///< High order triangle points of the mesh loaded
    helper::vector< HighOrderQuadPosition > m_highOrderQuadPositions; ///< High order quad points of the mesh loaded

    // Tab of 3D elements composition
    sofa::helper::vector< Topology::Tetrahedron > m_tetrahedra; ///< Tetrahedra of the mesh loaded
    sofa::helper::vector< Topology::Hexahedron > m_hexahedra; ///< Hexahedra of the mesh loaded
    
    // Groups
    helper::vector< PrimitiveGroup > m_edgesGroups; ///< Groups of Edges
    helper::vector< PrimitiveGroup > m_trianglesGroups; ///< Groups of Triangles
    helper::vector< PrimitiveGroup > m_quadsGroups; ///< Groups of Quads
    helper::vector< PrimitiveGroup > m_polygonsGroups; ///< Groups of Polygons
    helper::vector< PrimitiveGroup > m_tetrahedraGroups; ///< Groups of Tetrahedra
    helper::vector< PrimitiveGroup > m_hexahedraGroups; ///< Groups of Hexahedra
    helper::vector< PrimitiveGroup > m_pentahedraGroups; ///< Groups of Pentahedra
    helper::vector< PrimitiveGroup > m_pyramidsGroups; ///< Groups of Pyramids


    sofa::helper::vector<Vector3> texCoords; // for the moment, we suppose that texCoords is order 2 (2 texCoords for a vertex)
    sofa::helper::vector<Vector3> normals;
    sofa::helper::vector< sofa::helper::vector < sofa::helper::vector <int> > > facets;
    //sofa::core::objectmodel::Data< Material > material;
    Material material;

    std::vector<Material> materials;
    std::vector<PrimitiveGroup> groups;

    std::string textureName;

};

} // namespace io

} // namespace helper

} // namespace sofa

#endif
