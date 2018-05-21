/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

    sofa::helper::vector<Vector3> m_vertices;
    sofa::helper::vector< Topology::Edge > m_edges;
    sofa::helper::vector< Topology::Triangle > m_triangles;

    sofa::helper::vector< Topology::Quad > m_quads;
    sofa::helper::vector< Topology::Tetrahedron > m_tetrahedra;
    sofa::helper::vector< Topology::Hexahedron > m_hexahedra;
    


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
