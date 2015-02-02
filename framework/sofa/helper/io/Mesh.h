/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_IO_MESH_H
#define SOFA_HELPER_IO_MESH_H

#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/Factory.h>
#include <sofa/core/loader/PrimitiveGroup.h>
//#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

namespace io
{

using sofa::helper::vector;
using sofa::core::loader::Material;
using sofa::core::loader::PrimitiveGroup;
using sofa::defaulttype::Vector3;

using sofa::defaulttype::Vec4f;

class SOFA_HELPER_API Mesh
{

protected:
    vector<Vector3> vertices;
    vector<Vector3> texCoords; // for the moment, we suppose that texCoords is order 2 (2 texCoords for a vertex)
    vector<Vector3> normals;
    vector< vector < vector <int> > > facets;
    //sofa::core::objectmodel::Data< Material > material;
    Material material;

    std::vector<Material> materials;
    std::vector<PrimitiveGroup> groups;

    std::string textureName;
    
public:
    
    std::string loaderType;
    
public:

    vector<Vector3> & getVertices() { return vertices; }
    const vector<Vector3> & getVertices() const { return vertices; }
    vector<Vector3> & getTexCoords() { return texCoords; }
    const vector<Vector3> & getTexCoords() const { return texCoords; }
    vector<Vector3> & getNormals() { return normals; }
    const vector<Vector3> & getNormals() const { return normals; }
    vector< vector < vector <int> > > & getFacets() { return facets; }
    const vector< vector < vector <int> > > & getFacets() const { return facets; }

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
    
};

} // namespace io

} // namespace helper

} // namespace sofa

#endif
