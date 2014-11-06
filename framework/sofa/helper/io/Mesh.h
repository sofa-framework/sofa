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
#include <sofa/core/loader/Material.h>
#include <sofa/SofaFramework.h>

namespace sofa
{

namespace helper
{

namespace io
{

class SOFA_HELPER_API Mesh
{    
public:
    
    std::string loaderType;
    
public:
    typedef sofa::defaulttype::Vector3 Vector3;
    typedef sofa::core::loader::PrimitiveGroup PrimitiveGroup;
    typedef sofa::core::loader::Material Material;
    sofa::helper::vector<Vector3> & getVertices()
    {
        //std::cout << "vertices size : " << vertices.size() << std::endl;
        return vertices;
    };
    sofa::helper::vector<Vector3> & getTexCoords() { return texCoords; }
    sofa::helper::vector<Vector3> & getNormals() { return normals; }
    sofa::helper::vector< vector < vector <int> > > & getFacets()
    {
        //std::cout << "facets size : " << facets.size() << std::endl;
        return facets;
    };
    const Material& getMaterial() {return material; }

    const std::vector<Material>& getMaterials() {return materials; }
    const std::vector<PrimitiveGroup>& getGroups() {return groups; }

    std::string& getTextureName()
    {
        return textureName;
    };

    typedef Factory<std::string, Mesh, std::string> FactoryMesh;

    static Mesh* Create(std::string filename);
    static Mesh* Create(std::string loader, std::string filename);

    template<class Object>
    static Object* create(Object*, std::string arg)
    {
        return new Object(arg);
    }
    
protected:

    sofa::helper::vector<Vector3> vertices;
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
