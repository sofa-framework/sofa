/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_LOADER_OGREMESHLOADER_H
#define SOFA_COMPONENT_LOADER_OGREMESHLOADER_H

#include <sofa/core/loader/MeshLoader.h>
#include <Ogre.h>

namespace sofa
{
namespace component
{
namespace loader
{

class OgreMeshLoader : public sofa::core::loader::MeshLoader
{
public:
    SOFA_CLASS(OgreMeshLoader,sofa::core::loader::MeshLoader);

    OgreMeshLoader();

    bool canLoad();

    bool load();

protected:
    bool readMesh(Ogre::Mesh* mesh);
    void readMeshVertices(Ogre::VertexData* vertexData, helper::vector< sofa::defaulttype::Vector3 >& vertices);
    void readMeshNormals(Ogre::VertexData* vertexData, helper::vector< sofa::defaulttype::Vector3 >& normal);
    void readMeshTexCoords(Ogre::VertexData* vertexData, helper::vector< sofa::defaulttype::Vector2>& coord);
    //void readMeshIndices(Ogre::VertexData* vertexData, helper::vector< helper::fixed_array <unsigned int,3> >& indices);

    Data< helper::vector<sofa::defaulttype::Vector2> > texCoords;

};

}
}
}


#endif // SOFA_COMPONENT_LOADER_OGREMESHLOADER_H
