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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_LOADER_MESHOBJLOADER_H
#define SOFA_COMPONENT_LOADER_MESHOBJLOADER_H

#include <sofa/core/componentmodel/loader/MeshLoader.h>
#include <sofa/component/component.h>

namespace sofa
{

namespace component
{

namespace loader
{

//  using namespace sofa::defaulttype;
//  using namespace sofa::helper::io;
using sofa::defaulttype::Vec4f;

class SOFA_COMPONENT_LOADER_API MeshObjLoader : public sofa::core::componentmodel::loader::MeshLoader
{
public:
    enum FaceType { EDGE, TRIANGLE, QUAD };

    SOFA_CLASS(MeshObjLoader,sofa::core::componentmodel::loader::MeshLoader);

    MeshObjLoader();

    virtual bool load();

    template <class T>
    static bool canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
    {
        return BaseLoader::canCreate (obj, context, arg);
    }

protected:

    bool readOBJ (std::ifstream &file, const char* filename);
    bool readMTL (const char* filename, helper::vector <sofa::core::componentmodel::loader::Material>& materials);
    void addGroup (const sofa::core::componentmodel::loader::PrimitiveGroup& g);

    sofa::core::componentmodel::loader::Material material;
    std::string textureName;
    FaceType faceType;

public:

    Data <helper::vector <sofa::core::componentmodel::loader::Material> > materials;

    Data <helper::vector <helper::vector <int> > > texturesList;
    Data< helper::vector<sofa::defaulttype::Vector2> > texCoords;

    Data <helper::vector <helper::vector <int> > > normalsList; //TODO: check if it not the sames daata as normals and texCoords

    virtual std::string type() { return "The format of this mesh is OBJ."; }

    //    vector<Material>& getMaterialsArray() { return m_materials; }

};




} // namespace loader

} // namespace component

} // namespace sofa

#endif
