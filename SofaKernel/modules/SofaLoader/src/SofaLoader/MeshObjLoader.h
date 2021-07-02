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
#include <SofaLoader/config.h>
#include <sofa/core/loader/MeshLoader.h>
#include <sofa/type/SVector.h>
#include <sofa/type/Material.h>

namespace sofa::component::loader
{

using sofa::core::objectmodel::BaseData;

class SOFA_SOFALOADER_API MeshObjLoader : public sofa::core::loader::MeshLoader
{
public:
    enum FaceType { EDGE, TRIANGLE, QUAD, NBFACETYPE };

    SOFA_CLASS(MeshObjLoader,sofa::core::loader::MeshLoader);
protected:
    MeshObjLoader();
    ~MeshObjLoader() override;

public:
    bool doLoad() override;

protected:
    bool readOBJ (std::ifstream &file, const char* filename);
    bool readMTL (const char* filename, type::vector<sofa::type::Material>& d_materials);
    void addGroup (const sofa::core::loader::PrimitiveGroup& g);
    void doClearBuffers() override;

    std::string textureName;
    FaceType faceType;

public:
    Data<bool> d_handleSeams;
    Data<bool> d_loadMaterial;
    Data<sofa::type::Material> d_material;
    Data <type::vector<sofa::type::Material> > d_materials;
    Data <type::SVector <type::SVector <int> > > d_faceList;
    Data <type::SVector <type::SVector <int> > > d_texIndexList;
    Data <type::vector<sofa::type::Vector3> > d_positionsList;
    Data< type::vector<sofa::type::Vector2> > d_texCoordsList;
    Data <type::SVector<type::SVector<int> > > d_normalsIndexList;
    Data <type::vector<sofa::type::Vector3> > d_normalsList;
    Data< type::vector<sofa::type::Vector2> > d_texCoords;
    Data< bool > d_computeMaterialFaces;
    type::vector< Data <type::vector<unsigned int> >* > d_subsets_indices;

    /// If vertices have multiple normals/texcoords, then we need to separate them
    /// This vector store which input position is used for each vertex
    /// If it is empty then each vertex correspond to one position
    Data< type::vector<int> > d_vertPosIdx;

    /// Similarly this vector store which input normal is used for each vertex
    /// If it is empty then each vertex correspond to one normal
    Data< type::vector<int> > d_vertNormIdx;

    virtual std::string type() { return "The format of this mesh is OBJ."; }
};


} // namespace sofa::component::loader
