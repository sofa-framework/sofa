/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include "OgreVisualModel.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/io/Mesh.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/BaseMapping.h>

#include "OgrePlanarReflectionMaterial.h"
#include <iostream>

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(OgreVisualModel)

int OgreVisualModel::meshName=0; //static counter to get unique name for entities
bool OgreVisualModel::lightsEnabled=false;

OgreVisualModel::OgreVisualModel():
    materialFile(initData(&materialFile,"OGREmaterialFile", "Entry of material definition in a .material file")),
    culling(initData(&culling,true, "culling", "Activate Back-face culling in Ogre")),
    ogreObject(NULL), ogreNormalObject(NULL), needUpdate(false)
{
}


OgreVisualModel::~OgreVisualModel()
{
}

void OgreVisualModel::init()
{
    sofa::component::visualmodel::VisualModelImpl::init();
    helper::vector< core::BaseMapping *> m; this->getContext()->get<core::BaseMapping >(&m);
    for (unsigned int i=0; i<m.size(); ++i)
    {
        if (m[i]->getTo()[0] == this)
        {
            needUpdate=true;
            break;
        }
    }
    this->getContext()->get< BaseOgreShaderParameter >(&shaderParameters, core::objectmodel::BaseContext::Local);
    this->getContext()->get< OgreShaderTextureUnit >(&shaderTextureUnits, core::objectmodel::BaseContext::Local);
}

void OgreVisualModel::reinit()
{
    sofa::component::visualmodel::VisualModelImpl::reinit();

    if (materials.getValue().empty())
    {
        materialToMesh.begin()->second.updateMaterial(this->material.getValue());
    }
    else
    {
        const helper::vector<Material> &vecMaterials=materials.getValue();
        for (unsigned int i=0; i<vecMaterials.size(); ++i)
        {
            const std::string &name=vecMaterials[i].name;
            materialToMesh[name].updateMaterial(vecMaterials[i]);
        }
    }
}

bool OgreVisualModel::loadTexture(const std::string& filename)
{
    std::string file=filename;
    return sofa::helper::system::DataRepository.findFile(file);
}


void OgreVisualModel::updateNormals(
    const ResizableExtVector<Coord>& vertices,
    const ResizableExtVector<Triangle>& triangles,
    const ResizableExtVector<Quad>& quads)
{
    if(this->m_updateNormals == false && this->m_vnormals.getValue().size() == 0 ) return;

    ResizableExtVector<Deriv>& normals = *(this->m_vnormals.beginEdit());
    int nbn = vertices.size();

    normals.resize(nbn);

    for (int i = 0; i < nbn; i++)
        normals[i].clear();

    for (unsigned int i = 0; i < triangles.size(); i++)
    {
        const Coord v1 = vertices[triangles[i][0]];
        const Coord v2 = vertices[triangles[i][1]];
        const Coord v3 = vertices[triangles[i][2]];
        Coord n = cross(v2-v1, v3-v1);

        normals[triangles[i][0]] += n;
        normals[triangles[i][1]] += n;
        normals[triangles[i][2]] += n;
    }

    for (unsigned int i = 0; i < quads.size(); i++)
    {
        const Coord & v1 = vertices[quads[i][0]];
        const Coord & v2 = vertices[quads[i][1]];
        const Coord & v3 = vertices[quads[i][2]];
        const Coord & v4 = vertices[quads[i][3]];
        Coord n1 = cross(v2-v1, v4-v1);
        Coord n2 = cross(v3-v2, v1-v2);
        Coord n3 = cross(v4-v3, v2-v3);
        Coord n4 = cross(v1-v4, v3-v4);

        normals[quads[i][0]] += n1;
        normals[quads[i][1]] += n2;
        normals[quads[i][2]] += n3;
        normals[quads[i][3]] += n4;
    }

    for (unsigned int i = 0; i < normals.size(); i++)
        normals[i].normalize();

    m_vnormals.endEdit();
}

void OgreVisualModel::prepareMesh()
{

    const ResizableExtVector<Triangle>& triangles = this->getTriangles();
    const ResizableExtVector<Quad>& quads = this->getQuads();
    const ResizableExtVector<Coord>& vertices = this->getVertices();

    OgreVisualModel::updateNormals(vertices,triangles,quads);

    ++meshName;
    //Create a model for the normals
    {
        std::ostringstream s;
        s << "OgreVisualNormalModel["<<meshName<<"]";
        normalName=s.str();
        ogreNormalObject = (Ogre::ManualObject *) mSceneMgr->createMovableObject(normalName,"ManualObject");
        ogreNormalObject->setDynamic(true);
        ogreNormalObject->setCastShadows(false);

        //Create the Material to draw the normals
        s.str("");
        s << "OgreNormalMaterial[" << ++SubMesh::materialUniqueIndex << "]" ;
        currentMaterialNormals = Ogre::MaterialManager::getSingleton().create(s.str(), "General");
        currentMaterialNormals->setLightingEnabled(false);
    }

    //Create a model for the Mesh
    {
        std::ostringstream s;
        s << "OgreVisualModel["<<meshName<<"]";
        modelName=s.str();
        ogreObject = (Ogre::ManualObject *) mSceneMgr->createMovableObject(modelName,"ManualObject");
        ogreObject->setDynamic(needUpdate);
        ogreObject->setCastShadows(true);


        helper::vector<FaceGroup> &g=*(groups.beginEdit());
        if (g.empty())
        {
            SubMesh &mesh=materialToMesh[this->material.getValue().name];
            mesh.textureName = texturename.getValue();

            for (int i=vertices.size()-1; i>=0; --i) mesh.indices.insert(i);
            std::copy(triangles.begin(), triangles.end(), std::back_inserter(mesh.triangles));
            std::copy(quads.begin(), quads.end(),  std::back_inserter(mesh.quads));
            mesh.createMaterial(&shaderTextureUnits, this->material.getValue(), materialFile.getValue());
        }
        else
        {
            //Create a mesh for each material
            for (unsigned int i=0; i<g.size(); ++i)
            {
                SubMesh &mesh=materialToMesh[g[i].materialName];

                for (int t=0; t<g[i].nbt; ++t)
                {
                    const Triangle &T=triangles[g[i].t0+t];
                    mesh.indices.insert(T[0]);
                    mesh.indices.insert(T[1]);
                    mesh.indices.insert(T[2]);
                    mesh.triangles.push_back(T);
                }
                for (int q=0; q<g[i].nbq; ++q)
                {
                    const Quad &Q=quads[g[i].q0+q];
                    mesh.indices.insert(Q[0]);
                    mesh.indices.insert(Q[1]);
                    mesh.indices.insert(Q[2]);
                    mesh.indices.insert(Q[3]);
                    mesh.quads.push_back(Q);
                }

                if (mesh.material.isNull())
                {
                    mesh.textureName = texturename.getValue();
                    if (g[i].materialId < 0) mesh.createMaterial(&shaderTextureUnits, this->material.getValue(), materialFile.getValue());
                    else                     mesh.createMaterial(&shaderTextureUnits, this->materials.getValue()[g[i].materialId], materialFile.getValue());
                }
            }
        }

        std::map<std::string, SubMesh>::iterator it;
        int idx=0;
        for (it=materialToMesh.begin(); it!=materialToMesh.end(); ++it)
        {
            it->second.init(idx);
        }
    }
}

void OgreVisualModel::uploadNormals()
{
    const ResizableExtVector<Coord>& vertices = this->getVertices();
    const ResizableExtVector<Coord>& vnormals = this->getVnormals();
    for (unsigned int i = 0; i < vertices.size(); i++)
    {
        ogreNormalObject->position(vertices[i][0],vertices[i][1],vertices[i][2]);
        ogreNormalObject->position(vertices[i][0]+vnormals[i][0],vertices[i][1]+vnormals[i][1],vertices[i][2]+vnormals[i][2]);
        ogreNormalObject->index(2*i);
        ogreNormalObject->index(2*i+1);
    }
}


void OgreVisualModel::convertManualToMesh()
{
    if (mSceneMgr->hasEntity(modelName+"ENTITY"))
    {
        mSceneMgr->destroyEntity(modelName+"ENTITY");
        Ogre::MeshManager::getSingleton().remove(modelName+"MESH");
    }
    Ogre::MeshPtr ogreMesh = ogreObject->convertToMesh(modelName+"MESH", "General");
    ogreMesh->buildTangentVectors();
    Ogre::Entity *e = mSceneMgr->createEntity(modelName+"ENTITY", ogreMesh->getName());

    for (MatToMesh::iterator it=materialToMesh.begin(); it!=materialToMesh.end(); ++it)
        it->second.updateMeshCustomParameter(e);


    mSceneMgr->getRootSceneNode()->attachObject(e);
}


void OgreVisualModel::updateVisibility()
{
    if (lightsEnabled || !needUpdate) //Remove Manual Object, put mesh
    {
        if (ogreObject->isAttached())  mSceneMgr->getRootSceneNode()->detachObject(ogreObject);

        if (!mSceneMgr->hasEntity(modelName+"ENTITY")) return;
        Ogre::Entity *e=mSceneMgr->getEntity(modelName+"ENTITY");

        if (!this->getContext()->getShowVisualModels())
        {
            if (e->isAttached()) mSceneMgr->getRootSceneNode()->detachObject(e);
            if (ogreNormalObject->isAttached()) mSceneMgr->getRootSceneNode()->detachObject(ogreNormalObject);
        }
        else
        {
            if (!e->isAttached()) mSceneMgr->getRootSceneNode()->attachObject(e);

            if (this->getContext()->getShowNormals())
            {
                if (!ogreNormalObject->isAttached()) mSceneMgr->getRootSceneNode()->attachObject(ogreNormalObject);
            }
            else
            {
                if (ogreNormalObject->isAttached()) mSceneMgr->getRootSceneNode()->detachObject(ogreNormalObject);
            }
        }
    }
    else //Remove Mesh, put Manual Object
    {
        if (mSceneMgr->hasEntity(modelName+"ENTITY"))
        {
            Ogre::Entity *e=mSceneMgr->getEntity(modelName+"ENTITY");
            mSceneMgr->getRootSceneNode()->detachObject(e);
        }

        if (!this->getContext()->getShowVisualModels())
        {
            if (ogreObject->isAttached()) mSceneMgr->getRootSceneNode()->detachObject(ogreObject);
            if (ogreNormalObject->isAttached()) mSceneMgr->getRootSceneNode()->detachObject(ogreNormalObject);
        }
        else
        {
            if (!ogreObject->isAttached()) mSceneMgr->getRootSceneNode()->attachObject(ogreObject);

            if (this->getContext()->getShowNormals())
            {
                if (!ogreNormalObject->isAttached()) mSceneMgr->getRootSceneNode()->attachObject(ogreNormalObject);
            }
            else
            {
                if (ogreNormalObject->isAttached()) mSceneMgr->getRootSceneNode()->detachObject(ogreNormalObject);
            }
        }
    }
}

void OgreVisualModel::internalDraw(bool /*transparent*/)
{
    const ResizableExtVector<Coord>& vertices = this->getVertices();
    const ResizableExtVector<Coord>& normals = this->getVnormals();
    const ResizableExtVector<TexCoord>& texCoords= this->getVtexcoords();
    if (!ogreObject)
    {
        //If visual model is empty, return
        if (getTriangles().empty() && getQuads().empty()) return;

        prepareMesh();

        for (MatToMesh::iterator it=materialToMesh.begin(); it!=materialToMesh.end(); ++it)
            it->second.create(ogreObject,&shaderParameters,vertices,normals,texCoords);

        if (!needUpdate) convertManualToMesh();

        if (getContext()->getShowNormals())
        {
            ogreNormalObject->begin(currentMaterialNormals->getName(), Ogre::RenderOperation::OT_LINE_LIST);
            uploadNormals();
            ogreNormalObject->end();
        }
    }
    else
    {
        updateVisibility();
        if (!this->getContext()->getShowVisualModels()) return;
        if (!needUpdate) return;

        //Visual Model update
        for (MatToMesh::iterator it=materialToMesh.begin(); it!=materialToMesh.end(); ++it)
            it->second.update(vertices, normals,texCoords);

        if (lightsEnabled && vertices.size() != 0) convertManualToMesh();

        if (this->getContext()->getShowNormals())
        {
            //Normals
            if (ogreNormalObject->getNumSections())
            {
                ogreNormalObject->beginUpdate(0);
                uploadNormals();
                ogreNormalObject->end();
            }
            else
            {
                ogreNormalObject->begin(currentMaterialNormals->getName(), Ogre::RenderOperation::OT_LINE_LIST);
                uploadNormals();
                ogreNormalObject->end();
            }
        }
    }
}

void OgreVisualModel::setVisible(bool visible)
{
    if (!mSceneMgr->hasEntity(modelName+"ENTITY")) return;
    Ogre::Entity *e=mSceneMgr->getEntity(modelName+"ENTITY");

    e->setVisible(visible);

}

void OgreVisualModel::applyUVTransformation()
{
// 	for (unsigned int i=0;i<vtexcoords.size();++i)
// 	  {
// 	    vtexcoords[i][0] = vtexcoords[i][0];
// 	    vtexcoords[i][1] = 1-vtexcoords[i][1];
// 	  }
    this->applyUVScale(m_scaleTex.getValue()[0], m_scaleTex.getValue()[1]);
    this->applyUVTranslation(m_translationTex.getValue()[0],m_translationTex.getValue()[1]);
    m_scaleTex.setValue(TexCoord(1,1));
    m_translationTex.setValue(TexCoord(0,0));
}
int OgreVisualModelClass = sofa::core::RegisterObject("OGRE 3D Visual Model")
        .add < OgreVisualModel >();

}
}
}
