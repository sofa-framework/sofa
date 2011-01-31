/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "OgreVisualModel.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/io/Mesh.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/BaseMapping.h>

#include "OgreReflectionTexture.h"
#include <iostream>

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(OgreVisualModel)

int OgreVisualModel::meshName=0; //static counter to get unique name for entities
int SubMesh::materialUniqueIndex=0;

bool OgreVisualModel::lightsEnabled=false;



//-------------------------------------------
// SubMesh
//-------------------------------------------
void SubMesh::init(int &idx)
{
    storage.clear();

    const unsigned int numPrimitives=triangles.size()+2*quads.size();
    storage.resize(1+numPrimitives/maxPrimitives);

    index=idx; idx+=storage.size();

    for (unsigned int i=0; i<triangles.size(); ++i)
    {
        const unsigned int storageIdx=i/maxPrimitives;
        const Triangle &T=triangles[i];
        for (unsigned int idx=0; idx<Triangle::size(); ++idx) storage[storageIdx].indices.insert(T[idx]);
        storage[storageIdx].triangles.push_back(T);
    }

    for (unsigned int i=0; i<quads.size(); ++i)
    {
        const unsigned int storageIdx=(triangles.size()+i)/maxPrimitives;
        const Quad &Q=quads[i];
        for (unsigned int idx=0; idx<Quad::size(); ++idx)  storage[storageIdx].indices.insert(Q[idx]);
        storage[storageIdx].quads.push_back(Q);
    }

    computeGlobalToLocalPrimitives();
}

void SubMesh::computeGlobalToLocalPrimitives()
{
    for (unsigned int i=0; i<storage.size(); ++i) storage[i].computeGlobalToLocalPrimitives();
}


void SubMesh::create(Ogre::ManualObject *ogreObject,
        helper::vector< BaseOgreShaderParameter*> *parameters,
        const ResizableExtVector<Coord>& positions,
        const ResizableExtVector<Coord>& normals,
        const ResizableExtVector<TexCoord>& textCoords) const
{
    model=ogreObject;
    shaderParameters=parameters;

    const bool hasTexCoords= !(textCoords.empty());


    for (unsigned int i=0; i<storage.size(); ++i)
    {
        model->begin(material->getName(), Ogre::RenderOperation::OT_TRIANGLE_LIST);
        for (Indices::const_iterator it=storage[i].indices.begin(); it!=storage[i].indices.end(); ++it)
        {
            const int idx=*it;
            model->position(positions[idx][0],positions[idx][1],positions[idx][2]);
            model->normal(normals[idx][0],normals[idx][1],normals[idx][2]);
            if (hasTexCoords) model->textureCoord(textCoords[idx][0],textCoords[idx][1]);
        }

        for (unsigned int t=0; t<storage[i].triangles.size(); ++t)
        {
            const Triangle &T=storage[i].triangles[t];
            model->triangle(T[0],T[1],T[2]);
        }
        for (unsigned int q=0; q<storage[i].quads.size(); ++q)
        {
            const Quad &Q=storage[i].quads[q];
            model->quad(Q[0],Q[1],Q[2],Q[3]);
        }
        model->end();
    }

}

void SubMesh::update(const ResizableExtVector<Coord>& positions,
        const ResizableExtVector<Coord>& normals,
        const ResizableExtVector<TexCoord>& textCoords) const
{

    const bool hasTexCoords= !(textCoords.empty());
    for (unsigned int i=0; i<storage.size(); ++i)
    {
        model->beginUpdate(index+i);

        for (Indices::const_iterator it=storage[i].indices.begin(); it!=storage[i].indices.end(); ++it)
        {
            const int idx=*it;
            model->position(positions[idx][0],positions[idx][1],positions[idx][2]);
            model->normal(normals[idx][0],normals[idx][1],normals[idx][2]);
            if (hasTexCoords) model->textureCoord(textCoords[idx][0],textCoords[idx][1]);
        }
        for (unsigned int t=0; t<storage[i].triangles.size(); ++t)
        {
            const Triangle &T=storage[i].triangles[t];
            model->triangle(T[0],T[1],T[2]);
        }
        for (unsigned int q=0; q<storage[i].quads.size(); ++q)
        {
            const Quad &Q=storage[i].quads[q];
            model->quad(Q[0],Q[1],Q[2],Q[3]);
        }
        model->end();
    }

    if (!OgreVisualModel::lightsEnabled) updateManualObjectCustomParameter();
}

//-------------------------------------------

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


Ogre::MaterialPtr SubMesh::createMaterial(helper::vector< OgreShaderTextureUnit*> *textureUnits, const core::loader::Material &sofaMaterial, const std::string &name)
{
    shaderTextureUnits=textureUnits;
    shaderName=name;
    materialName = sofaMaterial.name;
    std::ostringstream s;
    s << "OgreVisualMaterial[" << ++materialUniqueIndex<< "]" ;

    if (!shaderName.empty())
    {
        Ogre::MaterialPtr cel=Ogre::MaterialManager::getSingleton().createOrRetrieve(shaderName, "General").first;
        material = cel->clone(s.str());
    }
    else
    {
        material = Ogre::MaterialManager::getSingleton().create(s.str(), "General");

        material->setReceiveShadows(true);
        material->getTechnique(0)->getPass(0)->setLightingEnabled(true);
    }

    updateMaterial(sofaMaterial);
    material->compile();
    return material;
}

void SubMesh::updateMaterial(const core::loader::Material &s)
{
    sofaMaterial=s;
    if (material.isNull()) return;


    for (unsigned int i=0; i<shaderTextureUnits->size(); ++i)
    {
        OgreShaderTextureUnit *textureUnit=(*shaderTextureUnits)[i];
        Ogre::TextureUnitState *tex=material->getTechnique(textureUnit->getTechniqueIndex())->getPass(textureUnit->getPassIndex())->getTextureUnitState(textureUnit->getTextureIndex());
        if (tex) tex->setTextureName(textureUnit->getTextureName());
        material->getTechnique(textureUnit->getTechniqueIndex())->getPass(textureUnit->getPassIndex())->createTextureUnitState(textureUnit->getTextureName(), textureUnit->getTextureIndex());
    }

    if (!shaderName.empty()) return;


    if (!textureName.empty() && Ogre::ResourceGroupManager::getSingleton().resourceExists("General",textureName) )
        material->getTechnique(0)->getPass(0)->createTextureUnitState(textureName);


    if (sofaMaterial.useDiffuse)
        material->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(sofaMaterial.diffuse[0],sofaMaterial.diffuse[1],sofaMaterial.diffuse[2],sofaMaterial.diffuse[3]));
    else
        material->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(0,0,0,0));

    if (sofaMaterial.useAmbient)
        material->getTechnique(0)->getPass(0)->setAmbient(Ogre::ColourValue(sofaMaterial.ambient[0],sofaMaterial.ambient[1],sofaMaterial.ambient[2],sofaMaterial.ambient[3]));
    else
        material->getTechnique(0)->getPass(0)->setAmbient(Ogre::ColourValue(0,0,0,0));

    if (sofaMaterial.useEmissive)
        material->getTechnique(0)->getPass(0)->setSelfIllumination(Ogre::ColourValue(sofaMaterial.emissive[0],sofaMaterial.emissive[1],sofaMaterial.emissive[2],sofaMaterial.emissive[3]));
    else
        material->getTechnique(0)->getPass(0)->setSelfIllumination(Ogre::ColourValue(0,0,0,0));

    if (sofaMaterial.useSpecular)
        material->getTechnique(0)->getPass(0)->setSpecular(Ogre::ColourValue(sofaMaterial.specular[0],sofaMaterial.specular[1],sofaMaterial.specular[2],sofaMaterial.specular[3]));
    else
        material->getTechnique(0)->getPass(0)->setSpecular(Ogre::ColourValue(0,0,0,0));

    if (sofaMaterial.useShininess)
        material->getTechnique(0)->getPass(0)->setShininess(Ogre::Real(sofaMaterial.shininess));
    else
        material->getTechnique(0)->getPass(0)->setShininess(Ogre::Real(45));

    if ( (sofaMaterial.useDiffuse && sofaMaterial.diffuse[3] < 1) ||
            (sofaMaterial.useAmbient && sofaMaterial.ambient[3] < 1) )
    {
        material->setDepthWriteEnabled(false);
        material->getTechnique(0)->setSceneBlending(Ogre::SBT_TRANSPARENT_ALPHA);
        material->setCullingMode(Ogre::CULL_NONE);
    }
}

void SubMesh::updateManualObjectCustomParameter() const
{
    for (unsigned int i=0; i<storage.size(); ++i)  updateCustomParameters(model->getSection(index+i));
}

void SubMesh::updateMeshCustomParameter(Ogre::Entity *entity) const
{
    for (unsigned int i=0; i<storage.size(); ++i)  updateCustomParameters(entity->getSubEntity(index+i));
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
