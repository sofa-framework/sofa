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
#include "SubMesh.h"
#include "OgreVisualModel.h"
#include "OgreShaderTextureUnit.h"

namespace sofa
{
namespace component
{
namespace visualmodel
{

int SubMesh::materialUniqueIndex=0;
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



}
}
}

