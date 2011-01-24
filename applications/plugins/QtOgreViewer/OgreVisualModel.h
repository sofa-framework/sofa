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
#ifndef OGREVISUALMODEL_H
#define OGREVISUALMODEL_H

#include <Ogre.h>
#include "DotSceneLoader.h"

#include "OgreShaderParameter.h"
#include "OgreShaderTextureUnit.h"

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/VisualModel.h>
#include <sofa/core/loader/Material.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/component/visualmodel/VisualModelImpl.h>
#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{


class SubMesh
{

    typedef VisualModelImpl::Triangle  Triangle;
    typedef VisualModelImpl::Quad Quad;
    typedef std::set< unsigned int > Indices;
    typedef helper::vector< Triangle > VecTriangles;
    typedef helper::vector< Quad > VecQuads;
    typedef ExtVec3fTypes::Coord Coord;
    typedef Vec<2, float> TexCoord;

    struct InternalStructure
    {
        Indices indices;
        VecTriangles triangles;
        VecQuads quads;

        template <class T>
        void  updatePrimitive(T& primitive)
        {
            for (unsigned int i=0; i<T::size(); ++i) primitive[i] = globalToLocalPrimitives[ primitive[i] ];
        }

        void computeGlobalToLocalPrimitives()
        {
            globalToLocalPrimitives.clear();
            unsigned int idx=0;
            for (Indices::const_iterator it=indices.begin(); it!=indices.end(); ++it)
                globalToLocalPrimitives.insert(std::make_pair(*it, idx++));

            for (VecTriangles::iterator it=triangles.begin(); it!=triangles.end(); ++it) updatePrimitive(*it);
            for (VecQuads::iterator it=quads.begin(); it!=quads.end(); ++it)             updatePrimitive(*it);
        }

        std::map< unsigned int, unsigned int > globalToLocalPrimitives;
    };

public:

    SubMesh():maxPrimitives(10000) {};

    int index;
    Indices indices;
    Ogre::MaterialPtr material;
    std::string materialName;
    std::string shaderName;
    std::string textureName;
    core::loader::Material sofaMaterial;

    VecTriangles triangles;
    VecQuads quads;

    void init(int &idx);
    void create(Ogre::ManualObject *,
            helper::vector< BaseOgreShaderParameter*> *,
            const ResizableExtVector<Coord>& positions,
            const ResizableExtVector<Coord>& normals,
            const ResizableExtVector<TexCoord>& textCoords) const;
    void update(const ResizableExtVector<Coord>& positions,
            const ResizableExtVector<Coord>& normals,
            const ResizableExtVector<TexCoord>& textCoords) const ;

    Ogre::MaterialPtr createMaterial(helper::vector< OgreShaderTextureUnit*> *, const core::loader::Material &sofaMaterial, const std::string &shaderName);
    void updateMaterial(const core::loader::Material &sofaMaterial);

    void updateMeshCustomParameter(Ogre::Entity *entity) const;
    void updateManualObjectCustomParameter() const;

    template <class ObjectType>
    void updateCustomParameters(ObjectType *section) const;

    static int materialUniqueIndex;
protected:

    void computeGlobalToLocalPrimitives();


    helper::vector< InternalStructure > storage;
    const unsigned int maxPrimitives;
    mutable Ogre::ManualObject *model;
    mutable helper::vector< BaseOgreShaderParameter*>* shaderParameters;
    helper::vector< OgreShaderTextureUnit*>* shaderTextureUnits;
};


class OgreVisualModel : public sofa::component::visualmodel::VisualModelImpl
{
public:
    SOFA_CLASS(OgreVisualModel,sofa::component::visualmodel::VisualModelImpl);
    typedef sofa::component::visualmodel::VisualModelImpl Inherit;
    OgreVisualModel();
    ~OgreVisualModel();
    void setOgreSceneManager(Ogre::SceneManager* m) {mSceneMgr=m;}
private:
    virtual void internalDraw(bool transparent=false);
public:
    virtual void init();
    virtual void reinit();
    virtual void initVisual() {internalDraw();}
    virtual void initTextures() {internalDraw();}

    virtual bool loadTexture(const std::string& filename);
    virtual void applyUVTransformation();

    virtual void setVisible(bool visible);

    static bool lightsEnabled;
protected:

    bool prepareMeshSimple();
    void prepareMesh();
    void updateVisibility();
    void uploadNormals();
    void convertManualToMesh();

    static int meshName;

    Data< std::string > materialOgre;
    Data< std::string > materialFile;
    Data< bool > culling;


    std::string modelName;
    std::string normalName;

    Ogre::ManualObject *ogreObject;
    Ogre::ManualObject *ogreNormalObject;
    Ogre::SceneManager* mSceneMgr;
    Ogre::MaterialPtr currentMaterial;
    Ogre::MaterialPtr currentMaterialNormals;


    helper::vector<BaseOgreShaderParameter*> shaderParameters;
    helper::vector<OgreShaderTextureUnit*>   shaderTextureUnits;

    typedef std::map<std::string, SubMesh> MatToMesh;
    MatToMesh materialToMesh;

    bool needUpdate;
};


template <class ObjectType>
void SubMesh::updateCustomParameters(ObjectType *section) const
{
    section->setCustomParameter(1, Ogre::Vector4(sofaMaterial.ambient.ptr()));
    section->setCustomParameter(2, Ogre::Vector4(sofaMaterial.diffuse.ptr()));
    section->setCustomParameter(3, Ogre::Vector4(sofaMaterial.specular.ptr()));
    section->setCustomParameter(4, Ogre::Vector4(sofaMaterial.shininess,0,0,0));

    for (unsigned int p=0; p<shaderParameters->size(); ++p)
    {
        if ((*shaderParameters)[p]->isDirty())
        {
            Ogre::Vector4 value;
            BaseOgreShaderParameter* parameter=(*shaderParameters)[p];
            parameter->getValue(value);
            section->setCustomParameter(parameter->getEntryPoint(), value);
        }
    }

};
}
}
}
#endif
