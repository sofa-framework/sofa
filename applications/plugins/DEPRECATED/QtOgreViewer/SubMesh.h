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
#ifndef SOFA_COMPONENT_VISUALMODEL_OGRESUBMESH
#define SOFA_COMPONENT_VISUALMODEL_OGRESUBMESH

#include <Ogre.h>
#include <SofaBaseVisual/VisualModelImpl.h>
#include "OgreShaderParameter.h"

namespace sofa
{
namespace component
{
namespace visualmodel
{

class BaseOgreShaderParameter;
class OgreShaderTextureUnit;

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
    void updateCustomParameters(ObjectType *section) const
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

    static int materialUniqueIndex;
protected:

    helper::vector< InternalStructure > storage;
    const unsigned int maxPrimitives;
    mutable Ogre::ManualObject *model;
    mutable helper::vector< BaseOgreShaderParameter*>* shaderParameters;
    helper::vector< OgreShaderTextureUnit*>* shaderTextureUnits;
};



}
}
}



#endif // SOFA_COMPONENT_VISUALMODEL_OGRESUBMESH
