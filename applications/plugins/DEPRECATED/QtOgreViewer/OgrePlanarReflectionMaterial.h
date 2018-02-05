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
#ifndef SOFA_GUI_QT_VIEWER_QTOGRE_OGREREFLECTIONTEXTURE_H
#define SOFA_GUI_QT_VIEWER_QTOGRE_OGREREFLECTIONTEXTURE_H


#include "OgreSceneObject.h"

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <string>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <Ogre.h>



namespace sofa
{
namespace component
{
namespace visualmodel
{


class OgreVisualModel;

class OgrePlanarReflectionMaterial : public core::visual::VisualModel, public Ogre::RenderTargetListener,
    public core::ogre::OgreSceneObject
{
public:
    SOFA_CLASS(OgrePlanarReflectionMaterial,core::visual::VisualModel);

    typedef sofa::defaulttype::Vec<2,int> Vec2i;

    OgrePlanarReflectionMaterial();
    virtual ~OgrePlanarReflectionMaterial();

    void initVisual();
    void reinit();

    // VisualModel methods

    void drawVisual();
    void updateVisual();

    //RenderTargetListenerMethods
    void preRenderTargetUpdate(const Ogre::RenderTargetEvent& evt);
    void postRenderTargetUpdate(const Ogre::RenderTargetEvent &evt);

    core::objectmodel::Data< sofa::defaulttype::Vector3 > normalReflectionPlane;
    core::objectmodel::Data< SReal> distanceToOrigin;
    core::objectmodel::Data< Vec2i > sizeReflectionPlane;
    core::objectmodel::Data< Vec2i > resolutionTexture;
    core::objectmodel::DataFileName textureFilename;
    core::objectmodel::Data<SReal>  blendingFactor;
    core::objectmodel::Data< bool > debugDraw;


protected:
    static int numInstances;
    Ogre::Root* mRoot;

    Ogre::Camera* mCamera;
    Ogre::MovablePlane* mPlane;

    Ogre::Entity* mDebugPlaneEntity;
    Ogre::SceneNode* mPlaneNode;
    Ogre::TexturePtr mRttPtr;
    Ogre::MaterialPtr mMatPtr;

    Ogre::MovablePlane* createReflectionPlane();
    Ogre::Camera* createReflectionCamera();
    Ogre::TexturePtr createRenderTargetTexture(Ogre::Camera& camera);
    Ogre::MaterialPtr createMaterial(const Ogre::TexturePtr& texture, const Ogre::Camera& projectionCamera);
    Ogre::SceneNode* createPlaneSceneNode(Ogre::MovablePlane& plane);
    Ogre::Entity* createDebugPlaneEntity(const Ogre::MovablePlane& plane);

    void updateReflectionCamera(const Ogre::MovablePlane& plane);
    void updatePlaneSceneNode();



    helper::vector< OgreVisualModel* > objectsToHide;




};
}
}
}

#endif // SOFA_GUI_QT_VIEWER_QTOGRE_OGREREFLECTIONTEXTURE_H
