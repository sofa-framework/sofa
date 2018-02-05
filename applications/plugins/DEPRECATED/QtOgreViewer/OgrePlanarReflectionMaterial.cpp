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
#include "OgrePlanarReflectionMaterial.h"
#include "OgreVisualModel.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/Tag.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/Simulation.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{

/*ObjectFactory code */

SOFA_DECL_CLASS(OgrePlanarReflectionMaterial);
int OgreReflectionTextureClass = core::RegisterObject("Generate a reflection texture with respect to a plane.")
        .add< OgrePlanarReflectionMaterial >()
        .addAlias("OgreReflectionTexture");


/* */


int OgrePlanarReflectionMaterial::numInstances = 0;

OgrePlanarReflectionMaterial::OgrePlanarReflectionMaterial():
    normalReflectionPlane(initData(&normalReflectionPlane,sofa::defaulttype::Vector3(0,1,0),"normal","Normal of the reflection plane."))
    ,distanceToOrigin(initData(&distanceToOrigin,0.0,"distance","Distance to origin along normal direction."))
    ,sizeReflectionPlane(initData(&sizeReflectionPlane,Vec2i(100,100),"size","Size of the reflection plane."))
    ,resolutionTexture(initData(&resolutionTexture,Vec2i(1024,1024),"resolutionTexture","The resolution of the generated texture." ))
    ,textureFilename(initData(&textureFilename,"textureFilename","Filename of an additional texture to blend with the relfection texture"))
    ,blendingFactor(initData(&blendingFactor,0.25,"blendingFactor","The blending factor between the reflection texture and textureFilename."))
    ,debugDraw(initData(&debugDraw,false,"debugDraw","Display a plane textured with the reflectionTexture."))
    ,mCamera(NULL)
    ,mPlane(NULL)
    ,mDebugPlaneEntity(NULL)
    ,mPlaneNode(NULL)
    ,mMatPtr(NULL)
{
    sofa::defaulttype::Vector3* normal = normalReflectionPlane.beginEdit();
    normal->normalize();
    normalReflectionPlane.endEdit();
    numInstances++;
}

OgrePlanarReflectionMaterial::~OgrePlanarReflectionMaterial()
{
    using namespace Ogre;
    mRttPtr->getBuffer()->getRenderTarget()->removeListener(this);
    MaterialManager::getSingleton().remove(mMatPtr->getName() );
    TextureManager::getSingleton().remove(mRttPtr->getName() );
    mPlaneNode->detachObject(mPlane);
    if( mDebugPlaneEntity->isAttached() )
    {
        mPlaneNode->detachObject(mDebugPlaneEntity);
    }

    mSceneMgr->destroyEntity(mPlane->getName());
    mSceneMgr->destroyEntity(mDebugPlaneEntity->getName());

    numInstances--;
    assert(numInstances >=0 );
}

Ogre::Camera* OgrePlanarReflectionMaterial::createReflectionCamera()
{
    std::ostringstream s;
    s << "ReflectionCamera[" << numInstances <<"]";
    Ogre::Camera* camera = mSceneMgr->createCamera(s.str() );

    return camera;
}

Ogre::MovablePlane* OgrePlanarReflectionMaterial::createReflectionPlane()
{
    std::ostringstream s;
    s << "ClipPlane[" << numInstances <<"]";
    Ogre::MovablePlane* plane = new Ogre::MovablePlane(s.str());
    plane->d = 0.0;
    Ogre::Vector3 n(normalReflectionPlane.getValue().elems[0]
            ,normalReflectionPlane.getValue().elems[1]
            ,normalReflectionPlane.getValue().elems[2]);
    plane->normal = n;
    return plane;
}

Ogre::SceneNode* OgrePlanarReflectionMaterial::createPlaneSceneNode(Ogre::MovablePlane& plane)
{
    Ogre::SceneNode* planeNode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
    //mPlaneNode->attachObject(mPlaneEntity);
    planeNode->attachObject(&plane);
    sofa::defaulttype::Vector3 dist = normalReflectionPlane.getValue() * distanceToOrigin.getValue();
    planeNode->setPosition(dist.elems[0],dist.elems[1],dist.elems[2]);
    return planeNode;
}

Ogre::Entity* OgrePlanarReflectionMaterial::createDebugPlaneEntity(const Ogre::MovablePlane& plane)
{
    using namespace Ogre;
    std::ostringstream s;
    s << "ReflectionPlane[" << numInstances <<"]";
    MeshManager::getSingleton().createPlane(s.str(),
            ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
            plane,
            sizeReflectionPlane.getValue().elems[0],
            sizeReflectionPlane.getValue().elems[1],
            1,1,true,1,1,1,
            plane.normal.perpendicular() )
    ;
    Ogre::Entity* planeEntity = mSceneMgr->createEntity("Plane",s.str());
    return planeEntity;
}

Ogre::TexturePtr OgrePlanarReflectionMaterial::createRenderTargetTexture(Ogre::Camera& camera)
{
    using namespace Ogre;
    std::ostringstream s;
    s << "Rtt[" << numInstances << "]";
    TexturePtr texture = TextureManager::getSingleton().createManual(
            s.str(),
            "General",
            Ogre::TEX_TYPE_2D,
            resolutionTexture.getValue().elems[0],
            resolutionTexture.getValue().elems[1],
            0,
            PF_R8G8B8,
            Ogre::TU_RENDERTARGET);
    Ogre::RenderTarget* rtt = texture->getBuffer()->getRenderTarget();
    Ogre::Viewport* v = rtt->addViewport( &camera );
    v->setClearEveryFrame(true);
    v->setBackgroundColour( Ogre::ColourValue::Black );
    return texture;
}

Ogre::MaterialPtr OgrePlanarReflectionMaterial::createMaterial(const Ogre::TexturePtr& texture, const Ogre::Camera& projectionCamera)
{
    using namespace::Ogre;

    Ogre::MaterialPtr matPtr = MaterialManager::getSingleton().create(this->getName(),"General");
    Ogre::TextureUnitState* t;
    if(! textureFilename.getValue().empty() )
    {
        t = matPtr->getTechnique(0)->getPass(0)->createTextureUnitState(textureFilename.getValue());
    }
    t = matPtr->getTechnique(0)->getPass(0)->createTextureUnitState(texture->getName() );
    t->setColourOperationEx(Ogre::LBX_BLEND_MANUAL, Ogre::LBS_TEXTURE, Ogre::LBS_CURRENT,
            Ogre::ColourValue::White, Ogre::ColourValue::White,
            blendingFactor.getValue());
    t->setTextureAddressingMode(Ogre::TextureUnitState::TAM_CLAMP);
    t->setProjectiveTexturing(true,&projectionCamera);

    matPtr->compile();
    return matPtr;
}


void OgrePlanarReflectionMaterial::initVisual()
{
    assert(mSceneMgr != NULL);

    using namespace Ogre;
    using namespace sofa::core::objectmodel;
    using namespace sofa::component::visualmodel;

    objectsToHide.clear();

    //Search from the root OgreVisualModels which share the same tag as this object.
    BaseContext* context = this->getContext();
    Tag reflectionTag = *this->getTags().begin();
    std::string reflectionTagString = reflectionTag;
    if( ! reflectionTagString.empty())
    {
        context->get< OgreVisualModel >( &objectsToHide,reflectionTag, BaseContext::SearchRoot);
    }

    if( !mPlane)
    {
        mPlane = createReflectionPlane();
    }

    if( !mCamera)
    {
        mCamera = createReflectionCamera();
    }
    if( mRttPtr.isNull())
    {
        mRttPtr = createRenderTargetTexture(*mCamera);
        mRttPtr->getBuffer()->getRenderTarget()->addListener(this);
    }
    if(!mDebugPlaneEntity)
    {
        mDebugPlaneEntity = createDebugPlaneEntity(*mPlane);
    }
    if( mMatPtr.isNull() )
    {
        mMatPtr = createMaterial(mRttPtr,*mCamera);
    }
    if( !mPlaneNode)
    {
        mPlaneNode = createPlaneSceneNode(*mPlane);
    }
}

void OgrePlanarReflectionMaterial::reinit()
{
    using namespace Ogre;
    mRttPtr->getBuffer()->getRenderTarget()->removeListener(this);

    MaterialManager::getSingleton().remove(mMatPtr->getName() );
    TextureManager::getSingleton().remove(mRttPtr->getName() );
    mPlaneNode->detachObject(mPlane);
    if( mDebugPlaneEntity->isAttached() )
    {
        mPlaneNode->detachObject(mDebugPlaneEntity);
    }

    mSceneMgr->destroyEntity(mPlane->getName());
    mSceneMgr->destroyEntity(mDebugPlaneEntity->getName());

    mPlane = createReflectionPlane();
    updateReflectionCamera(*mPlane);
    mRttPtr = createRenderTargetTexture(*mCamera);
    mDebugPlaneEntity = createDebugPlaneEntity(*mPlane);
    mMatPtr = createMaterial(mRttPtr,*mCamera);

    mPlaneNode->attachObject(mPlane);
    if( debugDraw.getValue() )
    {
        mDebugPlaneEntity->setMaterial(mMatPtr);
        mPlaneNode->attachObject(mDebugPlaneEntity);

    }
    mRttPtr->getBuffer()->getRenderTarget()->addListener(this);
    sofa::defaulttype::Vector3 dist = normalReflectionPlane.getValue() * distanceToOrigin.getValue();
    mPlaneNode->setPosition(dist.elems[0],dist.elems[1],dist.elems[2]);
}

void OgrePlanarReflectionMaterial::drawVisual()
{
}

void OgrePlanarReflectionMaterial::updateVisual()
{
}



void OgrePlanarReflectionMaterial::updateReflectionCamera(const Ogre::MovablePlane& plane)
{
    Ogre::Camera* sofaCamera = mSceneMgr->getCamera("sofaCamera");
    assert(sofaCamera);

    //mCamera->setNearClipDistance(sofaCamera->getNearClipDistance());
    mCamera->setFarClipDistance(sofaCamera->getFarClipDistance());
    mCamera->setAspectRatio(sofaCamera->getViewport()->getActualWidth() /
            sofaCamera->getViewport()->getActualHeight());
    mCamera->setFOVy(sofaCamera->getFOVy());
    mCamera->setOrientation(sofaCamera->getOrientation());
    mCamera->setPosition(sofaCamera->getPosition());

    mCamera->enableReflection(&plane);
    mCamera->enableCustomNearClipPlane(&plane);

}

void OgrePlanarReflectionMaterial::updatePlaneSceneNode()
{

}

void OgrePlanarReflectionMaterial::preRenderTargetUpdate(const Ogre::RenderTargetEvent& evt)
{
    if( evt.source == mRttPtr->getBuffer()->getRenderTarget() )
    {
        updateReflectionCamera(*mPlane);
        helper::vector< OgreVisualModel* >::iterator it;
        for ( it = objectsToHide.begin(); it != objectsToHide.end(); ++it )
        {
            (*it)->setVisible(false);
        }

        if( mDebugPlaneEntity->isAttached() )
        {
            mDebugPlaneEntity->setVisible(false);
        }
    }
}

void OgrePlanarReflectionMaterial::postRenderTargetUpdate(const Ogre::RenderTargetEvent& evt)
{
    if( evt.source == mRttPtr->getBuffer()->getRenderTarget() )
    {
        helper::vector< OgreVisualModel* >::iterator it;
        for ( it = objectsToHide.begin(); it != objectsToHide.end(); ++it )
        {
            (*it)->setVisible(true);
            (*it)->setVisible(true);
        }
        if( mDebugPlaneEntity->isAttached() )
        {
            mDebugPlaneEntity->setVisible(true);
        }
    }
}

}
}
}
