#ifndef SOFA_GUI_QT_VIEWER_QTOGRE_OGREREFLECTIONTEXTURE_H
#define SOFA_GUI_QT_VIEWER_QTOGRE_OGREREFLECTIONTEXTURE_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/VisualModel.h>
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

class OgreReflectionTexture : public core::VisualModel, public Ogre::RenderTargetListener
{
public:
    SOFA_CLASS(OgreReflectionTexture,core::objectmodel::BaseObject);

    typedef sofa::defaulttype::Vec<2,int> Vec2i;

    OgreReflectionTexture();
    virtual ~OgreReflectionTexture();

    void initVisual();
    void reinit();

    // VisualModel methods

    void drawVisual();
    void updateVisual();


    void setSceneManager( Ogre::SceneManager& sceneMgr);

    //RenderTargetListenerMethods
    void preRenderTargetUpdate(const Ogre::RenderTargetEvent& evt);
    void postRenderTargetUpdate(const Ogre::RenderTargetEvent &evt);

    Ogre::MaterialPtr getMaterialPtr(const std::string& name) const { return mMatPtr->clone(name); }

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
    Ogre::SceneManager* mSceneMgr;
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
