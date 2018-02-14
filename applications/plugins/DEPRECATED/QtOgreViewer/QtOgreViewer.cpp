/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <OgreConfigFile.h>
#if OGRE_VERSION >= 0x010700
#include "HelperLogics.h"
#endif

#include "QtOgreViewer.h"
#include "DotSceneLoader.h"

#include "OgreVisualModel.h"
#include "OgreViewerSetting.h"
#include "DrawToolOGRE.h"
#include "PropagateOgreSceneManager.h"


#include <sofa/gui/GUIManager.h>
#include <sofa/gui/qt/RealGUI.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glu.h>

#include <sofa/helper/Factory.inl>

#include <sofa/simulation/Simulation.h>

#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/ObjectFactory.h>

#include <qapplication.h>

#ifdef __linux__
#include <X11/Xlib.h>
#endif

#ifdef SOFA_QT4
#ifdef __linux__
#include <QX11Info>
#endif
#include <QWidget>
#include <QLabel>
#include <QToolBox>
#include <QVBoxLayout>
#else
#include <qwidget.h>
#include <qlabel.h>
#include <qtoolbox.h>
#endif

#include <fstream>


//#define SHOW_CONFIGDIALOG
#if OGRE_PLATFORM == OGRE_PLATFORM_APPLE
#include <CoreFoundation/CoreFoundation.h>
// This function will locate the path to our application on OS X,
// unlike windows you can not rely on the curent working directory
// for locating your configuration files and resources.
std::string macBundlePath()
{
    char path[1024];
    CFBundleRef mainBundle = CFBundleGetMainBundle();
    assert(mainBundle);

    CFURLRef mainBundleURL = CFBundleCopyBundleURL(mainBundle);
    assert(mainBundleURL);

    CFStringRef cfStringRef = CFURLCopyFileSystemPath( mainBundleURL, kCFURLPOSIXPathStyle);
    assert(cfStringRef);

    CFStringGetCString(cfStringRef, path, 1024, kCFStringEncodingASCII);

    CFRelease(mainBundleURL);
    CFRelease(cfStringRef);

    return std::string(path);
}
#endif

namespace sofa
{

namespace gui
{

namespace qt
{

namespace viewer
{
SOFA_DECL_CLASS (QtOgreViewer);
helper::SofaViewerCreator<QtOgreViewer> QtOgreViewer_class("ogre",false);
int QtOGREGUIClass = GUIManager::RegisterGUI ( "ogre", &qt::RealGUI::CreateGUI, &qt::RealGUI::InitGUI, 1 );

using sofa::simulation::Simulation;
using namespace sofa::helper::system::thread;

//Application principale
QtOgreViewer::QtOgreViewer( QWidget *parent, const char *name )
    : QGLWidget( parent, name )
{
    dirLight = pointLight = spotLight = NULL;
    this->setName("ogre");
#ifdef SOFA_QT4
    setWindowFlags(Qt::WindowStaysOnTopHint);
#else
    setWFlags(Qt::WNoAutoErase);
    setWFlags(Qt::WStyle_StaysOnTop);
#endif

    //*************************************************************************
    // Ogre Init
    //*************************************************************************//
    mRoot = NULL;
    mRenderWindow = NULL;
    mSceneMgr = NULL;
    mVp = NULL;
    drawUtility=NULL;
    //*************************************************************************
    // Interface Init
    //*************************************************************************//
    tabLights=0;
    tabCompositor=0;

    _waitForRender = false;

    capture.setCounter();
    m_isControlPressed = false;
    m_mouseLeftPressed = false;
    m_mouseRightPressed = false;
    m_mouseMiddlePressed = false;
    m_mousePos = QPoint(0, 0);
    m_mousePressPos = QPoint(0, 0);
    m_mTranslateVector = Ogre::Vector3::ZERO;
    m_mRotX = m_mRotY = 0.0f;
    m_mMoveSpeed = 10;
    m_mRotateSpeed = 36;
    _background = 0;
    _mouseInteractorMoving = false;
    _mouseInteractorSavedPosX = 0;
    _mouseInteractorSavedPosY = 0;
    number_visualModels=0;
    _video = false;
    showAxis=false;

#if OGRE_PLATFORM == OGRE_PLATFORM_APPLE
    mResourcePath = macBundlePath() + "/Contents/Resources/";
#else
    mResourcePath = "";
#endif

    //Default files
    sofa::helper::system::FileRepository repository;
    sceneName = sofa::helper::system::DataRepository.getFile("config/default.scene");

    // #if defined(SOFA_GPU_CUDA)
    // 	    mycudaInit(0);
    // #endif
}

QtOgreViewer::~QtOgreViewer()
{
    vparams->drawTool()->clear();

    if(mRoot != NULL)
    {
        mRoot->shutdown();
        delete mRoot;
        mRoot = NULL;
    }
};

void QtOgreViewer::setup()
{

    std::cout << "setupResources()...";
    setupResources();
    std::cout << "done." << std::endl;

    std::cout << "setupConfiguration()...";
    setupConfiguration();
    std::cout << "done." << std::endl;
    //if(!configure()) {std::cout << "No configuration \a\n";exit(0);}

    std::cout << "setupView()...";
    setupView();
    std::cout << "done." << std::endl;


    Ogre::TextureManager::getSingleton().setDefaultNumMipmaps(5);

    std::cout << "loadResources()...";
    loadResources();
    std::cout << "done." << std::endl;
}


//*****************************************************************************************
//Setup Ogre
void QtOgreViewer::setupResources()
{

    Ogre::String pluginsPath;
    Ogre::String configPath;
    Ogre::String ogreLog;
    Ogre::String ogreCfg;

#ifndef OGRE_STATIC_LIB
    ogreLog="config/Ogre.log";
    if ( !sofa::helper::system::DataRepository.findFile ( ogreLog ) )
    {
        std::string fileToBeCreated = sofa::helper::system::DataRepository.getFirstPath() + "/" + ogreLog;
        std::ofstream ofile(fileToBeCreated.c_str());
        ofile << "";
        ofile.close();
    }
    ogreLog = sofa::helper::system::DataRepository.getFile("config/Ogre.log");
    ogreCfg = sofa::helper::system::DataRepository.getFile("config/ogre.cfg");
#ifdef _WINDOWS
#ifdef NDEBUG
    pluginsPath = sofa::helper::system::DataRepository.getFile("config/plugins_win.cfg");
#else
    pluginsPath = sofa::helper::system::DataRepository.getFile("config/plugins_win_d.cfg");
#endif //NDEBUG
#else
    pluginsPath = sofa::helper::system::DataRepository.getFile("config/plugins.cfg");

#endif
#endif // OGRE_STATIC_LIB
    std::cout << "pluginsPath: " << pluginsPath << std::endl;
    std::cout << "ogreCfg: " << ogreCfg << std::endl;
    std::cout << "ogreLog: " << ogreLog << std::endl;
    mRoot = new Ogre::Root(pluginsPath, ogreCfg, ogreLog);
    if( mRoot == NULL)
    {
        std::cerr << "Failed to create Ogre::Root" << std::endl;

    }
    //        Ogre::LogManager::getSingleton().setLogDetail(Ogre::LL_LOW);
#ifndef WIN32
    Ogre::ResourceGroupManager::getSingleton().addResourceLocation("/","FileSystem","General");
#endif
    const std::vector< std::string > &paths=sofa::helper::system::DataRepository.getPaths();
    for (unsigned int i=0; i<paths.size(); ++i)  Ogre::ResourceGroupManager::getSingleton().addResourceLocation(paths[i] ,"FileSystem","General");

    Ogre::ResourceGroupManager::getSingleton().addResourceLocation(helper::system::SetDirectory::GetCurrentDir(),"FileSystem","General");
    Ogre::ResourceGroupManager::getSingleton().addResourceLocation(sofa::helper::system::DataRepository.getFirstPath() +"/config","FileSystem","General");
    Ogre::ResourceGroupManager::getSingleton().addResourceLocation(sofa::helper::system::DataRepository.getFirstPath() +"/textures","FileSystem","General");
    Ogre::ResourceGroupManager::getSingleton().addResourceLocation(sofa::helper::system::DataRepository.getFirstPath() +"/materials/programs","FileSystem","General");
    Ogre::ResourceGroupManager::getSingleton().addResourceLocation(sofa::helper::system::DataRepository.getFirstPath() +"/materials/scripts","FileSystem","General");
    Ogre::ResourceGroupManager::getSingleton().addResourceLocation(sofa::helper::system::DataRepository.getFirstPath() +"/materials/textures","FileSystem","General");

    std::string resourceCfgFile = "config/ogreResources.cfg";
    if( sofa::helper::system::DataRepository.findFile(resourceCfgFile) )
    {
        resourceCfgFile = sofa::helper::system::DataRepository.getFile("config/ogreResources.cfg");
        Ogre::ConfigFile cf;
        cf.load(resourceCfgFile);

        // Go through all sections & settings in the file
        Ogre::ConfigFile::SectionIterator seci = cf.getSectionIterator();

        Ogre::String secName, typeName, archName;
        while (seci.hasMoreElements())
        {
            secName = seci.peekNextKey();
            Ogre::ConfigFile::SettingsMultiMap *settings = seci.getNext();
            Ogre::ConfigFile::SettingsMultiMap::iterator i;
            for (i = settings->begin(); i != settings->end(); ++i)
            {
                typeName = i->first;
                archName = i->second;
#if OGRE_PLATFORM == OGRE_PLATFORM_APPLE
                // OS X does not set the working directory relative to the app,
                // In order to make things portable on OS X we need to provide
                // the loading with it's own bundle path location
                if (!StringUtil::startsWith(archName, "/", false)) // only adjust relative dirs
                    archName = String(macBundlePath() + "/" + archName);
#endif
                Ogre::ResourceGroupManager::getSingleton().addResourceLocation(
                    archName, typeName, secName);
            }
        }
    }
}



void QtOgreViewer::setupConfiguration(void)
{
#ifdef SHOW_CONFIGDIALOG
    mRoot->showConfigDialog();
#else
    //Rendering Device - will pick first available render device
#if OGRE_VERSION >= 0x010700
    Ogre::RenderSystem* mRenderSystem = *(mRoot->getAvailableRenderers().begin());
#else
    Ogre::RenderSystemList::iterator pRend = mRoot->getAvailableRenderers()->begin();
    Ogre::RenderSystem* mRenderSystem = *pRend;
#endif
    if ( mRenderSystem )
    {
        std::cout << "Render System set :" << std::endl;
        std::cout << mRenderSystem->getName() << std::endl;
    }
    else
    {
        std::cout << "Failed to retrieve a valid Render System !" << std::endl;
    }
    mRenderSystem->setShadingType(Ogre::SO_PHONG);

    //RenderSystem
    mRoot->setRenderSystem(mRenderSystem);
    mRenderSystem->validateConfigOptions();
#endif
    mRoot->initialise(false, "SOFA - OGRE");
}

void QtOgreViewer::drawSceneAxis() const
{
    defaulttype::Vector3 centerAxis(0,0,0);
    static defaulttype::Vector3 axisX(1,0,0);
    static defaulttype::Vector3 axisY(0,1,0);
    static defaulttype::Vector3 axisZ(0,0,1);
    const Ogre::Real l=size_world.length();

    const defaulttype::Vector3 posX=centerAxis+axisX*l*0.6;
    const defaulttype::Vector3 posY=centerAxis+axisY*l*0.6;
    const defaulttype::Vector3 posZ=centerAxis+axisZ*l*0.6;

    Ogre::Entity *axisXEntity;
    Ogre::Entity *axisYEntity;
    Ogre::Entity *axisZEntity;
    if (!mSceneMgr->hasEntity("axisX"))
    {
        Ogre::MaterialPtr axisMaterial = Ogre::MaterialManager::getSingleton().create("AxisViewerMaterial", "General");
        axisMaterial->setSelfIllumination(0.25,0.25,0.25);

        axisXEntity= mSceneMgr->createEntity( "axisX", "mesh/X.mesh" );
        axisYEntity= mSceneMgr->createEntity( "axisY", "mesh/Y.mesh" );
        axisZEntity= mSceneMgr->createEntity( "axisZ", "mesh/Z.mesh" );

        axisXEntity->setMaterial(axisMaterial);
        axisYEntity->setMaterial(axisMaterial);
        axisZEntity->setMaterial(axisMaterial);
    }
    else
    {
        axisXEntity=mSceneMgr->getEntity("axisX");
        axisYEntity=mSceneMgr->getEntity("axisY");
        axisZEntity=mSceneMgr->getEntity("axisZ");
    }

    if (!axisXEntity->isAttached())
    {
        nodeX->attachObject(axisXEntity);
        nodeX->setPosition(0,0,0);
        const Ogre::Real scale=l*0.075;
        nodeX->setScale(scale, scale, scale);
        nodeX->setPosition(posX[0], posX[1], posX[2]);
    }
    if (!axisYEntity->isAttached())
    {
        nodeY->attachObject(axisYEntity);
        nodeY->setPosition(0,0,0);
        const Ogre::Real scale=l*0.075;
        nodeY->setScale(scale, scale, scale);
        nodeY->setPosition(posY[0], posY[1], posY[2]);
    }
    if (!axisZEntity->isAttached())
    {
        nodeZ->attachObject(axisZEntity);
        nodeZ->setPosition(0,0,0);
        const Ogre::Real scale=l*0.075;
        nodeZ->setScale(scale, scale, scale);
        nodeZ->setPosition(posZ[0], posZ[1], posZ[2]);
    }

    vparams->drawTool()->drawArrow(centerAxis,posX,
            l*0.005, defaulttype::Vec<4,float>(1.0f,0.0f,0.0f,1.0f));
    vparams->drawTool()->drawArrow(centerAxis,posY,
            l*0.005, defaulttype::Vec<4,float>(0.0f,1.0f,0.0f,1.0f));
    vparams->drawTool()->drawArrow(centerAxis,posZ,
            l*0.005, defaulttype::Vec<4,float>(0.0f,0.0f,1.0f,1.0f));

}

//*****************************************************************************************
//called to redraw the window
void QtOgreViewer::updateIntern()
{
    if(mRenderWindow == NULL) setupView();
    if(mRenderWindow != NULL && !_waitForRender)
    {
        _waitForRender=true;
        // 	      mRoot->_fireFrameStarted();

        if (groot->getContext()->getShowWireFrame() && mCamera->getPolygonMode()!= Ogre::PM_WIREFRAME)
            mCamera->setPolygonMode(Ogre::PM_WIREFRAME);
        if (!groot->getContext()->getShowWireFrame() && mCamera->getPolygonMode()!= Ogre::PM_SOLID)
            mCamera->setPolygonMode(Ogre::PM_SOLID);

        //Not optimal, clear all the datas
        vparams->drawTool()->clear();

        static bool initViewer=false;
        if (initViewer)
        {
            sofa::simulation::getSimulation()->initTextures( simulation::getSimulation()->getVisualRoot());
            sofa::simulation::getSimulation()->draw(vparams,groot);
            sofa::simulation::getSimulation()->draw(vparams,simulation::getSimulation()->getVisualRoot());

            if (!initCompositing.empty())
            {
                //reset the compositor selection
                for (unsigned int j=0; j<compositorSelection.size(); ++j) compositorSelection[j]->setChecked(false);

                for (unsigned int i=0; i<initCompositing.size(); ++i)
                {
                    for (unsigned int j=0; j<compositorSelection.size(); ++j)
                    {
                        if (compositorSelection[j]->text().ascii() == initCompositing[i])
                        {
                            compositorSelection[j]->setChecked(true);
                        }
                    }
                }
                initCompositing.clear();
            }

        }
        else
        {
            initViewer=true;
        }

        if (showAxis) drawSceneAxis();
        //Remove previous mesh and entity
        if (mSceneMgr->hasEntity("drawUtilityENTITY"))
        {
            mSceneMgr->destroyEntity("drawUtilityENTITY");
        }
        if (!Ogre::MeshManager::getSingleton().getByName("drawUtilityMESH").isNull())
        {
            Ogre::MeshManager::getSingleton().remove("drawUtilityMESH");
        }
        //If the drawUtility has something to display, we convert to Mesh
        if (drawUtility->getNumSections())
        {
            Ogre::MeshPtr ogreMesh = drawUtility->convertToMesh("drawUtilityMESH", "General");
            Ogre::Entity *e = mSceneMgr->createEntity("drawUtilityENTITY", ogreMesh->getName());
            mSceneMgr->getRootSceneNode()->attachObject(e);
        }
        if (_video)
        {
#ifdef CAPTURE_PERIOD
            static int counter = 0;
            if ((counter++ % CAPTURE_PERIOD)==0)
            {
#endif
                std::string captureFilename = capture.findFilename();
                this->mRenderWindow->writeContentsToFile(captureFilename);
                std::cout << "Saved "<<this->mRenderWindow->getWidth()<<"x"<<this->mRenderWindow->getHeight()<<" screen image to "<<captureFilename<<std::endl;
#ifdef CAPTURE_PERIOD
            }
#endif

        }

        moveCamera();
        mRenderWindow->update();
        mRoot->renderOneFrame();
        if (_waitForRender)
        {
            _waitForRender = false;
        }

    }

}

void QtOgreViewer::updateViewerParameters()
{
    mSceneMgr->setAmbientLight(Ogre::ColourValue(ambient[0]->Value(),ambient[1]->Value(),ambient[2]->Value(),1));

    for (unsigned int i=0; i<dirLightOgreWidget.size(); ++i)
    {
        dirLightOgreWidget[i]->updateLight();
    }
    for (unsigned int i=0; i<pointLightOgreWidget.size(); ++i)
    {
        pointLightOgreWidget[i]->updateLight();
    }
    for (unsigned int i=0; i<spotLightOgreWidget.size(); ++i)
    {
        spotLightOgreWidget[i]->updateLight();
    }
    updateIntern();
}

//******************************Qt paint***********************************

void QtOgreViewer::showEvent(QShowEvent *e)
{
    if (!mRoot) {setup();}
    QGLWidget::showEvent(e);
}

void QtOgreViewer::initializeGL()
{
#ifdef SOFA_HAVE_GLEW
    glewInit();
    if (!GLEW_ARB_multitexture)
        std::cerr << "Error: GL_ARB_multitexture not supported\n";
#endif
}

void QtOgreViewer::paintGL()
{
    updateIntern();
    emit( redrawn() );
}

//*****************************************************************************************
//Initialize the rendering window: create a sofa simulation, Ogre window...
void QtOgreViewer::setupView()
{
    Ogre::NameValuePairList params;
    //The external windows handle parameters are platform-specific
    Ogre::String externalWindowHandleParams;

#ifdef SOFA_QT4
#if defined(Q_WS_MAC) || defined(WIN32)

    externalWindowHandleParams = Ogre::StringConverter::toString((unsigned int)(this->parentWidget()->winId()));
#else
    //poslong:posint:poslong:poslong (display*:screen:windowHandle:XVisualInfo*) for GLX - According to Ogre Docs
    QX11Info info = x11Info();
    externalWindowHandleParams  = Ogre::StringConverter::toString((unsigned long)(info.display()));
    externalWindowHandleParams += ":";
    externalWindowHandleParams += Ogre::StringConverter::toString((unsigned int)(info.screen()));
    externalWindowHandleParams += ":";
    externalWindowHandleParams += Ogre::StringConverter::toString((unsigned long)(winId()));
    externalWindowHandleParams += ":";
    externalWindowHandleParams += Ogre::StringConverter::toString((unsigned long)(info.visual()));
#endif
    //Add the extrenal window handle parameters to the existing params set.
    params["externalWindowHandle"] = externalWindowHandleParams;
#else
    Display* display = qt_xdisplay();
    int screen = qt_xscreen();

    params["parentWindowHandle"] =
        Ogre::StringConverter::toString ((unsigned long)display) +
        ":" + Ogre::StringConverter::toString ((unsigned long)screen) +
        ":" + Ogre::StringConverter::toString ((unsigned long)parentWidget()->winId());

#endif
    //Finally create our window.
    std::cout<< "createRenderWindow :" << width() << "x" << height() << "...";
    mRenderWindow = mRoot->createRenderWindow( "OgreWindow", width(), height(), false, &params );
    std::cout<< "done.";


    mRenderWindow->setActive(true);
    WId ogreWinId = 0x0;

    mRenderWindow->getCustomAttribute( "WINDOW", &ogreWinId );
    //QWidget::create(ogreWinId);


#ifdef SOFA_QT4
    setAttribute( Qt::WA_PaintOnScreen, true );
    setAttribute( Qt::WA_NoBackground );
#endif
    _beginTime = CTime::getTime();
}




//*****************************************************************************************
//Initialize Sofa with the scene, and load the components to Ogre

void QtOgreViewer::createScene()
{
    using namespace Ogre;


    if (groot==NULL)
    {
        groot = simulation::getSimulation()->createNewGraph("");
    }

    DotSceneLoader loader;
    std::string::size_type pos_point = this->sceneFileName.rfind(".");
    if (pos_point != std::string::npos)
    {
        sceneFile = this->sceneFileName;
        sceneFile.resize(pos_point); sceneFile += ".scene";
        if ( sofa::helper::system::DataRepository.findFile(sceneFile))
            sceneName = sofa::helper::system::DataRepository.getFile(sceneFile);
    }

    loader.parseDotScene(sceneName, "General", NULL); //mSceneMgr);
    mSceneMgr = loader.getSceneManager();



    mSceneMgr->setAmbientLight(loader.environment.ambientColour);
    ambient[0]->setValue(loader.environment.ambientColour[0]);
    ambient[1]->setValue(loader.environment.ambientColour[1]);
    ambient[2]->setValue(loader.environment.ambientColour[2]);


    if (mSceneMgr->hasCamera("sofaCamera"))
        mCamera = mSceneMgr->getCamera("sofaCamera");
    else
    {
        // Create the camera
        mCamera = mSceneMgr->createCamera("sofaCamera");
        // Position it at 50 in Z direction
        mCamera->setPosition(Ogre::Vector3(0,1,0));
        // Look back along -Z
        mCamera->lookAt(Ogre::Vector3(0,0,1));
        mCamera->setNearClipDistance(loader.environment.nearClipDistance);
        mCamera->setFarClipDistance(loader.environment.farClipDistance);
    }
    //Always yaw around the camera Y axis.
    fixedAxis = Ogre::Vector3::UNIT_Y;
    mCamera->setFixedYawAxis(true, fixedAxis);

    Ogre::SceneNode* camNode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
    camNode->attachObject(mCamera);

    zeroNode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
    nodeX = mSceneMgr->getRootSceneNode()->createChildSceneNode();
    nodeX->rotate(Ogre::Vector3::UNIT_Z, Ogre::Radian(Ogre::Degree(-90)));
    nodeY = mSceneMgr->getRootSceneNode()->createChildSceneNode();
    nodeY->rotate(Ogre::Vector3::UNIT_Y, Ogre::Radian(Ogre::Degree(180)));
    nodeZ = mSceneMgr->getRootSceneNode()->createChildSceneNode();
    nodeZ->rotate(Ogre::Vector3::UNIT_X, Ogre::Radian(Ogre::Degree(90)));


    mCamera->setAutoTracking(true, zeroNode);
    mCamera->setCastShadows(false);
    // Create one viewport, entire window
    if (mVp)
    {
        mRenderWindow->removeViewport(0);
    }
    mVp = mRenderWindow->addViewport(mCamera);
    mVp->setBackgroundColour(loader.environment.backgroundColour);
    mVp->setDimensions(0.0 ,0.0 , 1.0, 1.0);
    showEntireScene();

    if (!drawUtility)
    {
        drawUtility = (Ogre::ManualObject *) mSceneMgr->createMovableObject("drawUtility","ManualObject");
        drawUtility->setDynamic(true);
        drawUtility->setCastShadows(true);
    }

    //Add Directional light in the GUI
    for (unsigned int i=0; i<loader.directionalLights.size(); ++i)
    {
        if (dirLightOgreWidget.size() > i && mSceneMgr->hasLight(loader.directionalLights[i]))
            dirLightOgreWidget[i]->restoreLight(loader.directionalLights[i]);
        else
            addDirLight(loader.directionalLights[i]);
    }
    numDirLight->setValue(loader.directionalLights.size());
    //Add Point light in the GUI
    for (unsigned int i=0; i<loader.pointLights.size(); ++i)
    {
        if (pointLightOgreWidget.size() > i && mSceneMgr->hasLight(loader.pointLights[i]))
            pointLightOgreWidget[i]->restoreLight(loader.pointLights[i]);
        else
            addPointLight(loader.pointLights[i]);
    }
    numPointLight->setValue(loader.pointLights.size());
    //Add Spot light in the GUI
    for (unsigned int i=0; i<loader.spotLights.size(); ++i)
    {
        if (spotLightOgreWidget.size() > i && mSceneMgr->hasLight(loader.spotLights[i]))
            spotLightOgreWidget[i]->restoreLight(loader.spotLights[i]);
        else
            addSpotLight(loader.spotLights[i]);
    }
    numSpotLight->setValue(loader.spotLights.size());


    //By default, we don't use shadows
    setLightActivated(false);

    //************************************************************************************************
    // Alter the camera aspect ratio to match the viewport
    mCamera->setAspectRatio(Real(mVp->getActualWidth()) / Real(mVp->getActualHeight()));
}

void QtOgreViewer::createTextures()
{

    using namespace Ogre;

    TexturePtr tex = TextureManager::getSingleton().createManual(
            "HalftoneVolume",
            "General",
            TEX_TYPE_3D,
            64,64,64,
            0,
            PF_A8
            );

    HardwarePixelBufferSharedPtr ptr = tex->getBuffer(0,0);
    ptr->lock(HardwareBuffer::HBL_DISCARD);
    const PixelBox &pb = ptr->getCurrentLock();
    Ogre::uint8 *data = static_cast<Ogre::uint8*>(pb.data);

    size_t height = pb.getHeight();
    size_t width = pb.getWidth();
    size_t depth = pb.getDepth();
    size_t rowPitch = pb.rowPitch;
    size_t slicePitch = pb.slicePitch;

    for (size_t z = 0; z < depth; ++z)
    {
        for (size_t y = 0; y < height; ++y)
        {
            for(size_t x = 0; x < width; ++x)
            {
                float fx = 32-(float)x+0.5f;
                float fy = 32-(float)y+0.5f;
                float fz = 32-((float)z)/3+0.5f;
                float distanceSquare = fx*fx+fy*fy+fz*fz;
                data[slicePitch*z + rowPitch*y + x] =  0x00;
                if (distanceSquare < 1024.0f)
                    data[slicePitch*z + rowPitch*y + x] +=  0xFF;
            }
        }
    }
    ptr->unlock();

    Ogre::Viewport *vp = mVp;

    TexturePtr tex2 = TextureManager::getSingleton().createManual(
            "DitherTex",
            "General",
            TEX_TYPE_2D,
            vp->getActualWidth(),vp->getActualHeight(),1,
            0,
            PF_A8
            );

    HardwarePixelBufferSharedPtr ptr2 = tex2->getBuffer(0,0);
    ptr2->lock(HardwareBuffer::HBL_DISCARD);
    const PixelBox &pb2 = ptr2->getCurrentLock();
    Ogre::uint8 *data2 = static_cast<Ogre::uint8*>(pb2.data);

    size_t height2 = pb2.getHeight();
    size_t width2 = pb2.getWidth();
    size_t rowPitch2 = pb2.rowPitch;

    for (size_t y = 0; y < height2; ++y)
    {
        for(size_t x = 0; x < width2; ++x)
        {
            data2[rowPitch2*y + x] = Ogre::Math::RangeRandom(64.0,192);
        }
    }

    ptr2->unlock();
}

void QtOgreViewer::createEffects()
{
    /// Motion blur effect
    Ogre::CompositorPtr comp3 = Ogre::CompositorManager::getSingleton().create(
            "MotionBlur", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME
            );
    {
        Ogre::CompositionTechnique *t = comp3->createTechnique();
        {
            Ogre::CompositionTechnique::TextureDefinition *def = t->createTextureDefinition("scene");
            def->width = 0;
            def->height = 0;
            def->formatList.push_back(Ogre::PF_R8G8B8);
        }
        {
            Ogre::CompositionTechnique::TextureDefinition *def = t->createTextureDefinition("sum");
            def->width = 0;
            def->height = 0;
            def->formatList.push_back(Ogre::PF_R8G8B8);
        }
        {
            Ogre::CompositionTechnique::TextureDefinition *def = t->createTextureDefinition("temp");
            def->width = 0;
            def->height = 0;
            def->formatList.push_back(Ogre::PF_R8G8B8);
        }
        /// Render scene
        {
            Ogre::CompositionTargetPass *tp = t->createTargetPass();
            tp->setInputMode(Ogre::CompositionTargetPass::IM_PREVIOUS);
            tp->setOutputName("scene");
        }
        /// Initialisation pass for sum texture
        {
            Ogre::CompositionTargetPass *tp = t->createTargetPass();
            tp->setInputMode(Ogre::CompositionTargetPass::IM_PREVIOUS);
            tp->setOutputName("sum");
            tp->setOnlyInitial(true);
        }
        /// Do the motion blur
        {
            Ogre::CompositionTargetPass *tp = t->createTargetPass();
            tp->setInputMode(Ogre::CompositionTargetPass::IM_NONE);
            tp->setOutputName("temp");
            {
                Ogre::CompositionPass *pass = tp->createPass();
                pass->setType(Ogre::CompositionPass::PT_RENDERQUAD);
                pass->setMaterialName("Ogre/Compositor/Combine");
                pass->setInput(0, "scene");
                pass->setInput(1, "sum");
            }
        }
        /// Copy back sum texture
        {
            Ogre::CompositionTargetPass *tp = t->createTargetPass();
            tp->setInputMode(Ogre::CompositionTargetPass::IM_NONE);
            tp->setOutputName("sum");
            {
                Ogre::CompositionPass *pass = tp->createPass();
                pass->setType(Ogre::CompositionPass::PT_RENDERQUAD);
                pass->setMaterialName("Ogre/Compositor/Copyback");
                pass->setInput(0, "temp");
            }
        }
        /// Display result
        {
            Ogre::CompositionTargetPass *tp = t->getOutputTargetPass();
            tp->setInputMode(Ogre::CompositionTargetPass::IM_NONE);
            {
                Ogre::CompositionPass *pass = tp->createPass();
                pass->setType(Ogre::CompositionPass::PT_RENDERQUAD);
                pass->setMaterialName("Ogre/Compositor/MotionBlur");
                pass->setInput(0, "sum");
            }
        }
    }

    /// Heat vision effect
    Ogre::CompositorPtr comp4 = Ogre::CompositorManager::getSingleton().create(
            "HeatVision", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME
            );
    {
        Ogre::CompositionTechnique *t = comp4->createTechnique();
#if OGRE_VERSION >= 0x010700
        t->setCompositorLogicName("HeatVision");
#endif
        {
            Ogre::CompositionTechnique::TextureDefinition *def = t->createTextureDefinition("scene");
            def->width = 256;
            def->height = 256;
            def->formatList.push_back(Ogre::PF_R8G8B8);
        }
        {
            Ogre::CompositionTechnique::TextureDefinition *def = t->createTextureDefinition("temp");
            def->width = 256;
            def->height = 256;
            def->formatList.push_back(Ogre::PF_R8G8B8);
        }
        /// Render scene
        {
            Ogre::CompositionTargetPass *tp = t->createTargetPass();
            tp->setInputMode(Ogre::CompositionTargetPass::IM_PREVIOUS);
            tp->setOutputName("scene");
        }
        /// Light to heat pass
        {
            Ogre::CompositionTargetPass *tp = t->createTargetPass();
            tp->setInputMode(Ogre::CompositionTargetPass::IM_NONE);
            tp->setOutputName("temp");
            {
                Ogre::CompositionPass *pass = tp->createPass();
                pass->setType(Ogre::CompositionPass::PT_RENDERQUAD);
                pass->setIdentifier(0xDEADBABE); /// Identify pass for use in listener
                pass->setMaterialName("Fury/HeatVision/LightToHeat");
                pass->setInput(0, "scene");
            }
        }
        /// Display result
        {
            Ogre::CompositionTargetPass *tp = t->getOutputTargetPass();
            tp->setInputMode(Ogre::CompositionTargetPass::IM_NONE);
            {
                Ogre::CompositionPass *pass = tp->createPass();
                pass->setType(Ogre::CompositionPass::PT_RENDERQUAD);
                pass->setMaterialName("Fury/HeatVision/Blur");
                pass->setInput(0, "temp");
            }
        }
    }
}

void QtOgreViewer::updateCompositor(bool v)
{
    QCheckBox *check=(QCheckBox *) sender();
    Ogre::CompositorManager::getSingleton().setCompositorEnabled(mVp,check->text().ascii(), v);
    updateIntern();
}

void QtOgreViewer::registerCompositors()
{

    bool fullInit= (mCompositorNames.empty());
    //iterate through Compositor Managers resources and add name keys to menu
    Ogre::CompositorManager::ResourceMapIterator resourceIterator =
        Ogre::CompositorManager::getSingleton().getResourceIterator();


    // add all compositor resources to the view container
    while (resourceIterator.hasMoreElements())
    {
        Ogre::ResourcePtr resource = resourceIterator.getNext();
        const Ogre::String& compositorName = resource->getName();
        // Don't add base Ogre/Scene compositor to view
        if (compositorName == "Ogre/Scene")
            continue;

        if (fullInit) mCompositorNames.push_back(compositorName);

        int addPosition = -1;
        if (compositorName == "HDR")
        {
            // HDR must be first in the chain
            addPosition = 0;
        }
        try
        {
            Ogre::CompositorManager::getSingleton().addCompositor(mVp, compositorName, addPosition);
            Ogre::CompositorManager::getSingleton().setCompositorEnabled(mVp, compositorName, false);
        }
        catch (...)
        {
        }
    }
    if (fullInit)
    {
        QGridLayout* grid=(QGridLayout* ) compositorWidget->layout();
        const unsigned int maxRow=ceil(mCompositorNames.size()*0.5);
        for (unsigned int i=0; i<mCompositorNames.size(); ++i)
        {
            int nextColumn= (i >= maxRow)?1:0;
            QCheckBox *check = new QCheckBox(QString(mCompositorNames[i].c_str()),compositorWidget);
            grid->addWidget(check,i%(maxRow),nextColumn);
            compositorSelection.push_back(check);
            connect(check, SIGNAL(toggled(bool)),this, SLOT(updateCompositor(bool)));
        }
    }
}

void QtOgreViewer::showEntireScene()
{
    if (mSceneMgr == NULL) return;

    //In case new Visual Model appeared

    sofa::simulation::ogre::PropagateOgreSceneManager v(core::ExecParams::defaultInstance(),mSceneMgr);
    v.execute(groot);
    v.execute(sofa::simulation::getSimulation()->getVisualRoot());

    mSceneMgr->_updateSceneGraph (mCamera);
    //************************************************************************************************
    //Calculate the World Bounding Box
    //Scene Bounding Box
    simulation::getSimulation()->computeBBox(groot, sceneMinBBox.ptr(), sceneMaxBBox.ptr());
    simulation::getSimulation()->computeBBox(simulation::getSimulation()->getVisualRoot(), sceneMinBBox.ptr(), sceneMaxBBox.ptr(),false);

    size_world = Ogre::Vector3 (sceneMaxBBox[0] - sceneMinBBox[0],sceneMaxBBox[1] - sceneMinBBox[1],sceneMaxBBox[2] - sceneMinBBox[2]);
    float max = std::max(std::max(size_world.x,size_world.y),size_world.z);
    float min = std::min(std::min(size_world.x,size_world.y),size_world.z);

    _factorWheel=(size_world[0]+size_world[1]+size_world[2])/3.0;
    if (_factorWheel <= 0.0f) _factorWheel=1.0f;

    //frustrum
    if (min <= 0) return;

    mCamera->setFarClipDistance(Ogre::Real( 100*max));
    //	    mCamera->setNearClipDistance(Ogre::Real( min)); //cause a crash of wrong BBox

    //position
    Ogre::Vector3 camera_position((sceneMaxBBox[0]+sceneMinBBox[0])/2.0f,(sceneMaxBBox[1]+sceneMinBBox[1])/2.0f,(sceneMaxBBox[2]+sceneMinBBox[2])/2.0f);

    zeroNode->setPosition(camera_position);
    mCamera->setPosition(camera_position);
    mCamera->moveRelative(Ogre::Vector3(0.0,0.0,2*std::max(size_world[0],std::max( size_world[1], size_world[2]))
                                       ));
    return;
}



void QtOgreViewer::resize()
{
    if (mRenderWindow && mVp)
    {
        emit(resizeW(width())); emit(resizeH(height()));
        mRenderWindow->windowMovedOrResized();
        mVp->setDimensions(0,0, 1.0, 1.0);
        mCamera->setAspectRatio(Ogre::Real(width()) / Ogre::Real(height()));
        mRenderWindow->resize(width(), height());
        update();
    }
}

//*********************************Qt Control*******************************
void QtOgreViewer::resizeEvent(QResizeEvent *)
{
    resize();
}

void QtOgreViewer::setCameraMode(component::visualmodel::BaseCamera::CameraType mode)
{
    SofaViewer::setCameraMode(mode);

    switch (mode)
    {
    case core::visual::VisualParams::ORTHOGRAPHIC_TYPE:
    {
        const sofa::defaulttype::Vector3 center((sceneMinBBox+sceneMaxBBox)*0.5);

        Ogre::Vector3 cameraPosition(mCamera->getPosition());
        const sofa::defaulttype::Vector3 cameraPos(cameraPosition.x,cameraPosition.y,cameraPosition.z);

        SReal d=(center-cameraPos).norm();

        mCamera->setProjectionType(Ogre::PT_ORTHOGRAPHIC);
        SReal wRatio = mCamera->getOrthoWindowWidth()/mCamera->getOrthoWindowHeight();
        mCamera->setOrthoWindow(d*wRatio,d);
    }
    break;
    case core::visual::VisualParams::PERSPECTIVE_TYPE:
        mCamera->setProjectionType(Ogre::PT_PERSPECTIVE);
        break;
    }
}

//*****************************************************************************************
//Keyboard Events

void QtOgreViewer::keyPressEvent ( QKeyEvent * e )
{
    if( isControlPressed() ) // pass event to the scene data structure
    {
        //cerr<<"QtViewer::keyPressEvent, key = "<<e->key()<<" with Control pressed "<<endl;
        sofa::core::objectmodel::KeypressedEvent keyEvent(e->key());
        groot->propagateEvent(sofa::core::ExecParams::defaultInstance(), &keyEvent);
    }
    else  // control the GUI
    {
        switch(e->key())
        {
        case Qt::Key_A:
        {
            showAxis=!showAxis;
            if (!showAxis)
            {
                nodeX->detachAllObjects();
                nodeY->detachAllObjects();
                nodeZ->detachAllObjects();
            }
            break;
        }
        case Qt::Key_B:
        {
            if (_background == 0 && backgroundColour == Vector3())
                _background=2;
            else if (_background == 2 && backgroundColour == Vector3(1,1,1))
                _background = 1;
            else
                _background = (_background+1)%3;

            switch(_background)
            {
            case 0:
                mVp->setBackgroundColour(Ogre::ColourValue(backgroundColour[0],
                        backgroundColour[1],
                        backgroundColour[2]));
                break;
            case 1:
                mVp->setBackgroundColour(Ogre::ColourValue(0,0,0));
                break;
            case 2:
                mVp->setBackgroundColour(Ogre::ColourValue(1,1,1));
                break;
            }
            updateIntern();
            break;
        }
        case Qt::Key_C:
        {
            showEntireScene();
            break;
        }
        case Qt::Key_L:
            // --- Modify the shadow type
        {
            setLightActivated(!shadow);
            break;
        }
        default:
        {
            SofaViewer::keyPressEvent(e);
            e->ignore();
        }
        }
    }
    updateIntern();
}

void QtOgreViewer::keyReleaseEvent ( QKeyEvent * e )
{
    SofaViewer::keyReleaseEvent(e);
}

//*****************************************************************************************
//Mouse Events

void QtOgreViewer::mousePressEvent(QMouseEvent* evt)
{
    if (!updateInteractor(evt))
    {
        m_mousePos = evt->globalPos();

        if(evt->button() == Qt::LeftButton)
            m_mouseLeftPressed = true;
        if(evt->button() == Qt::RightButton)
            m_mouseRightPressed = true;
        if(evt->button() == Qt::MidButton)
            m_mouseMiddlePressed = true;
    }

}


void QtOgreViewer::mouseReleaseEvent(QMouseEvent* evt)
{
    if (!updateInteractor(evt))
    {
        m_mousePos = evt->globalPos();
        Q_UNUSED(evt);
        m_mouseLeftPressed = false;
        m_mouseRightPressed = false;
        m_mouseMiddlePressed = false;
        m_mRotX = m_mRotY = Ogre::Degree(0);
        m_mTranslateVector = Ogre::Vector3::ZERO;

    }


}

void QtOgreViewer::mouseMoveEvent(QMouseEvent* evt)
{

    if (!updateInteractor(evt))
    {
        Ogre::Vector3 pos_cam= mCamera->getPosition();
        const float dist = (zeroNode->getPosition()-pos_cam).length();
        const float factor = 0.001f;
        if( m_mouseRightPressed )
        {
            Ogre::Vector3 d(-(evt->globalX() - m_mousePos.x()) * dist*factor,(evt->globalY() - m_mousePos.y())* dist*factor,0);
            mCamera->moveRelative(d);
            pos_cam = mCamera->getPosition()-pos_cam;
            zeroNode->translate(pos_cam);
        }
        if( m_mouseMiddlePressed )
        {
            m_mTranslateVector.z -= (evt->globalY() - m_mousePos.y()) *_factorWheel*0.001;
            mCamera->moveRelative(m_mTranslateVector);
        }
        if( m_mouseLeftPressed )
        {
            m_mTranslateVector.x -=  (evt->globalX() - m_mousePos.x()) * dist*factor*5.0;
            m_mTranslateVector.y -= -(evt->globalY() - m_mousePos.y()) * dist*factor*5.0;
            mCamera->moveRelative(m_mTranslateVector);
            float new_dist = (zeroNode->getPosition()-mCamera->getPosition()).length();
            mCamera->moveRelative(Ogre::Vector3(0,0,dist-new_dist));

            Ogre::Vector3 p=mCamera->getPosition(); p.normalise();
            if (fixedAxis.absDotProduct(p) > 0.5)
            {
                fixedAxis=mCamera->getUp();
                mCamera->setFixedYawAxis(true, fixedAxis);
            }
        }

        m_mousePos = evt->globalPos();
        if (m_mouseLeftPressed || m_mouseMiddlePressed ||  m_mouseRightPressed ) this->update();
    }
}


void QtOgreViewer::wheelEvent(QWheelEvent* evt)
{
    SReal displacement=evt->delta()*_factorWheel*0.0005;
    m_mTranslateVector.z +=  displacement;
    mCamera->moveRelative(m_mTranslateVector);
    if (currentCamera->getCameraType() == core::visual::VisualParams::ORTHOGRAPHIC_TYPE)
    {
        SReal wRatio = mCamera->getOrthoWindowWidth()/mCamera->getOrthoWindowHeight();
        mCamera->setOrthoWindow(displacement*wRatio+mCamera->getOrthoWindowWidth(),
                displacement+mCamera->getOrthoWindowHeight());
    }
    this->update();
}



void QtOgreViewer::setScene(sofa::simulation::Node* scene, const char* filename, bool keepParams)
{
    //Create Compositors

    pickDone=false;
    numDirLight->setValue(0);
    numPointLight->setValue(0);
    numSpotLight->setValue(0);
    SofaViewer::setScene(scene, filename, keepParams);

    createScene();
    drawToolOGRE.setOgreObject(drawUtility);
    vparams->drawTool()->setPolygonMode(0,false); //Disable culling
    drawToolOGRE.setSceneMgr(mSceneMgr);
    updateIntern();
    resize();


    static bool firstTime = true;
    if (firstTime)
    {
        // Register the compositor logics
        // See comment in beginning of HelperLogics.h for explanation
        Ogre::CompositorManager& compMgr = Ogre::CompositorManager::getSingleton();
#if OGRE_VERSION >= 0x010700
        compMgr.registerCompositorLogic("GaussianBlur", new GaussianBlurLogic);
        compMgr.registerCompositorLogic("HDR", new HDRLogic);
        compMgr.registerCompositorLogic("HeatVision", new HeatVisionLogic);
#else
        compMgr.addCompositor(mVp,"GaussianBlur");
        compMgr.addCompositor(mVp,"HDR");
        compMgr.addCompositor(mVp,"HeatVision");
#endif
        firstTime = false;
    }
    createTextures();
    createEffects();
    registerCompositors();


}

bool QtOgreViewer::updateInteractor(QMouseEvent * e)
{
    if(e->state()&Qt::ShiftButton)
    {
        SofaViewer::mouseEvent(e);
        return true;
    }
    return false;
}

void QtOgreViewer::moveRayPickInteractor(int eventX, int eventY)
{
    Vec3d position, direction;
    if (currentCamera->getCameraType() == core::visual::VisualParams::PERSPECTIVE_TYPE)
    {
        sofa::defaulttype::Vec3d  p0, px, py, pz, px1, py1;
        GLint viewPort[4] = {0,0,width(), height()};
        Ogre::Matrix4 modelViewMatrix = mCamera->getViewMatrix().transpose();
        Ogre::Matrix4 projectionMatrix = mCamera->getProjectionMatrix();

        double modelViewMatrixGL[16] =
        {
            modelViewMatrix[0][0], modelViewMatrix[0][1], modelViewMatrix[0][2], modelViewMatrix[0][3],
            modelViewMatrix[1][0], modelViewMatrix[1][1], modelViewMatrix[1][2], modelViewMatrix[1][3],
            modelViewMatrix[2][0], modelViewMatrix[2][1], modelViewMatrix[2][2], modelViewMatrix[2][3],
            modelViewMatrix[3][0], modelViewMatrix[3][1], modelViewMatrix[3][2], modelViewMatrix[3][3],
        };


        double projectionMatrixGL[16] =
        {
            projectionMatrix[0][0], projectionMatrix[0][1], projectionMatrix[0][2], projectionMatrix[0][3],
            projectionMatrix[1][0], projectionMatrix[1][1], projectionMatrix[1][2], projectionMatrix[1][3],
            projectionMatrix[2][0], projectionMatrix[2][1], projectionMatrix[2][2], -1,
            projectionMatrix[3][0], projectionMatrix[3][1], projectionMatrix[3][2], projectionMatrix[3][3],
        };

        gluUnProject(eventX, viewPort[3]-1-(eventY),  0, modelViewMatrixGL, projectionMatrixGL, viewPort, &(p0[0]), &(p0[1]), &(p0[2]));
        gluUnProject(eventX+1, viewPort[3]-1-(eventY),0, modelViewMatrixGL, projectionMatrixGL, viewPort, &(px[0]), &(px[1]), &(px[2]));
        gluUnProject(eventX, viewPort[3]-1-(eventY+1),0, modelViewMatrixGL, projectionMatrixGL, viewPort, &(py[0]), &(py[1]), &(py[2]));
        gluUnProject(eventX, viewPort[3]-1-(eventY),  1, modelViewMatrixGL, projectionMatrixGL, viewPort, &(pz[0]), &(pz[1]), &(pz[2]));
        gluUnProject(eventX+1, viewPort[3]-1-(eventY), 0.1, modelViewMatrixGL, projectionMatrixGL, viewPort, &(px1[0]), &(px1[1]), &(px1[2]));
        gluUnProject(eventX, viewPort[3]-1-(eventY+1), 0, modelViewMatrixGL, projectionMatrixGL, viewPort, &(py1[0]), &(py1[1]), &(py1[2]));


        if ( pickDone)
            pz*=-1;

        px1 -= pz;
        py1 -= pz;

        px -= p0;
        py -= p0;
        pz -= p0;

        double r0 = sqrt(px.norm2() + py.norm2());
        double r1 = sqrt(px1.norm2() + py1.norm2());
        r1 = r0 + (r1-r0) / pz.norm();
        px.normalize();
        py.normalize();
        pz.normalize();
        sofa::defaulttype::Mat4x4d transform;
        transform.identity();
        transform[0][0] = px[0];
        transform[1][0] = px[1];
        transform[2][0] = px[2];
        transform[0][1] = py[0];
        transform[1][1] = py[1];
        transform[2][1] = py[2];
        transform[0][2] = pz[0];
        transform[1][2] = pz[1];
        transform[2][2] = pz[2];
        transform[0][3] = p0[0];
        transform[1][3] = p0[1];
        transform[2][3] = p0[2];


        sofa::defaulttype::Mat3x3d mat; mat = transform;
        sofa::defaulttype::Quat q; q.fromMatrix(mat);



        position  = transform*Vec4d(0,0,0,1);
        direction = transform*Vec4d(0,0,1,0);
        direction.normalize();

    }
    else
    {
        SReal w=mCamera->getOrthoWindowWidth()/(SReal)mRenderWindow->getWidth();
        SReal h=mCamera->getOrthoWindowHeight()/(SReal)mRenderWindow->getHeight();

        SReal posW=eventX-mRenderWindow->getWidth()*0.5;
        SReal posH=-1*(eventY-mRenderWindow->getHeight()*0.5);

        const Ogre::Vector3 &posCamera=mCamera->getPosition();
        Ogre::Vector3 p= posCamera + mCamera->getUp()*h*posH + mCamera->getRight()*w*posW;
        Ogre::Vector3 d=mCamera->getUp().crossProduct(mCamera->getRight());
        direction[0]=d.x; direction[1]=d.y; direction[2]=d.z;
        position[0]=p.x;  position[1]=p.y;  position[2]=p.z;
    }
    pick.updateRay(position, direction);
}

void QtOgreViewer::removeViewerTab(QTabWidget *t)
{
    if (tabLights)
        t->removePage(tabLights);
    if (tabCompositor)
        t->removePage(tabCompositor);
}

void QtOgreViewer::configureViewerTab(QTabWidget *t)
{
    /*
      if (dirLight)
      {
      if (!tabLights)
      t->addTab(tabLights,QString("Lights"));

      return;
      }*/
    addTabulationLights(t);
    addTabulationCompositor(t);

    needUpdateParameters = true;
}


void QtOgreViewer::addTabulationLights(QTabWidget *t)
{
    if (!tabLights)
    {
        tabLights = new QWidget();


        QVBoxLayout *l = new QVBoxLayout( tabLights, 0, 1, "tabLights");

        QToolBox *lightToolBox = new QToolBox(tabLights);
        lightToolBox->setCurrentIndex( 1 );
        //------------------------------------------------------------------------
        //Informations
        Q3GroupBox *groupInfo = new Q3GroupBox(QString("Information"), tabLights);
        groupInfo->setColumns(4);
        QWidget     *global    = new QWidget(groupInfo);
        QGridLayout *globalLayout = new QGridLayout(global);

        saveLightsButton = new QPushButton(QString("Save Lights"),global);
        globalLayout->addWidget(saveLightsButton,0,0);
        connect( saveLightsButton, SIGNAL( clicked() ), this, SLOT( saveLights() ) );

        globalLayout->addWidget(new QLabel(QString("Ambient"),global),2,0);
        for (unsigned int i=0; i<3; ++i)
        {
            std::ostringstream s;
            s<<"ambient" <<i;
            ambient[i] = new WDoubleLineEdit(global,s.str().c_str());
            ambient[i]->setMinValue( 0.0f);
            ambient[i]->setMaxValue( 1.0f);
            globalLayout->addWidget(ambient[i],2,i+1);
            connect( ambient[i], SIGNAL( returnPressed() ), this, SLOT( updateViewerParameters() ) );
        }
        l->addWidget(groupInfo);
        //------------------------------------------------------------------------
        //Directional Lights
        dirLight  = new QWidget( tabLights);
        QVBoxLayout *dirLightLayout = new QVBoxLayout(dirLight);
        lightToolBox->addItem( dirLight,QString("Directional Lights"));

        //   Information
        Q3GroupBox *infoDirLight = new Q3GroupBox( QString("Information"), dirLight);
        infoDirLight->setColumns(3);
        dirLightLayout->addWidget(infoDirLight);

        new QLabel(QString("Number:"),infoDirLight);
        numDirLight = new QSpinBox(infoDirLight); numDirLight->setMaximumWidth(SIZE_ENTRY);
        connect( numDirLight, SIGNAL( valueChanged(int)), this, SLOT( resizeDirLight(int)));

        dirLightLayout->addStretch();
        //------------------------------------------------------------------------
        //Point Lights
        pointLight  = new QWidget( tabLights);
        QVBoxLayout *pointLightLayout = new QVBoxLayout(pointLight);
        lightToolBox->addItem( pointLight,QString("Point Lights"));

        //   Information
        Q3GroupBox *infoPointLight = new Q3GroupBox( QString("Information"), pointLight);
        infoPointLight->setColumns(3);
        pointLightLayout->addWidget(infoPointLight);


        new QLabel(QString("Number:"),infoPointLight);
        numPointLight = new QSpinBox(infoPointLight); numPointLight->setMaximumWidth(SIZE_ENTRY);
        connect( numPointLight, SIGNAL( valueChanged(int)), this, SLOT( resizePointLight(int)));

        pointLightLayout->addStretch();
        //------------------------------------------------------------------------
        //Spot Lights
        spotLight  = new QWidget( tabLights);
        QVBoxLayout *spotLightLayout = new QVBoxLayout(spotLight);
        lightToolBox->addItem( spotLight,QString("Spot Lights"));

        //   Information
        Q3GroupBox *infoSpotLight = new Q3GroupBox( QString("Information"), spotLight);
        infoSpotLight->setColumns(3);
        spotLightLayout->addWidget(infoSpotLight);


        new QLabel(QString("Number:"),infoSpotLight);
        numSpotLight = new QSpinBox(infoSpotLight); numSpotLight->setMaximumWidth(SIZE_ENTRY);
        connect( numSpotLight, SIGNAL( valueChanged(int)), this, SLOT( resizeSpotLight(int)));
        spotLightLayout->addStretch();

        //------------------------------------------------------------------------
        //Add the tool box
        l->addWidget(lightToolBox);
    }
    t->addTab(tabLights,QString("Lights"));
}
void QtOgreViewer::addTabulationCompositor(QTabWidget *t)
{
    if (!tabCompositor)
    {
        tabCompositor = new QWidget();
        QVBoxLayout* l=new QVBoxLayout(tabCompositor);
        l->add( new QLabel(QString("Effects: "),tabCompositor));

        compositorWidget = new QWidget(tabCompositor);
        new QGridLayout(compositorWidget,0,2);
        l->add(compositorWidget);

        l->addSpacerItem(new QSpacerItem(10,10,QSizePolicy::Minimum, QSizePolicy::Expanding));
    }
    t->addTab(tabCompositor,QString("Compositor"));
}

void QtOgreViewer::addDirLight(std::string lightName)
{
    const int i=dirLightOgreWidget.size();
    dirLightOgreWidget.push_back( new QOgreDirectionalLightWidget(mSceneMgr,dirLight,lightName));
    ( (QVBoxLayout*) dirLight->layout())->insertWidget(i+1,dirLightOgreWidget[i]);
    connect(dirLightOgreWidget[i], SIGNAL( isDirty()), this, SLOT( updateViewerParameters() ) );
    dirLightOgreWidget[i]->show();
}

void QtOgreViewer::resizeDirLight(int v)
{
    const int sizeLight = dirLightOgreWidget.size();

    if ( v == sizeLight) return;
    else if (v < sizeLight)
    {
        for (int i=v; i<sizeLight; ++i)
            delete dirLightOgreWidget[i];
        dirLightOgreWidget.resize(v);
    }
    else if (v > sizeLight)
    {

        for (int i=sizeLight; i<v; ++i)
        {
            //Lights
            addDirLight();
        }
    }
    mRenderWindow->update();
};

void QtOgreViewer::addPointLight(std::string lightName)
{
    const int i=pointLightOgreWidget.size();
    pointLightOgreWidget.push_back( new QOgrePointLightWidget(mSceneMgr,pointLight,lightName));
    ( (QVBoxLayout*) pointLight->layout())->insertWidget(i+1,pointLightOgreWidget[i]);
    connect(pointLightOgreWidget[i], SIGNAL( isDirty()), this, SLOT( updateViewerParameters() ) );
    pointLightOgreWidget[i]->show();
}
void QtOgreViewer::resizePointLight(int v)
{
    const int sizeLight = pointLightOgreWidget.size();

    if ( v == sizeLight) return;
    else if (v < sizeLight)
    {
        for (int i=v; i<sizeLight; ++i)
            delete pointLightOgreWidget[i];

        pointLightOgreWidget.resize(v);
    }
    else if (v > sizeLight)
    {

        for (int i=sizeLight; i<v; ++i)
        {
            //Lights
            addPointLight();
        }
    }
    mRenderWindow->update();
};

void QtOgreViewer::addSpotLight(std::string lightName)
{
    const int i=spotLightOgreWidget.size();
    spotLightOgreWidget.push_back( new QOgreSpotLightWidget(mSceneMgr,spotLight,lightName));
    ( (QVBoxLayout*) spotLight->layout())->insertWidget(i+1,spotLightOgreWidget[i]);
    connect(spotLightOgreWidget[i], SIGNAL( isDirty()), this, SLOT( updateViewerParameters() ) );
    spotLightOgreWidget[i]->show();
}
void QtOgreViewer::resizeSpotLight(int v)
{
    const int sizeLight = spotLightOgreWidget.size();

    if ( v == sizeLight) return;
    else if (v < sizeLight)
    {
        for (int i=v; i<sizeLight; ++i)
            delete spotLightOgreWidget[i];

        spotLightOgreWidget.resize(v);
    }
    else if (v > sizeLight)
    {

        for (int i=sizeLight; i<v; ++i)
        {
            //Lights
            addSpotLight();
        }
    }
    mRenderWindow->update();
};

void QtOgreViewer::saveLights()
{
    std::ofstream out(sceneFile.c_str());
    if (!out.fail())
    {
        out << "<scene formatVersion=\"\">\n";
        out << "\t<environment>\n";
        out << "\t\t<colourBackground ";
        out << "r=\"" << backgroundColour[0] << "\" ";
        out << "g=\"" << backgroundColour[1] << "\" ";
        out << "b=\"" << backgroundColour[2] << "\" ";
        out << "a=\"" << 0                              << "\" />\n";

        out << "\t\t<colourAmbient ";
        out << "r=\"" << ambient[0]->getValue() << "\" ";
        out << "g=\"" << ambient[1]->getValue() << "\" ";
        out << "b=\"" << ambient[2]->getValue() << "\" />\n";
        out << "\t</environment>\n";
        out << "\n";
        out << "\t<nodes>\n";

        if (dirLightOgreWidget.size())
        {
            out << "\t\t<node name=\"DirectionalLightNode\">\n";
            for (unsigned int i=0; i<dirLightOgreWidget.size(); ++i)
            {
                dirLightOgreWidget[i]->save(out);
            }
            out << "\t\t</node>\n";
        }
        if (pointLightOgreWidget.size())
        {
            out << "\t\t<node name=\"PointLightNode\">\n";
            for (unsigned int i=0; i<pointLightOgreWidget.size(); ++i)
            {
                pointLightOgreWidget[i]->save(out);
            }
            out << "\t\t</node>\n";
        }

        if (spotLightOgreWidget.size())
        {
            out << "\t\t<node name=\"SpotLightNode\">\n";
            for (unsigned int i=0; i<spotLightOgreWidget.size(); ++i)
            {
                spotLightOgreWidget[i]->save(out);
            }
            out << "\t\t</node>\n";
        }

        out << "\t</nodes>\n";
        out << "</scene>";
    }
}

void QtOgreViewer::configure(sofa::component::configurationsetting::ViewerSetting* v)
{

    SofaViewer::configure(v);
    if (component::configurationsetting::OgreViewerSetting *viewerConf=dynamic_cast<component::configurationsetting::OgreViewerSetting *>(v))
    {
        //initCompositing=viewerConf->getCompositors();
        setLightActivated(viewerConf->getShadows());
        mCamera->getSceneManager()->setShadowTechnique(shadow);

    }
}

void QtOgreViewer::setLightActivated(bool b)
{
    if (b)
    {
        pickDone=true;
        shadow = Ogre::SHADOWTYPE_STENCIL_ADDITIVE;
        sofa::component::visualmodel::OgreVisualModel::lightsEnabled=true;
    }
    else
    {
        shadow = Ogre::SHADOWTYPE_NONE;
        sofa::component::visualmodel::OgreVisualModel::lightsEnabled=false;
    }
    mCamera->getSceneManager()->setShadowTechnique(shadow);
}

void QtOgreViewer::setBackgroundColour(float r, float g, float b)
{
    SofaViewer::setBackgroundColour(r,g,b);
    mVp->setBackgroundColour(Ogre::ColourValue(r,g,b,1.0));
}

QString QtOgreViewer::helpString()
{

    QString text("<H1>QtOgreViewer</H1><br>\
                         <ul>\
                         <li><b>Mouse</b>: TO NAVIGATE<br></li>\
                         <li><b>Shift & Left Button</b>: TO PICK OBJECTS<br></li>\
                         <li><b>A</b>: TO DRAW THE SCENE AXIS<br></li>\
                         <li><b>C</b>: TO CENTER THE VIEW<br></li>\
                         <li><b>B</b>: TO CHANGE THE BACKGROUND<br></li>\
                         <li><b>T</b>: TO CHANGE BETWEEN A PERSPECTIVE OR AN ORTHOGRAPHIC CAMERA<br></li>\
                         <li><b>L</b>: TO DRAW SHADOWS<br></li>\
                         <li><b>P</b>: TO SAVE A SEQUENCE OF OBJ<br>\
                         Each time the frame is updated an obj is exported<br></li>\
                         <li><b>I</b>: TO SAVE A SCREENSHOT<br>\
                         The captured images are saved in the running project directory under the name format capturexxxx.bmp<br></li>\
                         <li><b>T</b>: TO CHANGE BETWEEN A PERSPECTIVE OR AN ORTHOGRAPHIC CAMERA<br></li>\
                         <li><b>V</b>: TO SAVE A VIDEO<br>\
                         Each time the frame is updated a screenshot is saved<br></li>\
                         <li><b>Esc</b>: TO QUIT ::sofa:: <br></li></ul>");

    return text;
}

} //viewer

} //qt

} //gui

} //sofa
