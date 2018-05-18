/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "SofaPhysicsAPI.h"
#include "SofaPhysicsSimulation.h"

#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glu.h>
#include <sofa/helper/io/Image.h>
#include <sofa/helper/gl/RAII.h>

#include <SofaSimulationTree/TreeSimulation.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/core/ObjectFactory.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <sofa/core/objectmodel/GUIEvent.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/init.h>

#include <sofa/gui/BaseGUI.h>
#include "fakegui.h"

#include <math.h>
#include <iostream>

#include "../plugins/SceneCreator/SceneCreator.h"

SofaPhysicsAPI::SofaPhysicsAPI(bool useGUI, int GUIFramerate)
    : impl(new SofaPhysicsSimulation(useGUI, GUIFramerate))
{
}

SofaPhysicsAPI::~SofaPhysicsAPI()
{
    if (impl != NULL)
    {
        delete impl;
        impl = NULL;
    }
}

const char *SofaPhysicsAPI::APIName()
{
    return impl->APIName();
}

bool SofaPhysicsAPI::load(const char* filename)
{
    return impl->load(filename);
}

void SofaPhysicsAPI::createScene()
{
    std::cout << "SofaPhysicsAPI::createScene" <<std::endl;
    return impl->createScene();
}

void SofaPhysicsAPI::start()
{
    impl->start();
}

void SofaPhysicsAPI::stop()
{
    impl->stop();
}

void SofaPhysicsAPI::step()
{
    impl->step();
}

void SofaPhysicsAPI::reset()
{
    impl->reset();
}

void SofaPhysicsAPI::resetView()
{
    impl->resetView();
}

void SofaPhysicsAPI::sendValue(const char* name, double value)
{
    impl->sendValue(name, value);
}

void SofaPhysicsAPI::drawGL()
{
    impl->drawGL();
}

unsigned int SofaPhysicsAPI::getNbOutputMeshes()
{
    return impl->getNbOutputMeshes();
}

SofaPhysicsOutputMesh** SofaPhysicsAPI::getOutputMesh(unsigned int meshID)
{
    return impl->getOutputMesh(meshID);
}

SofaPhysicsOutputMesh** SofaPhysicsAPI::getOutputMeshes()
{
    return impl->getOutputMeshes();
}

bool SofaPhysicsAPI::isAnimated() const
{
    return impl->isAnimated();
}

void SofaPhysicsAPI::setAnimated(bool val)
{
    impl->setAnimated(val);
}

double SofaPhysicsAPI::getTimeStep() const
{
    return impl->getTimeStep();
}

void SofaPhysicsAPI::setTimeStep(double dt)
{
    impl->setTimeStep(dt);
}

double SofaPhysicsAPI::getTime() const
{
    return impl->getTime();
}

double SofaPhysicsAPI::getCurrentFPS() const
{
    return impl->getCurrentFPS();
}

double* SofaPhysicsAPI::getGravity() const
{
    return impl->getGravity();
}

void SofaPhysicsAPI::setGravity(double* gravity)
{
    impl->setGravity(gravity);
}

const char* SofaPhysicsAPI::getSceneFileName() const
{
    return impl->getSceneFileName();
}

unsigned int SofaPhysicsAPI::getNbDataMonitors()
{
    return impl->getNbDataMonitors();
}

SofaPhysicsDataMonitor** SofaPhysicsAPI::getDataMonitors()
{
    return impl->getDataMonitors();
}

unsigned int SofaPhysicsAPI::getNbDataControllers()
{
    return impl->getNbDataControllers();
}

SofaPhysicsDataController** SofaPhysicsAPI::getDataControllers()
{
    return impl->getDataControllers();
}

////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////

using namespace sofa::defaulttype;
using namespace sofa::helper::gl;
using namespace sofa::core::objectmodel;

static sofa::core::ObjectFactory::ClassEntry::SPtr classVisualModel;

SofaPhysicsSimulation::SofaPhysicsSimulation(bool useGUI_, int GUIFramerate_)
    : useGUI(useGUI_)
    , GUIFramerate(GUIFramerate_)
{
    sofa::helper::init();
    static bool first = true;
    if (first)
    {
        if ( !useGUI )
        {
          // FakeGUI to be able to receive messages
          FakeGUI::Create();
        }
        else
        {
          sofa::gui::initMain();

          int argc= 1;
          char* argv[]= { const_cast<char*>("a") };

          if (sofa::gui::GUIManager::Init(argv[0],"qt"))
              std::cerr << "ERROR in sofa::gui::GUIManager::Init()" << std::endl;

          if (sofa::gui::GUIManager::createGUI(NULL))
              std::cerr << "ERROR in sofa::gui::GUIManager::CreateGUI()" << std::endl;

          sofa::gui::GUIManager::SetDimension(600,600);
        }
        first = false;
    }    

    m_RootNode = NULL;
    initGLDone = false;
    initTexturesDone = false;
    texLogo = NULL;
    lastW = 0;
    lastH = 0;
    vparams = sofa::core::visual::VisualParams::defaultInstance();

    m_Simulation = new sofa::simulation::tree::TreeSimulation();
    sofa::simulation::setSimulation(m_Simulation);

    sofa::component::initComponentGeneral();

    sofa::core::ObjectFactory::AddAlias("VisualModel", "OglModel", true,
            &classVisualModel);

    sofa::helper::system::PluginManager::getInstance().init();

    timeTicks = sofa::helper::system::thread::CTime::getRefTicksPerSec();
    frameCounter = 0;
    currentFPS = 0.0;
    lastRedrawTime = 0;
}

SofaPhysicsSimulation::~SofaPhysicsSimulation()
{
    for (std::map<SofaOutputMesh*, SofaPhysicsOutputMesh*>::const_iterator it = outputMeshMap.begin(), itend = outputMeshMap.end(); it != itend; ++it)
    {
        if (it->second) delete it->second;
    }
    outputMeshMap.clear();

    if ( useGUI ) {
      // GUI Cleanup
      //groot = dynamic_cast<sofa::simulation::Node*>( sofa::gui::GUIManager::CurrentSimulation() );

      //if (groot!=NULL)
      //  sofa::simulation::getSimulation()->unload(groot);


      //sofa::gui::GUIManager::closeGUI();
    }
}

const char *SofaPhysicsSimulation::APIName()
{
    return "SofaPhysicsSimulation API";
}

bool SofaPhysicsSimulation::load(const char* cfilename)
{
    std::string filename = cfilename;
    std::cout << "FROM APP: SofaPhysicsSimulation::load(" << filename << ")" << std::endl;
    sofa::helper::BackTrace::autodump();

    //bool wasAnimated = isAnimated();
    bool success = true;
    sofa::helper::system::DataRepository.findFile(filename);
    m_RootNode = m_Simulation->load(filename.c_str());
    if (m_RootNode.get())
    {
        sceneFileName = filename;
        m_Simulation->init(m_RootNode.get());
        updateOutputMeshes();

        if ( useGUI ) {
          sofa::gui::GUIManager::SetScene(m_RootNode.get(),cfilename);
        }
    }
    else
    {
        m_RootNode = m_Simulation->createNewGraph("");
        success = false;
    }
    initTexturesDone = false;
    lastW = 0;
    lastH = 0;
    lastRedrawTime = sofa::helper::system::thread::CTime::getRefTime();

//    if (isAnimated() != wasAnimated)
//        animatedChanged();
    return success;
}

void SofaPhysicsSimulation::createScene()
{
    m_RootNode = sofa::modeling::createRootWithCollisionPipeline();
    if (m_RootNode.get())
    {
        m_RootNode->setGravity( Vec3d(0,-9.8,0) );
        this->createScene_impl();

        m_Simulation->init(m_RootNode.get());

        updateOutputMeshes();
    }
    else
        std::cerr <<"Error: can't get m_RootNode" << std::endl;
}

void SofaPhysicsSimulation::createScene_impl()
{
    if (!m_RootNode.get())
        return;
}


void SofaPhysicsSimulation::sendValue(const char* name, double value)
{
    // send a GUIEvent to the tree
    if (m_RootNode!=0)
    {
        std::ostringstream oss;
        oss << value;
        sofa::core::objectmodel::GUIEvent event("",name,oss.str().c_str());
        m_RootNode->propagateEvent(sofa::core::ExecParams::defaultInstance(), &event);
    }
    this->update();
}

bool SofaPhysicsSimulation::isAnimated() const
{
    if (getScene())
        return getScene()->getContext()->getAnimate();
    return false;
}

void SofaPhysicsSimulation::setAnimated(bool val)
{
    if (val) start();
    else stop();
}

double SofaPhysicsSimulation::getTimeStep() const
{
    if (getScene())
        return getScene()->getContext()->getDt();
    else
        return 0.0;
}

void SofaPhysicsSimulation::setTimeStep(double dt)
{
    if (getScene())
    {
        getScene()->getContext()->setDt(dt);
    }
}

double SofaPhysicsSimulation::getTime() const
{
    if (getScene())
        return getScene()->getContext()->getTime();
    else
        return 0.0;
}

double SofaPhysicsSimulation::getCurrentFPS() const
{
    return currentFPS;
}

double *SofaPhysicsSimulation::getGravity() const
{
    double* gravityVec = new double[3];

    if (getScene())
    {
        const Vec3d& g = getScene()->getContext()->getGravity();
        gravityVec[0] = g.x();
        gravityVec[1] = g.y();
        gravityVec[2] = g.z();
    }

    return gravityVec;
}

void SofaPhysicsSimulation::setGravity(double* gravity)
{
    Vec3d g = Vec3d(gravity[0], gravity[1], gravity[2]);
    getScene()->getContext()->setGravity(g);
}


void SofaPhysicsSimulation::start()
{
    std::cout << "FROM APP: start()" << std::endl;
    if (isAnimated()) return;
    if (getScene())
    {
        getScene()->getContext()->setAnimate(true);
        //animatedChanged();
    }
}

void SofaPhysicsSimulation::stop()
{
    std::cout << "FROM APP: stop()" << std::endl;
    if (!isAnimated()) return;
    if (getScene())
    {
        getScene()->getContext()->setAnimate(false);
        //animatedChanged();
    }
}


void SofaPhysicsSimulation::reset()
{
    std::cout << "FROM APP: reset()" << std::endl;
    if (getScene())
    {
        getSimulation()->reset(getScene());
        this->update();
    }
}

void SofaPhysicsSimulation::resetView()
{
    if (getScene() && currentCamera)
    {
        currentCamera->setDefaultView(getScene()->getGravity());
        if (!sceneFileName.empty())
        {
            std::string viewFileName = sceneFileName + ".view";
            if (!currentCamera->importParametersFromFile(viewFileName))
                currentCamera->setDefaultView(getScene()->getGravity());
        }
    }
}

void SofaPhysicsSimulation::update()
{
}

void SofaPhysicsSimulation::step()
{
    sofa::simulation::Node* groot = getScene();
    if (!groot) return;
    beginStep();
    getSimulation()->animate(groot);
    getSimulation()->updateVisual(groot);
    if ( useGUI ) {
      sofa::gui::BaseGUI* gui = sofa::gui::GUIManager::getGUI();
      gui->stepMainLoop();
      if (GUIFramerate)
      {
          sofa::helper::system::thread::ctime_t curtime = sofa::helper::system::thread::CTime::getRefTime();
          if ((curtime-lastRedrawTime) > (double)timeTicks/GUIFramerate)
          {
              lastRedrawTime = curtime;
              gui->redraw();
          }
      }
    }
    endStep();
}

void SofaPhysicsSimulation::beginStep()
{
}

void SofaPhysicsSimulation::endStep()
{
    update();
    updateCurrentFPS();
    updateOutputMeshes();
}

void SofaPhysicsSimulation::updateCurrentFPS()
{
    if (frameCounter==0)
    {
        sofa::helper::system::thread::ctime_t t = sofa::helper::system::thread::CTime::getRefTime();
        for (int i=0; i<10; i++)
            stepTime[i] = t;
    }
    else
    {
        if ((frameCounter%10) == 0)
        {
            sofa::helper::system::thread::ctime_t curtime = sofa::helper::system::thread::CTime::getRefTime();
            int i = ((frameCounter/10)%10);
            currentFPS = ((double)timeTicks / (curtime - stepTime[i]))*(frameCounter<100?frameCounter:100);
            stepTime[i] = curtime;
            if ( useGUI ) {
                sofa::gui::BaseGUI* gui = sofa::gui::GUIManager::getGUI();
                gui->showFPS(currentFPS);
            }
        }
    }
    ++frameCounter;
}

void SofaPhysicsSimulation::updateOutputMeshes()
{
    sofa::simulation::Node* groot = getScene();
    if (!groot)
    {
        sofaOutputMeshes.clear();
        outputMeshes.clear();

        return;
    }
    sofaOutputMeshes.clear();    
    groot->get<SofaOutputMesh>(&sofaOutputMeshes, sofa::core::objectmodel::BaseContext::SearchDown);

    outputMeshes.resize(sofaOutputMeshes.size());

    for (unsigned int i=0; i<sofaOutputMeshes.size(); ++i)
    {
        SofaOutputMesh* sMesh = sofaOutputMeshes[i];
        SofaPhysicsOutputMesh*& oMesh = outputMeshMap[sMesh];
        if (oMesh == NULL)
        {
            oMesh = new SofaPhysicsOutputMesh;
            oMesh->impl->setObject(sMesh);
        }
        outputMeshes[i] = oMesh;
    }
}

unsigned int SofaPhysicsSimulation::getNbOutputMeshes()
{
    return outputMeshes.size();
}

SofaPhysicsOutputMesh** SofaPhysicsSimulation::getOutputMesh(unsigned int meshID)
{
    if (meshID >= outputMeshes.size())
        return NULL;
    else
        return &(outputMeshes[meshID]);
}

SofaPhysicsOutputMesh** SofaPhysicsSimulation::getOutputMeshes()
{
    if (outputMeshes.empty())
        return NULL;
    else
        return &(outputMeshes[0]);
}

unsigned int SofaPhysicsSimulation::getNbDataMonitors()
{
    return dataMonitors.size();
}

SofaPhysicsDataMonitor** SofaPhysicsSimulation::getDataMonitors()
{
    if (dataMonitors.empty())
    {
        sofa::simulation::Node* groot = getScene();
        if (!groot)
        {
            return NULL;
        }
        groot->get<SofaDataMonitor>(&sofaDataMonitors, sofa::core::objectmodel::BaseContext::SearchDown);
        dataMonitors.resize(sofaDataMonitors.size());
        for (unsigned int i=0; i<sofaDataMonitors.size(); ++i)
        {
            SofaDataMonitor* sData = sofaDataMonitors[i];
            SofaPhysicsDataMonitor* oData = new SofaPhysicsDataMonitor;
            oData->impl->setObject(sData);
            dataMonitors[i] = oData;
        }
    }
    return &(dataMonitors[0]);
}

unsigned int SofaPhysicsSimulation::getNbDataControllers()
{
    return dataControllers.size();
}

SofaPhysicsDataController** SofaPhysicsSimulation::getDataControllers()
{
    if (dataControllers.empty())
    {
        sofa::simulation::Node* groot = getScene();
        if (!groot)
        {
            return NULL;
        }
        groot->get<SofaDataController>(&sofaDataControllers, sofa::core::objectmodel::BaseContext::SearchDown);
        dataControllers.resize(sofaDataControllers.size());
        for (unsigned int i=0; i<sofaDataControllers.size(); ++i)
        {
            SofaDataController* sData = sofaDataControllers[i];
            SofaPhysicsDataController* oData = new SofaPhysicsDataController;
            oData->impl->setObject(sData);
            dataControllers[i] = oData;
        }
    }
    return &(dataControllers[0]);
}

void SofaPhysicsSimulation::drawGL()
{
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    if (!initGLDone)
    {
#ifdef SOFA_HAVE_GLEW
        glewInit();
#endif
        //Load texture for logo
        std::string imageFileName = "textures/SOFA_logo.bmp";
        if (sofa::helper::system::DataRepository.findFile(imageFileName))
        {            
            if (texLogo)
            {
                delete texLogo;
                texLogo = NULL;
            }

            sofa::helper::io::Image* image = sofa::helper::io::Image::FactoryImage::getInstance()->createObject("bmp", sofa::helper::system::DataRepository.getFile(imageFileName));
            texLogo = new sofa::helper::gl::Texture(image);
            texLogo->init();
        }
        
        initGLDone = true;
    }

    const int vWidth = viewport[2];
    const int vHeight = viewport[3];

    if (texLogo && texLogo->getImage())
    {
        int w = 0;
        int h = 0;
        h = texLogo->getImage()->getHeight();
        w = texLogo->getImage()->getWidth();

        Enable <GL_TEXTURE_2D> tex;
        glDisable(GL_DEPTH_TEST);
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(-0.5, vWidth, -0.5, vHeight, -1.0, 1.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        if (texLogo)
            texLogo->bind();

        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);

        glColor3f(1.0f, 1.0f, 1.0f);
        glBegin(GL_QUADS);
        glTexCoord2d(0.0, 0.0);
        glVertex3d((vWidth-w)/2, (vHeight-h)/2, 0.0);

        glTexCoord2d(1.0, 0.0);
        glVertex3d( vWidth-(vWidth-w)/2, (vHeight-h)/2, 0.0);

        glTexCoord2d(1.0, 1.0);
        glVertex3d( vWidth-(vWidth-w)/2, vHeight-(vHeight-h)/2, 0.0);

        glTexCoord2d(0.0, 1.0);
        glVertex3d((vWidth-w)/2, vHeight-(vHeight-h)/2, 0.0);
        glEnd();

        glBindTexture(GL_TEXTURE_2D, 0);

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glDisable(GL_BLEND);
    }

    if (m_RootNode.get())
    {
        sofa::simulation::Node* groot = m_RootNode.get();
        if (!initTexturesDone)
        {
            std::cout << "INIT VISUAL" << std::endl;
            getSimulation()->initTextures(groot);
            bool setView = false;
            groot->get(currentCamera);
            if (!currentCamera)
            {
                currentCamera = sofa::core::objectmodel::New<sofa::component::visualmodel::InteractiveCamera>();
                currentCamera->setName(sofa::core::objectmodel::Base::shortName(currentCamera.get()));
                groot->addObject(currentCamera);
                currentCamera->p_position.forceSet();
                currentCamera->p_orientation.forceSet();
                currentCamera->bwdInit();
            }
            setView = true;
            //}

            vparams->sceneBBox() = groot->f_bbox.getValue();
            currentCamera->setBoundingBox(vparams->sceneBBox().minBBox(), vparams->sceneBBox().maxBBox());
            currentCamera->setViewport(vWidth, vHeight);
            if (setView)
                resetView();
            std::cout << "initTexturesDone" << std::endl;
            initTexturesDone = true;
        }

        glDepthFunc(GL_LEQUAL);
        glClearDepth(1.0);
        glClear(GL_DEPTH_BUFFER_BIT);
        glEnable(GL_NORMALIZE);
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
        static GLfloat    lightPosition[4] = { -0.7f, 0.3f, 0.0f, 1.0f};
        static GLfloat    lmodel_ambient[]    = {0.0f, 0.0f, 0.0f, 0.0f};
        static GLfloat    ambientLight[4] = { 0.5f, 0.5f, 0.5f, 1.0f};
        static GLfloat    diffuseLight[4] = { 0.9f, 0.9f, 0.9f, 1.0f};
        static GLfloat    specularLight[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
        static GLfloat    specularMat[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_FALSE);
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);

        // Setup 'light 0'
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
        glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight);
        glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);

        // Enable color tracking
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

        // All materials hereafter have full specular reflectivity with a high shine
        glMaterialfv(GL_FRONT, GL_SPECULAR, specularMat);
        glMateriali(GL_FRONT, GL_SHININESS, 128);

        glShadeModel(GL_SMOOTH);

        // Define background color
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);

        // Turn on our light and enable color along with the light
        //glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glColor4f(1,1,1,1);
        glDisable(GL_COLOR_MATERIAL);
        glEnable(GL_LIGHTING);
        glEnable(GL_DEPTH_TEST);

        vparams->sceneBBox() = groot->f_bbox.getValue();

        vparams->viewport() = sofa::helper::make_array(viewport[0], viewport[1], viewport[2], viewport[3]);

        if (vWidth != lastW || vHeight != lastH)
        {
            lastW = vWidth;
            lastH = vHeight;
            if (currentCamera)
                currentCamera->setViewport(vWidth, vHeight);
            calcProjection();
        }

        // \todo {epernod this is not possible anymore}
//        currentCamera->getOpenGLMatrix(lastModelviewMatrix);

        glMatrixMode(GL_PROJECTION);
        glLoadMatrixd(lastProjectionMatrix);
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixd(lastModelviewMatrix);

        getSimulation()->draw(vparams,groot);

        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        glDisableClientState(GL_NORMAL_ARRAY);

    }



    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
}


// ---------------------------------------------------------
// --- Reshape of the window, reset the projection
// ---------------------------------------------------------
void SofaPhysicsSimulation::calcProjection()
{
    int width = lastW;
    int height = lastH;
    double xNear, yNear/*, xOrtho, yOrtho*/;
    double xFactor = 1.0, yFactor = 1.0;
    double offset;
    double xForeground, yForeground, zForeground, xBackground, yBackground,
           zBackground;
    Vector3 center;

    /// Camera part
    if (!currentCamera)
        return;
    sofa::simulation::Node* groot = getScene();
    if (groot && (!groot->f_bbox.getValue().isValid()))
    {
        vparams->sceneBBox() = groot->f_bbox.getValue();
        currentCamera->setBoundingBox(vparams->sceneBBox().minBBox(), vparams->sceneBBox().maxBBox());
    }
    currentCamera->computeZ();

    vparams->zNear() = currentCamera->getZNear();
    vparams->zFar() = currentCamera->getZFar();

    xNear = 0.35 * vparams->zNear();
    yNear = 0.35 * vparams->zNear();
    offset = 0.001 * vparams->zNear(); // for foreground and background planes

    /*xOrtho = fabs(vparams->sceneTransform().translation[2]) * xNear
            / vparams->zNear();
    yOrtho = fabs(vparams->sceneTransform().translation[2]) * yNear
            / vparams->zNear();*/

    if ((height != 0) && (width != 0))
    {
        if (height > width)
        {
            xFactor = 1.0;
            yFactor = (double) height / (double) width;
        }
        else
        {
            xFactor = (double) width / (double) height;
            yFactor = 1.0;
        }
    }
    vparams->viewport() = sofa::helper::make_array(0,0,width,height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    xFactor *= 0.01;
    yFactor *= 0.01;

    //std::cout << xNear << " " << yNear << std::endl;

    zForeground = -vparams->zNear() - offset;
    zBackground = -vparams->zFar() + offset;

    if (currentCamera->getCameraType() == sofa::core::visual::VisualParams::PERSPECTIVE_TYPE)
        gluPerspective(currentCamera->getFieldOfView(), (double) width / (double) height, vparams->zNear(), vparams->zFar());
    else
    {
        double ratio = vparams->zFar() / (vparams->zNear() * 20);
        Vector3 tcenter = vparams->sceneTransform() * center;
        if (tcenter[2] < 0.0)
        {
            ratio = -300 * (tcenter.norm2()) / tcenter[2];
        }
        glOrtho((-xNear * xFactor) * ratio, (xNear * xFactor) * ratio, (-yNear
                * yFactor) * ratio, (yNear * yFactor) * ratio,
                vparams->zNear(), vparams->zFar());
    }

    xForeground = -zForeground * xNear / vparams->zNear();
    yForeground = -zForeground * yNear / vparams->zNear();
    xBackground = -zBackground * xNear / vparams->zNear();
    yBackground = -zBackground * yNear / vparams->zNear();

    xForeground *= xFactor;
    yForeground *= yFactor;
    xBackground *= xFactor;
    yBackground *= yFactor;

    glGetDoublev(GL_PROJECTION_MATRIX,lastProjectionMatrix);

    glMatrixMode(GL_MODELVIEW);
}
