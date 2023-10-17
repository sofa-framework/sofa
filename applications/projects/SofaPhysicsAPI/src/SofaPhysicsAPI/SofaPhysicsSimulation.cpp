/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#include <sofa/gl/gl.h>
#include <sofa/gl/glu.h>
#include <sofa/helper/io/Image.h>
#include <sofa/gl/RAII.h>

#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/Utils.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/GUIEvent.h>

#include <sofa/simulation/graph/DAGSimulation.h>

#include <sofa/gui/common/GUIManager.h>
#include <sofa/gui/common/init.h>
#include <sofa/helper/init.h>

#include <sofa/gui/common/BaseGUI.h>
#include "fakegui.h"

#include <sofa/type/Vec.h>

#include <cmath>
#include <iostream>

#include <sofa/simulation/graph/SimpleApi.h>

#include <sofa/component/init.h>

SofaPhysicsAPI::SofaPhysicsAPI(bool useGUI, int GUIFramerate)
    : impl(new SofaPhysicsSimulation(useGUI, GUIFramerate))
{
}

SofaPhysicsAPI::~SofaPhysicsAPI()
{
    if (impl != nullptr)
    {
        delete impl;
        impl = nullptr;
    }
}

const char *SofaPhysicsAPI::APIName()
{
    return impl->APIName();
}

int SofaPhysicsAPI::load(const char* filename)
{
    return impl->load(filename);
}

int SofaPhysicsAPI::unload()
{
    return impl->unload();
}

const char* SofaPhysicsAPI::loadSofaIni(const char* pathIniFile)
{
    auto sofaIniFilePath = std::string(pathIniFile);

    std::map<std::string, std::string> iniFileValues = sofa::helper::Utils::readBasicIniFile(sofaIniFilePath);
    std::string shareDir = "SHARE_DIR Path Not found";
    // and add them to DataRepository
    if (iniFileValues.find("SHARE_DIR") != iniFileValues.end())
    {
        shareDir = iniFileValues["SHARE_DIR"];
        if (!sofa::helper::system::FileSystem::isAbsolute(shareDir))
            shareDir = "./" + shareDir;
        sofa::helper::system::DataRepository.addFirstPath(shareDir);
    }
    if (iniFileValues.find("EXAMPLES_DIR") != iniFileValues.end())
    {
        std::string examplesDir = iniFileValues["EXAMPLES_DIR"];
        if (!sofa::helper::system::FileSystem::isAbsolute(examplesDir))
            examplesDir = "./" + examplesDir;
        sofa::helper::system::DataRepository.addFirstPath(examplesDir);
    }
    if (iniFileValues.find("PYTHON_DIR") != iniFileValues.end())
    {
        std::string pythonDir = iniFileValues["PYTHON_DIR"];
        if (!sofa::helper::system::FileSystem::isAbsolute(pythonDir))
            pythonDir = "./" + pythonDir;
        sofa::helper::system::DataRepository.addFirstPath(pythonDir);
    }

    char* cstr = new char[shareDir.length() + 1];
    std::strcpy(cstr, shareDir.c_str());

    return cstr;
}

int SofaPhysicsAPI::loadPlugin(const char* pluginPath)
{
    return impl->loadPlugin(pluginPath);
}

void SofaPhysicsAPI::createScene()
{
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

unsigned int SofaPhysicsAPI::getNbOutputMeshes() const
{
    return impl->getNbOutputMeshes();
}

SofaPhysicsOutputMesh* SofaPhysicsAPI::getOutputMeshPtr(unsigned int meshID) const
{
    return impl->getOutputMeshPtr(meshID);
}

SofaPhysicsOutputMesh* SofaPhysicsAPI::getOutputMeshPtr(const char* name) const
{
    return impl->getOutputMeshPtr(name);
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

int SofaPhysicsAPI::getGravity(double* values) const
{
    return impl->getGravity(values);
}

void SofaPhysicsAPI::setGravity(double* gravity)
{
    impl->setGravity(gravity);
}

int SofaPhysicsAPI::activateMessageHandler(bool value)
{
    return impl->activateMessageHandler(value);
}

int SofaPhysicsAPI::getNbMessages()
{
    return impl->getNbMessages();
}

const char* SofaPhysicsAPI::getMessage(int messageId, int& msgType)
{
    std::string value = impl->getMessage(messageId, msgType);
    char* cstr = new char[value.length() + 1];
    std::strcpy(cstr, value.c_str());
    return cstr;
}

int SofaPhysicsAPI::clearMessages()
{
    return impl->clearMessages();
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
using namespace sofa::gl;
using namespace sofa::core::objectmodel;

static sofa::core::ObjectFactory::ClassEntry::SPtr classVisualModel;
using sofa::helper::logging::MessageDispatcher;
using sofa::helper::logging::LoggingMessageHandler;

SofaPhysicsSimulation::SofaPhysicsSimulation(bool useGUI_, int GUIFramerate_)
    : m_msgIsActivated(false)
    , useGUI(useGUI_)
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
          sofa::gui::common::init();

          char* argv[]= { const_cast<char*>("a") };

          if (sofa::gui::common::GUIManager::Init(argv[0],"qt"))
              std::cerr << "ERROR in sofa::gui::common::GUIManager::Init()" << std::endl;

          if (sofa::gui::common::GUIManager::createGUI(NULL))
              std::cerr << "ERROR in sofa::gui::common::GUIManager::CreateGUI()" << std::endl;

          sofa::gui::common::GUIManager::SetDimension(600,600);
        }
        first = false;
    }    

    // create message handler
    m_msgHandler = new LoggingMessageHandler();
    MessageDispatcher::clearHandlers();
    MessageDispatcher::addHandler(m_msgHandler);

    m_RootNode = NULL;
    initGLDone = false;
    initTexturesDone = false;
    texLogo = nullptr;
    lastW = 0;
    lastH = 0;
    vparams = sofa::core::visual::VisualParams::defaultInstance();

    assert(sofa::simulation::getSimulation());

    sofa::component::init(); // force dependency on Sofa.Component

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
      //groot = dynamic_cast<sofa::simulation::Node*>( sofa::gui::common::GUIManager::CurrentSimulation() );

      //if (groot!=NULL)
      //  sofa::simulation::getSimulation()->unload(groot);


      //sofa::gui::common::GUIManager::closeGUI();
    }

    MessageDispatcher::rmHandler(m_msgHandler);
    if (m_msgIsActivated)
        m_msgHandler->deactivate();

    if (m_msgHandler != nullptr)
    {
        delete m_msgHandler;
        m_msgHandler = nullptr;
    }
}

const char *SofaPhysicsSimulation::APIName()
{
    return "SofaPhysicsSimulation API";
}

int SofaPhysicsSimulation::load(const char* cfilename)
{
    std::string filename = cfilename;
    sofa::helper::BackTrace::autodump();

    sofa::helper::system::DataRepository.findFile(filename);
    m_RootNode = sofa::simulation::node::load(filename.c_str());
    int result = API_SUCCESS;
    if (m_RootNode.get())
    {
        sceneFileName = filename;
        sofa::simulation::node::initRoot(m_RootNode.get());
        result = updateOutputMeshes();

        if ( useGUI ) {
          sofa::gui::common::GUIManager::SetScene(m_RootNode.get(),cfilename);
        }
    }
    else
    {
        m_RootNode = sofa::simulation::getSimulation()->createNewGraph("");
        return API_SCENE_FAILED;
    }
    initTexturesDone = false;
    lastW = 0;
    lastH = 0;
    lastRedrawTime = sofa::helper::system::thread::CTime::getRefTime();

    return result;
}

int SofaPhysicsSimulation::unload()
{
    if (m_RootNode.get())
    {
        sofa::simulation::node::unload(m_RootNode);
    }
    else
    {
        msg_error("SofaPhysicsSimulation") << "Error: can't get scene root node.";
        return API_SCENE_NULL;
    }

    return API_SUCCESS;
}

int SofaPhysicsSimulation::loadPlugin(const char* pluginPath)
{
    sofa::helper::system::PluginManager::PluginLoadStatus plugres = sofa::helper::system::PluginManager::getInstance().loadPlugin(pluginPath);
    if (plugres == sofa::helper::system::PluginManager::PluginLoadStatus::SUCCESS || plugres == sofa::helper::system::PluginManager::PluginLoadStatus::ALREADY_LOADED)
        return API_SUCCESS;
    else if (plugres == sofa::helper::system::PluginManager::PluginLoadStatus::INVALID_LOADING)
        return API_PLUGIN_INVALID_LOADING;
    else if (plugres == sofa::helper::system::PluginManager::PluginLoadStatus::MISSING_SYMBOL)
        return API_PLUGIN_MISSING_SYMBOL;
    else if (plugres == sofa::helper::system::PluginManager::PluginLoadStatus::PLUGIN_FILE_NOT_FOUND)
        return API_PLUGIN_FILE_NOT_FOUND;
    else
        return API_PLUGIN_LOADING_FAILED;
}

void SofaPhysicsSimulation::createScene()
{
    m_RootNode = sofa::simulation::getSimulation()->createNewGraph("root");
    sofa::simpleapi::createObject(m_RootNode, "CollisionPipeline", { {"name","Collision Pipeline"} });
    sofa::simpleapi::createObject(m_RootNode, "BruteForceBroadPhase", { {"name","Broad Phase Detection"} });
    sofa::simpleapi::createObject(m_RootNode, "BVHNarrowPhase", { {"name","Narrow Phase Detection"} });
    sofa::simpleapi::createObject(m_RootNode, "MinProximityIntersection", { {"name","Proximity"},
                                                               {"alarmDistance", "0.3"},
                                                               {"contactDistance", "0.2"} });

    sofa::simpleapi::createObject(m_RootNode, "CollisionResponse", {
                                {"name", "Contact Manager"},
                                {"response", "PenalityContactForceField"}
        });

    if (m_RootNode.get())
    {
        m_RootNode->setGravity({ 0,-9.8,0 });
        this->createScene_impl();

        sofa::simulation::node::initRoot(m_RootNode.get());

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
        const auto& g = getScene()->getContext()->getGravity();
        gravityVec[0] = g.x();
        gravityVec[1] = g.y();
        gravityVec[2] = g.z();
    }

    return gravityVec;
}

int SofaPhysicsSimulation::getGravity(double* values) const
{
    if (getScene())
    {
        const auto& g = getScene()->getContext()->getGravity();
        values[0] = g.x();
        values[1] = g.y();
        values[2] = g.z();
        return API_SUCCESS;
    }
    else
    {
        return API_SCENE_NULL;
    }
}

void SofaPhysicsSimulation::setGravity(double* gravity)
{
    const auto& g = sofa::type::Vec3d(gravity[0], gravity[1], gravity[2]);
    getScene()->getContext()->setGravity(g);
}


void SofaPhysicsSimulation::start()
{
    if (isAnimated()) return;
    if (getScene())
    {
        getScene()->getContext()->setAnimate(true);
        //animatedChanged();
    }
}

void SofaPhysicsSimulation::stop()
{
    if (!isAnimated()) return;
    if (getScene())
    {
        getScene()->getContext()->setAnimate(false);
        //animatedChanged();
    }
}


void SofaPhysicsSimulation::reset()
{
    if (getScene())
    {
        sofa::simulation::node::reset(getScene());
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
    sofa::simulation::node::animate(groot);
    sofa::simulation::node::updateVisual(groot);
    if ( useGUI ) {
      sofa::gui::common::BaseGUI* gui = sofa::gui::common::GUIManager::getGUI();
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
                sofa::gui::common::BaseGUI* gui = sofa::gui::common::GUIManager::getGUI();
                gui->showFPS(currentFPS);
            }
        }
    }
    ++frameCounter;
}

int SofaPhysicsSimulation::updateOutputMeshes()
{
    sofa::simulation::Node* groot = getScene();
    if (!groot)
    {
        sofaOutputMeshes.clear();
        outputMeshes.clear();

        return API_SCENE_NULL;
    }
    sofaOutputMeshes.clear();    
    groot->get<SofaOutputMesh>(&sofaOutputMeshes, sofa::core::objectmodel::BaseContext::SearchRoot);

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

    return sofaOutputMeshes.size();
}

unsigned int SofaPhysicsSimulation::getNbOutputMeshes() const
{
    return outputMeshes.size();
}

SofaPhysicsOutputMesh* SofaPhysicsSimulation::getOutputMeshPtr(unsigned int meshID) const
{
    if (meshID >= outputMeshes.size())
        return nullptr;
    else
        return outputMeshes[meshID];
}

SofaPhysicsOutputMesh* SofaPhysicsSimulation::getOutputMeshPtr(const char* name) const
{
    const auto nameStr = std::string(name);
    for (SofaPhysicsOutputMesh* mesh : outputMeshes)
    {
        if (nameStr.compare(std::string(mesh->getName())) == 0)
            return mesh;
    }

    return nullptr;
}

SofaPhysicsOutputMesh** SofaPhysicsSimulation::getOutputMesh(unsigned int meshID)
{
    if (meshID >= outputMeshes.size())
        return nullptr;
    else
        return &(outputMeshes[meshID]);
}

SofaPhysicsOutputMesh** SofaPhysicsSimulation::getOutputMeshes()
{
    if (outputMeshes.empty())
        return nullptr;
    else
        return &(outputMeshes[0]);
}


int SofaPhysicsSimulation::activateMessageHandler(bool value)
{
    if (value)
        m_msgHandler->activate();
    else
        m_msgHandler->deactivate();

    m_msgIsActivated = value;

    return API_SUCCESS;
}

int SofaPhysicsSimulation::getNbMessages()
{
    return static_cast<int>(m_msgHandler->getMessages().size());
}

std::string SofaPhysicsSimulation::getMessage(int messageId, int& msgType)
{
    const std::vector<sofa::helper::logging::Message>& msgs = m_msgHandler->getMessages();

    if (messageId >= (int)msgs.size()) {
        msgType = -1;
        return "Error messageId out of bounds";
    }

    msgType = static_cast<int>(msgs[messageId].type());
    return msgs[messageId].messageAsString();
}

int SofaPhysicsSimulation::clearMessages()
{
    m_msgHandler->reset();

    return API_SUCCESS;
}


unsigned int SofaPhysicsSimulation::getNbDataMonitors()
{

#if SOFAPHYSICSAPI_HAVE_SOFAVALIDATION == 1
    return dataMonitors.size();
#else
    msg_error("SofaPhysicsSimulation") << "did not implement getNbDataMonitors()";
    return 0;
#endif
}

SofaPhysicsDataMonitor** SofaPhysicsSimulation::getDataMonitors()
{
#if SOFAPHYSICSAPI_HAVE_SOFAVALIDATION == 1
    if (dataMonitors.empty())
    {
        sofa::simulation::Node* groot = getScene();
        if (!groot)
        {
            return nullptr;
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
#else
    msg_error("SofaPhysicsSimulation") << "did not implement getDataMonitors()";
    return nullptr;
#endif
}

unsigned int SofaPhysicsSimulation::getNbDataControllers()
{
#if SOFAPHYSICSAPI_HAVE_SOFAVALIDATION == 1
    return dataControllers.size();
#else
    msg_error("SofaPhysicsSimulation") << "did not implement getNbDataControllers()";
    return 0;
#endif
}

SofaPhysicsDataController** SofaPhysicsSimulation::getDataControllers()
{
#if SOFAPHYSICSAPI_HAVE_SOFAVALIDATION == 1
    if (dataControllers.empty())
    {
        sofa::simulation::Node* groot = getScene();
        if (!groot)
        {
            return nullptr;
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
#else
    msg_error("SofaPhysicsSimulation") << "did not implement getDataControllers()";
    return nullptr;
#endif
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
        glewInit();
        //Load texture for logo
        std::string imageFileName = "textures/SOFA_logo.bmp";
        if (sofa::helper::system::DataRepository.findFile(imageFileName))
        {            
            if (texLogo)
            {
                delete texLogo;
                texLogo = nullptr;
            }

            sofa::helper::io::Image* image = sofa::helper::io::Image::FactoryImage::getInstance()->createObject("bmp", sofa::helper::system::DataRepository.getFile(imageFileName));
            texLogo = new sofa::gl::Texture(image);
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
            sofa::simulation::node::initTextures(groot);
            bool setView = false;
            groot->get(currentCamera);
            if (!currentCamera)
            {
                currentCamera = sofa::core::objectmodel::New<sofa::component::visual::InteractiveCamera>();
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

        vparams->viewport() = sofa::type::make_array(viewport[0], viewport[1], viewport[2], viewport[3]);

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

        sofa::simulation::node::draw(vparams,groot);

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
    sofa::type::Vec3 center;

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
    vparams->viewport() = sofa::type::make_array(0,0,width,height);

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
        auto tcenter = center;
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
