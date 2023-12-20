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
#pragma once

#include <SofaPhysicsAPI/config.h>
#include "SofaPhysicsAPI.h"
#include "SofaPhysicsOutputMesh_impl.h"

#include <sofa/helper/system/thread/CTime.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/gl/DrawToolGL.h>
#include <sofa/component/visual/InteractiveCamera.h>
#include <sofa/gl/Texture.h>
#include <sofa/simulation/Node.h>
#include <sofa/helper/logging/LoggingMessageHandler.h>

#include <map>

#if SOFAPHYSICSAPI_HAVE_SOFAVALIDATION == 1
#include "SofaPhysicsDataMonitor_impl.h"
#include "SofaPhysicsDataController_impl.h"
#endif

class SOFA_SOFAPHYSICSAPI_API SofaPhysicsSimulation
{
public:
    SofaPhysicsSimulation(bool useGUI_ = false, int GUIFramerate_ = 0);
    virtual ~SofaPhysicsSimulation();

    const char* APIName();

    /// Load an XML file containing the main scene description. Will return API_SUCCESS or API_SCENE_FAILED if loading failed
    int load(const char* filename);

    /// Call unload of the current scene graph. Will return API_SUCCESS or API_SCENE_NULL if scene is null
    int unload();

    /// Method to load a specific SOFA plugin using it's full path @param pluginPath. Return error code.
    int loadPlugin(const char* pluginPath);
    void createScene();

    void start();
    void stop();
    void step();
    void reset();
    void resetView();
    void sendValue(const char* name, double value);
    void drawGL();

    /// return the number of SofaPhysicsOutputMesh (i.e @sa outputMeshes size)
    unsigned int getNbOutputMeshes() const;

    /// return pointer to the SofaPhysicsOutputMesh at the @param meshID position in @sa outputMeshes. Return nullptr if out of bounds.
    SofaPhysicsOutputMesh* getOutputMeshPtr(unsigned int meshID) const; 
    /// return pointer to the SofaPhysicsOutputMesh with the name equal to @param name in @sa outputMeshes. Return nullptr if not found.
    SofaPhysicsOutputMesh* getOutputMeshPtr(const char* name) const;

    SofaPhysicsOutputMesh** getOutputMesh(unsigned int meshID);
    SofaPhysicsOutputMesh** getOutputMeshes();

    bool isAnimated() const;
    void setAnimated(bool val);

    double getTimeStep() const;
    void   setTimeStep(double dt);
    double getTime() const;
    double getCurrentFPS() const;
    double* getGravity() const;
    int getGravity(double* values) const;

    void setGravity(double* gravity);

    /// message API
    /// Method to activate/deactivate SOFA MessageHandler according to @param value. Will store status in @sa m_msgIsActivated. Return Error code.
    int activateMessageHandler(bool value);
    /// Method to get the number of messages in queue
    int getNbMessages();
    /// Method to return the queued message of index @param messageId and its type level inside @param msgType
    std::string getMessage(int messageId, int& msgType);
    /// Method clear the list of queued messages. Return Error code.
    int clearMessages();

    unsigned int getNbDataMonitors();
    SofaPhysicsDataMonitor** getDataMonitors();

    unsigned int getNbDataControllers();
    SofaPhysicsDataController** getDataControllers();

    typedef SofaPhysicsOutputMesh::Impl::SofaOutputMesh SofaOutputMesh;
    typedef SofaPhysicsOutputMesh::Impl::SofaVisualOutputMesh SofaVisualOutputMesh;

#if SOFAPHYSICSAPI_HAVE_SOFAVALIDATION == 1
    typedef SofaPhysicsDataMonitor::Impl::SofaDataMonitor SofaDataMonitor;
    typedef SofaPhysicsDataController::Impl::SofaDataController SofaDataController;
#endif
protected:

    sofa::simulation::Node::SPtr m_RootNode;
    std::string sceneFileName;
    /// Pointer to the LoggingMessageHandler
    sofa::helper::logging::LoggingMessageHandler* m_msgHandler;
    /// Status of the LoggingMessageHandler
    bool m_msgIsActivated;

    sofa::component::visual::BaseCamera::SPtr currentCamera;

    std::map<SofaOutputMesh*, SofaPhysicsOutputMesh*> outputMeshMap;

    std::vector<SofaOutputMesh*> sofaOutputMeshes;

    std::vector<SofaPhysicsOutputMesh*> outputMeshes;


#if SOFAPHYSICSAPI_HAVE_SOFAVALIDATION == 1
    std::vector<SofaDataMonitor*> sofaDataMonitors;
    std::vector<SofaPhysicsDataMonitor*> dataMonitors;

    std::vector<SofaDataController*> sofaDataControllers;
    std::vector<SofaPhysicsDataController*> dataControllers;
#endif

    sofa::gl::Texture *texLogo;
    double lastProjectionMatrix[16];
    double lastModelviewMatrix[16];
    GLint lastW, lastH;
    GLint lastViewport[4];
    bool initGLDone;
    bool initTexturesDone;
    bool useGUI;
    int GUIFramerate;
    sofa::core::visual::VisualParams* vparams;

    sofa::helper::system::thread::ctime_t stepTime[10];
    sofa::helper::system::thread::ctime_t timeTicks;
    sofa::helper::system::thread::ctime_t lastRedrawTime;
    int frameCounter;
    double currentFPS;

    void update();
    int updateOutputMeshes();
    void updateCurrentFPS();
    void beginStep();
    void endStep();
    void calcProjection();

    virtual void createScene_impl();

public:

    const char* getSceneFileName() const
    {
        return sceneFileName.c_str();
    }
    sofa::simulation::Simulation* getSimulation() const
    {
        return sofa::simulation::getSimulation();
    }
    sofa::simulation::Node* getScene() const
    {
        return m_RootNode.get();
    }

    sofa::simulation::Node::SPtr getRootNode() const
    {
        return m_RootNode;
    }

};
