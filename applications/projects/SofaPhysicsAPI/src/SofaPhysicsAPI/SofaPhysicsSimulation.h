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

#include "SofaPhysicsAPI.h"
#include "SofaPhysicsOutputMesh_impl.h"

#include <sofa/helper/system/thread/CTime.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/gl/DrawToolGL.h>
#include <sofa/component/visual/InteractiveCamera.h>
#include <sofa/gl/Texture.h>
#include <sofa/simulation/Node.h>

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

    bool load(const char* filename);
    void createScene();

    void start();
    void stop();
    void step();
    void reset();
    void resetView();
    void sendValue(const char* name, double value);
    void drawGL();

    unsigned int getNbOutputMeshes();
    SofaPhysicsOutputMesh** getOutputMesh(unsigned int meshID);
    SofaPhysicsOutputMesh** getOutputMeshes();

    bool isAnimated() const;
    void setAnimated(bool val);

    double getTimeStep() const;
    void   setTimeStep(double dt);
    double getTime() const;
    double getCurrentFPS() const;
    double* getGravity() const;
    void setGravity(double* gravity);

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

    sofa::simulation::Simulation* m_Simulation;
    sofa::simulation::Node::SPtr m_RootNode;
    std::string sceneFileName;
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
    void updateOutputMeshes();
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
        return m_Simulation;
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
