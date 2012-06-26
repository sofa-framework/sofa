#ifndef SOFAPHYSICSSIMULATION_IMPL_H
#define SOFAPHYSICSSIMULATION_IMPL_H

#include "SofaPhysicsSimulation.h"

#include <sofa/simulation/common/Simulation.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/visual/DrawToolGL.h>
#include <sofa/component/visualmodel/InteractiveCamera.h>
#include <sofa/helper/gl/Texture.h>

class SofaPhysicsSimulation::Impl
{
public:
    Impl();
    ~Impl();

    bool load(std::string filename);
    void start();
    void stop();
    void step();
    void reset();
    void resetView();
    void sendValue(std::string name, double value);
    void drawGL();

    bool isAnimated() const;
    void setAnimated(bool val);

    double getTimeStep() const;
    void   setTimeStep(double dt);
    double getCurrentFPS() const;

protected:

    sofa::simulation::Simulation* m_Simulation;
    sofa::simulation::Node::SPtr m_RootNode;
    std::string sceneFileName;
    sofa::component::visualmodel::BaseCamera::SPtr currentCamera;
    sofa::helper::gl::Texture *texLogo;
    double lastProjectionMatrix[16];
    double lastModelviewMatrix[16];
    GLint lastW, lastH;
    GLint lastViewport[4];
    bool initGLDone;
    bool initTexturesDone;
    sofa::core::visual::VisualParams* vparams;
    sofa::core::visual::DrawToolGL   drawTool;

    sofa::helper::system::thread::ctime_t stepTime[10];
    sofa::helper::system::thread::ctime_t timeTicks;
    int frameCounter;
    double currentFPS;

    void beginStep();
    void endStep();
    void update();
    void calcProjection();

public:

    const std::string& getSceneFileName() const
    {
        return sceneFileName;
    }
    sofa::simulation::Simulation* getSimulation() const
    {
        return m_Simulation;
    }
    sofa::simulation::Node* getScene() const
    {
        return m_RootNode.get();
    }

};

#endif
