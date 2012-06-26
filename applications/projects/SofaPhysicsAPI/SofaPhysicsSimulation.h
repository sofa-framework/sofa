#ifndef SOFAPHYSICSSIMULATION_H
#define SOFAPHYSICSSIMULATION_H

#include <sofa/simulation/common/Simulation.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/visual/DrawToolGL.h>
#include <sofa/component/visualmodel/InteractiveCamera.h>
#include <sofa/helper/gl/Texture.h>

class SofaPhysicsSimulation
{
public:
    SofaPhysicsSimulation();
    ~SofaPhysicsSimulation();

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

    const std::string& getSceneFileName() const;

    double getTimeStep() const;
    void   setTimeStep(double dt);
    double getCurrentFPS() const;

    class Impl;
protected:
    Impl* impl;
};

#endif // SOFASIMULATION_H
