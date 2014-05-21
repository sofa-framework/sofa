#ifndef SCENE_H
#define SCENE_H

#include <QObject>
#include "SofaScene.h"
//#include <sofa/simulation/graph/DAGSimulation.h>

class QTimer;
//typedef sofa::simulation::graph::DAGSimulation SofaSimulation;

class Scene : public QObject, public sofa::newgui::SofaScene
{
    Q_OBJECT

public:
    explicit Scene(QObject *parent = 0);

signals:
    /// Sent after step() or reset() to notify that redraw is needed
    void stepEnd();
    /// new state: play or pause (for changing the icon)
    void sigPlaying(bool playing);
    /// scene was just opened
    void opened();

public slots:
    /**
     * @brief Clear the current scene and open a new one
     * @param filename new scene to open
     */
    Q_INVOKABLE void open(const char* filename);
    /// re-open the current scene
    Q_INVOKABLE void reload();
    /// Apply one simulation time step
    Q_INVOKABLE void step();
    /// Set the length of the simulation time step
    Q_INVOKABLE void setTimeStep(SReal dt);
    /// toggle play/pause
    Q_INVOKABLE void playpause();
    /// set play or pause, depending on the parameter
    Q_INVOKABLE void play( bool p=true );
    /// pause the animation
    Q_INVOKABLE void pause();
    /// restart at the beginning, without reloading the file
    Q_INVOKABLE void reset();

public:
    /// Length of the simulation time step
    Q_INVOKABLE SReal dt() const;
    /// true if simulation is running, false if it is paused
    Q_INVOKABLE bool isPlaying() const;

private:
    SReal _dt;
    QTimer *_timer;

};

#endif // SCENE_H