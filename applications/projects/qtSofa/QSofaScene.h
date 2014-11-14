#ifndef QSOFASCENE_H
#define QSOFASCENE_H

#include <QObject>
#include <QTimer>

#include "SofaGL.h"

/**
 * @brief The QSofaScene class is a SofaScene which can be connected to other Qt objects, such as viewers, using signals and slots.
 * It contains the basic simulation functions, but no graphics capabilities.
 *
 * @author Francois Faure, 2014
 */
class QSofaScene : public QObject, public sofa::simplegui::SofaScene
{
    Q_OBJECT
public:
    explicit QSofaScene(QObject *parent = 0);

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
    void open(const char* filename);
//    /// re-open the current scene
//    void reload();
    /// Apply one simulation time step
    void step();
    /// Set the length of the simulation time step
    void setTimeStep( SReal dt );
    /// toggle play/pause
    void playpause();
    /// set play or pause, depending on the parameter
    void play( bool p=true );
    /// pause the animation
    void pause();
    /// restart at the beginning, without reloading the file
    void reset();
    /// print the graph on the standard output
    void printGraph();

public:
    /// Length of the simulation time step
    SReal dt() const;
    /// true if simulation is running, false if it is paused
    bool isPlaying() const;
private:
    SReal _dt;
    QTimer *_timer;


};

#endif // QSOFASCENE_H
