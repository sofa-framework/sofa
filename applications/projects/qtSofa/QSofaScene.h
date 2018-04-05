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
