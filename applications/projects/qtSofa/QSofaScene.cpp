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
#include "QSofaScene.h"

QSofaScene::QSofaScene(QObject *parent) :
    QObject(parent)
  , _dt(0.04)
{
    _timer = new QTimer(this);
    connect(_timer, SIGNAL(timeout()), this, SLOT(step()));

}

void QSofaScene::open(const char *filename )
{
    SofaScene::open(filename);
    emit opened();
}


void QSofaScene::step()
{
    SofaScene::step(_dt);
    emit stepEnd();
}

void QSofaScene::setTimeStep( SReal dt ){
    _dt = dt;
}

SReal QSofaScene::dt() const { return _dt; }

bool QSofaScene::isPlaying() const { return _timer->isActive(); }

void QSofaScene::play( bool p )
{
    if( p ) {
        _timer->start(_dt);
        emit sigPlaying(true);
    }
    else {
        _timer->stop();
        emit sigPlaying(false);
    }
}

void QSofaScene::pause() { play(false); }

void QSofaScene::playpause()
{
    play( !_timer->isActive() );
}



void QSofaScene::reset()
{
    SofaScene::reset();
    emit stepEnd();
}

void QSofaScene::printGraph(){
    SofaScene::printGraph();
}
