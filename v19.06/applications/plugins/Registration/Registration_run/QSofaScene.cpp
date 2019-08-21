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