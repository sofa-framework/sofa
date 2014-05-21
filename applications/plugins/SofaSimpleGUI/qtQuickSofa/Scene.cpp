#ifdef FALSE

#include "Scene.h"

Scene::Scene(QObject *parent) :
    QObject(parent)
  , _dt(0.04)
{
    _timer = new QTimer(this);
    connect(_timer, SIGNAL(timeout()), this, SLOT(step()));

}

void Scene::open(const char *filename )
{
    SofaScene::open(filename);
    emit opened();
}

void Scene::reload() { open(_currentFileName.c_str()); }


void Scene::step()
{
    SofaScene::step(_dt);
    emit stepEnd();
}

void Scene::setTimeStep( SReal dt ){
    _dt = dt;
}

SReal Scene::dt() const { return _dt; }

bool Scene::isPlaying() const { return _timer->isActive(); }

void Scene::play( bool p )
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

void Scene::pause() { play(false); }

void Scene::playpause()
{
    play( !_timer->isActive() );
}

void Scene::reset()
{
    SofaScene::reset();
    emit stepEnd();
}

#endif
