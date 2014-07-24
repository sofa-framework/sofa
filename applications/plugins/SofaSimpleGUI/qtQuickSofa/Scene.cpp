#include "stdafx.h"
#include "Scene.h"

#include <QTimer>

Scene::Scene(QObject *parent) :
    QObject(parent)
  , _dt(0.04)
{
    _timer = new QTimer(this);
    connect(_timer, SIGNAL(timeout()), this, SLOT(step()));

}

void Scene::open(const QString& filename)
{
	QString finalFilename = QUrl(filename).toLocalFile().toLatin1().constData();
	if(finalFilename.isEmpty())
		finalFilename = filename;

	qDebug() << "Url:" << finalFilename;

    SofaScene::open(finalFilename.toLatin1().constData());
	if(!_currentFileName.empty())
		emit opened();
}

void Scene::reload()
{
	open(_currentFileName.c_str());
}

void Scene::step()
{
	if(!isLoaded())
		return;

    SofaScene::step(_dt);
    emit stepEnd();
}

void Scene::setTimeStep( SReal dt ){
    _dt = dt;
}

SReal Scene::dt() const
{
	return _dt;
}

bool Scene::isLoaded() const
{
	return !_currentFileName.empty();
}

bool Scene::isPlaying() const
{
	return _timer->isActive();
}

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

void Scene::pause()
{
	play(false);
}

void Scene::playpause()
{
    play( !_timer->isActive() );
}

void Scene::reset()
{
    SofaScene::reset();
    emit stepEnd();
}