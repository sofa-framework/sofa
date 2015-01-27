#include "Viewer.h"
#include "Scene.h"
#include "Camera.h"

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/visual/DrawToolGL.h>

#include <QtQuick/qquickwindow.h>
#include <QQmlEngine>
#include <QQmlContext>
#include <QVector>
#include <QVector4D>
#include <QTime>
#include <QThread>
#include <qqml.h>
#include <qmath.h>

namespace sofa
{

namespace qtquick
{

Viewer::Viewer(QQuickItem* parent) : QQuickItem(parent),
	myScene(0),
	myCamera(0)
{
    setFlag(QQuickItem::ItemHasContents);

	connect(this, &Viewer::sceneChanged, this, &Viewer::handleSceneChanged);
	connect(this, &Viewer::scenePathChanged, this, &Viewer::handleScenePathChanged);
    connect(this, SIGNAL(windowChanged(QQuickWindow*)), this, SLOT(handleWindowChanged(QQuickWindow*)));
}

Viewer::~Viewer()
{
	/*sofa::core::visual::VisualParams* _vparams = sofa::core::visual::VisualParams::defaultInstance();
	if(_vparams && _vparams->drawTool())
	{
		delete _vparams->drawTool();
		_vparams->drawTool() = 0;
	}*/
}

void Viewer::classBegin()
{
	if(!myScene)
	{
		QQmlContext* context = QQmlEngine::contextForObject(this);
		if(context)
		{
			QVariant sceneVariant = context->contextProperty("scene");
			if(sceneVariant.canConvert(QMetaType::QObjectStar))
			{
				Scene* scene = qobject_cast<Scene*>(sceneVariant.value<QObject*>());
				if(scene)
					setScene(scene);
			}
		}
	}
}

void Viewer::componentComplete()
{

}

void Viewer::setScene(Scene* newScene)
{
	if(newScene == myScene)
		return;

	myScene = newScene;

	sceneChanged(newScene);
}

void Viewer::setCamera(Camera* newCamera)
{
	if(newCamera == myCamera)
		return;

	myCamera = newCamera;

	cameraChanged(newCamera);
}

QVector3D Viewer::mapFromWorld(const QVector3D& point)
{
	if(!myCamera)
		return QVector3D();

	QVector4D nsPosition = (myCamera->projection() * myCamera->view() * QVector4D(point, 1.0));
	nsPosition /= nsPosition.w();

	return QVector3D((nsPosition.x() * 0.5 + 0.5) * qCeil(width()) + 0.5, qCeil(height()) - (nsPosition.y() * 0.5 + 0.5) * qCeil(height()) + 0.5, (nsPosition.z() * 0.5 + 0.5));
}

QVector3D Viewer::mapToWorld(const QVector3D& point)
{
	if(!myCamera)
		return QVector3D();

	QVector3D nsPosition = QVector3D(point.x() / (double) qCeil(width()) * 2.0 - 1.0, (1.0 - point.y() / (double) qCeil(height())) * 2.0 - 1.0, point.z() * 2.0 - 1.0);
	QVector4D vsPosition = myCamera->projection().inverted() * QVector4D(nsPosition, 1.0);
	vsPosition /= vsPosition.w();

	return (myCamera->model() * vsPosition).toVector3D();
}

double Viewer::computeDepth(const QVector3D& point)
{
	if(!myCamera)
		return 0.0;

	QVector4D csPosition = myCamera->projection() * myCamera->view() * QVector4D(point, 1.0);

	return (csPosition.z() / csPosition.w()) * 0.5 + 0.5;
}

QVector3D Viewer::projectOnViewPlane(const QVector3D& point, double depth)
{
	if(!myCamera)
		return QVector3D();

	QVector4D csPosition = myCamera->projection() * myCamera->view() * QVector4D(point, 1.0);
	QVector4D nsPosition = csPosition / csPosition.w();

	return mapToWorld(QVector3D((nsPosition.x() * 0.5 + 0.5) * qCeil(width()), (1.0 - (nsPosition.y() * 0.5 + 0.5)) * qCeil(height()), depth));
}

void Viewer::handleSceneChanged(Scene* scene)
{
	if(scene)
	{
		if(scene->isReady())
			scenePathChanged();

		connect(scene, &Scene::loaded, this, &Viewer::scenePathChanged);
	}
}

void Viewer::handleScenePathChanged()
{
	if(!myScene || !myScene->isReady())
		return;

	viewAll();
}

void Viewer::handleWindowChanged(QQuickWindow* window)
{
    if(window)
    {
        window->setClearBeforeRendering(false);
        connect(window, SIGNAL(beforeRendering()), this, SLOT(paint()), Qt::DirectConnection);
    }
}

void Viewer::paint()
{
    if(!window())
        return;

    // compute the correct viewer position and size
    QPointF realPos(mapToScene(QPointF(0.0, 0.0)));
    realPos.setX(realPos.x() * window()->devicePixelRatio());
    realPos.setY((window()->height() - height()) * window()->devicePixelRatio() - realPos.y() * window()->devicePixelRatio());  // OpenGL has its Y coordinate inverted compared to Qt

    QPoint pos(qFloor(realPos.x()), qFloor(realPos.y()));
    QSize size((qCeil(width()) + qCeil(pos.x() - realPos.x())) * window()->devicePixelRatio(), (qCeil((height()) + qCeil(pos.y() - realPos.y())) * window()->devicePixelRatio()));
	if(!size.isValid())
        return;

    // clear the viewer rectangle and just its area, not the whole OpenGL buffer
    glScissor(pos.x(), pos.y(), size.width(), size.height());
    glEnable(GL_SCISSOR_TEST);
    glClearColor(0.25, 0.25, 0.25, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);

	if(!myScene || !myScene->isReady())
		return;

	if(!myScene->isInit())
		myScene->init();

    // set the viewer viewport
    glViewport(pos.x(), pos.y(), size.width(), size.height());

    glDisable(GL_DEPTH_TEST);

	if(!myCamera)
		return;

	myCamera->setAspectRatio(size.width() / (double) size.height());

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadMatrixf(myCamera->projection().constData());

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadMatrixf(myCamera->view().constData());
	
	// Get the camera position
	QVector3D camera_position(myCamera->eye());
	float cx = camera_position[0]; 
	float cy = camera_position[1];
	float cz = camera_position[2];

	float light_position[] = { cx, cy, cz, 0.0f};	// Use of the camera position for light
	float light_ambient[]  = { 0.0f, 0.0f, 0.0f, 0.0f};
	float light_diffuse[]  = { 1.0f, 1.0f, 1.0f, 0.0f};
	float light_specular[] = { 1.0f, 1.0f, 1.0f, 0.0f};

	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);

	glDisable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	// qt does not release its shader program and we do not use one so we have to release the current bound program
	glUseProgram(0);

	// prepare the sofa visual params
	sofa::core::visual::VisualParams* _vparams = sofa::core::visual::VisualParams::defaultInstance();
	if(_vparams)
	{
		if(!_vparams->drawTool())
		{
			_vparams->drawTool() = new sofa::core::visual::DrawToolGL();
			_vparams->setSupported(sofa::core::visual::API_OpenGL);
		}

		GLint _viewport[4];
		GLdouble _mvmatrix[16], _projmatrix[16];

		glGetIntegerv (GL_VIEWPORT, _viewport);
		glGetDoublev (GL_MODELVIEW_MATRIX, _mvmatrix);
		glGetDoublev (GL_PROJECTION_MATRIX, _projmatrix);

		_vparams->viewport() = sofa::helper::fixed_array<int, 4>(_viewport[0], _viewport[1], _viewport[2], _viewport[3]);
		_vparams->sceneBBox() = myScene->sofaSimulation()->GetRoot()->f_bbox.getValue();
		_vparams->setProjectionMatrix(_projmatrix);
		_vparams->setModelViewMatrix(_mvmatrix);
	}

    myScene->draw();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

void Viewer::viewAll()
{
	if(!myCamera || !myScene || !myScene->isReady())
		return;

	QVector3D min, max;
	myScene->computeBoundingBox(min, max);

	myCamera->fit(min, max);
}

}

}
