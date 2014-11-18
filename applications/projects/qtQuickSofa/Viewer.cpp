#include <GL/glew.h>
#include "Viewer.h"
#include "Scene.h"
#include "Camera.h"

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/visual/DrawToolGL.h>
#include <QtQuick/qquickwindow.h>
#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLContext>
#include <QVector>
#include <QVector4D>
#include <QTime>
#include <QThread>

Viewer::Viewer() :
	myScene(0),
	myCamera(new Camera(this)),
	myTexturesDirty(false),
	myProgram(0)
{
	setFlag(QQuickItem::ItemHasContents);

	connect(this, &Viewer::sceneChanged, this, &Viewer::handleSceneChanged);
    connect(this, SIGNAL(windowChanged(QQuickWindow*)), this, SLOT(handleWindowChanged(QQuickWindow*)));
}

Viewer::~Viewer()
{
	
}

void Viewer::handleSceneChanged(Scene* scene)
{
	if(scene)
	{
		if(scene->isReady())
		{
			myTexturesDirty = true;
			viewAll();
		}

		connect(scene, &Scene::loaded, this, [&]() {myTexturesDirty = true; viewAll();});
	}
}

void Viewer::handleWindowChanged(QQuickWindow* window)
{
    if(window)
    {
        // Connect the beforeRendering signal to our paint function.
        // Since this call is executed on the rendering thread it must be
        // a Qt::DirectConnection
        connect(window, SIGNAL(beforeRendering()), this, SLOT(paint()), Qt::DirectConnection);
        connect(window, SIGNAL(beforeSynchronizing()), this, SLOT(sync()), Qt::DirectConnection);

        // If we allow QML to do the clearing, they would clear what we paint
        // and nothing would show.
        window->setClearBeforeRendering(false);
    }
}

void Viewer::paint()
{
	if(!myProgram)
	{
		glewInit();

        myProgram = new QOpenGLShaderProgram(this);
        myProgram->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                           "attribute highp vec4 vertices;"
                                           "varying highp vec2 coords;"
                                           "void main() {"
                                           "    gl_Position = vertices;"
                                           "    coords = vertices.xy;"
                                           "}");
        myProgram->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                           "uniform lowp float t;"
                                           "varying highp vec2 coords;"
                                           "void main() {"
                                           "    lowp float i = 1. - (pow(abs(coords.x), 4.) + pow(abs(coords.y), 4.));"
                                           "    i = smoothstep(t - 0.8, t + 0.8, i);"
                                           "    i = floor(i * 20.) / 20.;"
                                           "    gl_FragColor = vec4(coords * .5 + .5, i, i);"
                                           "}");

        //myProgram->bindAttributeLocation("vertices", 0);
        myProgram->link();
    }
	
	// we need to bind/release a shader once before drawing anything, currently i don't know why ...
	myProgram->bind();
	myProgram->release();

    // compute the correct viewer position
    QPointF pos = mapToScene(QPointF(0.0, 0.0));
    pos.setY(window()->height() - height() - pos.y()); // opengl has its Y coordinate inverted compared to qt

    // clear the viewer rectangle and just its area, not the whole OpenGL buffer
    glScissor(pos.x(), pos.y(), width(), height());
    glEnable(GL_SCISSOR_TEST);
    glClearColor(0.25, 0.25, 0.25, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);

	if(!myScene || !myScene->isReady())
		return;

    // set the viewer viewport
    glViewport(pos.x(), pos.y(), width(), height());

    glDisable(GL_DEPTH_TEST);

	myCamera->setAspectRatio(width() / (float) height());

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadMatrixf(myCamera->projection().constData());

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadMatrixf(myCamera->modelView().constData());

	GLfloat light_position[] = { 25.0, 0.0, 25.0, 1.0 }; // w = 0.0 => directional light ; w = 1.0 => point light (hemi) or spot light
	GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 0.0 };
	GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 0.0 };
	GLfloat light_specular[] = { 1.0, 1.0, 1.0, 0.0 };

	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);

	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
	glDepthMask(GL_TRUE);

	sofa::core::visual::VisualParams* _vparams = sofa::core::visual::VisualParams::defaultInstance();
	if(!_vparams->drawTool())
	{
		_vparams->drawTool() = new sofa::core::visual::DrawToolGL();
		_vparams->setSupported(sofa::core::visual::API_OpenGL);
	}

	if(_vparams)
	{
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

	if(myTexturesDirty)
	{
		myScene->sofaSimulation()->initTextures(myScene->sofaSimulation()->GetRoot().get());
		myTexturesDirty = false;
	}

	myScene->sofaSimulation()->updateVisual(myScene->sofaSimulation()->GetRoot().get());
	myScene->sofaSimulation()->draw(_vparams, myScene->sofaSimulation()->GetRoot().get());

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

void Viewer::viewAll()
{
	if(!myScene || !myScene->isReady())
		return;

	SReal min[3], max[3];
	myScene->sofaSimulation()->computeTotalBBox(myScene->sofaSimulation()->GetRoot().get(), min, max );

	myCamera->fit(QVector3D(min[0], min[1], min[2]), QVector3D(max[0], max[1], max[2]));
}

void Viewer::sync()
{
    
}
