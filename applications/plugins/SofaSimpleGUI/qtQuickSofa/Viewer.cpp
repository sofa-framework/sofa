/****************************************************************************
**
** Copyright (C) 2013 Digia Plc and/or its subsidiary(-ies).
** Contact: http://www.qt-project.org/legal
**
** This file is part of the demonstration applications of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:LGPL$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and Digia.  For licensing terms and
** conditions see http://qt.digia.com/licensing.  For further information
** use the contact form at http://qt.digia.com/contact-us.
**
** GNU Lesser General Public License Usage
** Alternatively, this file may be used under the terms of the GNU Lesser
** General Public License version 2.1 as published by the Free Software
** Foundation and appearing in the file LICENSE.LGPL included in the
** packaging of this file.  Please review the following information to
** ensure the GNU Lesser General Public License version 2.1 requirements
** will be met: http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
**
** In addition, as a special exception, Digia gives you certain additional
** rights.  These rights are described in the Digia Qt LGPL Exception
** version 1.1, included in the file LGPL_EXCEPTION.txt in this package.
**
** GNU General Public License Usage
** Alternatively, this file may be used under the terms of the GNU
** General Public License version 3.0 as published by the Free Software
** Foundation and appearing in the file LICENSE.GPL included in the
** packaging of this file.  Please review the following information to
** ensure the GNU General Public License version 3.0 requirements will be
** met: http://www.gnu.org/copyleft/gpl.html.
**
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include "stdafx.h"

#include "../SofaGL.h"
#include "Viewer.h"
#include "Scene.h"

#include <QtQuick/qquickwindow.h>
#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLContext>
#include <QVector>
#include <QVector4D>
#include <QTime>
#include <QThread>

Viewer::Viewer() :
	myScene(0),
	mySofaGL(0),
	myProgram(0)
{
	setFlag(QQuickItem::ItemHasContents);

    connect(this, SIGNAL(windowChanged(QQuickWindow*)), this, SLOT(handleWindowChanged(QQuickWindow*)));
}

Viewer::~Viewer()
{
	delete myProgram;
}

void Viewer::setScene(Scene* scene)
{
	if(!scene || scene == myScene)
		return;

	qDebug() << "setScene:" << myScene;

	delete myScene;
	myScene = scene;

	delete mySofaGL;
	mySofaGL = 0;

	connect(myScene, SIGNAL(opened()), this, SLOT(sceneModification()));

	if(window())
		window()->update();
}

void Viewer::sceneModification()
{
	delete mySofaGL;
	mySofaGL = 0;

	// do not init a new SofaGL here since we may not have a valid GL context

	if(window())
		window()->update();
}

void Viewer::handleWindowChanged(QQuickWindow *win)
{
    if(win)
    {
        // Connect the beforeRendering signal to our paint function.
        // Since this call is executed on the rendering thread it must be
        // a Qt::DirectConnection
        connect(win, SIGNAL(beforeRendering()), this, SLOT(paint()), Qt::DirectConnection);
        connect(win, SIGNAL(beforeSynchronizing()), this, SLOT(sync()), Qt::DirectConnection);

        // If we allow QML to do the clearing, they would clear what we paint
        // and nothing would show.
        win->setClearBeforeRendering(false);
    }
}

void Viewer::paint()
{
	qDebug() << "a" << QThread::currentThreadId() << flush;
	
	if(!myScene)
	{
		myScene = new Scene();
		myScene->open("C:/MyFiles/Sofa/examples/Demos/caduceus.scn");
	}

	qDebug() << "a1" << flush;
	if(!myProgram)
	{
        myProgram = new QOpenGLShaderProgram();
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
	
	// we need to bind/release a shader once before drawing anything, for now i don't know why ...
	myProgram->bind();
	myProgram->release();

	qDebug() << "b" << mySofaGL << myScene << flush;
	if(!mySofaGL && myScene && myScene->isLoaded())
	{
		sofa::newgui::SofaScene* sofaScene = dynamic_cast<sofa::newgui::SofaScene*>(myScene);
		if(sofaScene)
		{
			delete mySofaGL;
			mySofaGL = new sofa::newgui::SofaGL(sofaScene);
			mySofaGL->init();
		}
	}
	qDebug() << "h" << flush;

    // compute the correct viewer position
    QPointF pos = mapToScene(QPointF(0.0, 0.0));
    pos.setY(window()->height() - height() - pos.y()); // opengl has its Y coordinate inverted compared to qt

    // clear the viewer rectangle and just its area, not the whole OpenGL buffer
    glScissor(pos.x(), pos.y(), width(), height());
    glEnable(GL_SCISSOR_TEST);
    glClearColor(0.25, 0.25, 0.25, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);

    // set the viewer viewport
    glViewport(pos.x(), pos.y(), width(), height());
    //glViewport(0, 0, width(), height());

    glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//glOrtho(-20.0, 20.0, -20.0, 20.0, -100.0, 100.0);
	gluPerspective(55.0, (GLfloat) width()/(GLfloat) height(), 0.1, 1000.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslated(0.0, -30.0, -100.0);

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

	if(mySofaGL && myScene && myScene->isLoaded())
	{
		qDebug() << "i" << flush;
		myScene->step();
		qDebug() << "j" << flush;
		mySofaGL->draw();
		qDebug() << "k" << flush;
	}

	qDebug() << "l" << flush;
}

void Viewer::sync()
{
    
}
