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

#include "Viewer.h"

#include <QtQuick/qquickwindow.h>
#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLContext>
#include <QVector>
#include <QVector4D>
#include <QTime>

Viewer::Viewer() :
    myRenderShaderProgram(0),
    myCompositionShaderProgram(0),
    myFramebuffer(0),
    m_t(0),
    m_thread_t(0)
{
    connect(this, SIGNAL(windowChanged(QQuickWindow*)), this, SLOT(handleWindowChanged(QQuickWindow*)));
}

void Viewer::setT(qreal t)
{
    if (t == m_t)
        return;
    m_t = t;
    emit tChanged();
    if (window())
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
        connect(win, SIGNAL(afterRendering()), this, SLOT(grab()), Qt::DirectConnection);
        connect(win, SIGNAL(beforeSynchronizing()), this, SLOT(sync()), Qt::DirectConnection);

        // If we allow QML to do the clearing, they would clear what we paint
        // and nothing would show.
        win->setClearBeforeRendering(false);
    }
}

void Viewer::paint()
{
    if(!myRenderShaderProgram)
    {
        myRenderShaderProgram = new QOpenGLShaderProgram();
        myRenderShaderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/resource/render.vs");
        myRenderShaderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/resource/render.fs");

        myRenderShaderProgram->bindAttributeLocation("vertices", 0);
        myRenderShaderProgram->link();

        connect(window()->openglContext(), SIGNAL(aboutToBeDestroyed()),
                this, SLOT(cleanup()), Qt::DirectConnection);
    }

    QSize size = QSize(width(), height());
    if(!myFramebuffer || size != myFramebuffer->size())
    {
        delete myFramebuffer;
        myFramebuffer = new QOpenGLFramebufferObject(size, QOpenGLFramebufferObject::CombinedDepthStencil);
    }

    myFramebuffer->bind();

    myRenderShaderProgram->bind();

    myRenderShaderProgram->enableAttributeArray(0);

    float values[] = {
        -1, -1,
        1, -1,
        -1, 1,
        1, 1
    };
    myRenderShaderProgram->setAttributeArray(0, GL_FLOAT, values, 2);
    myRenderShaderProgram->setUniformValue("Time", (float) m_thread_t);

    // compute the correct viewer position
    //QPointF pos = mapToScene(QPointF(0.0, 0.0));
    //pos.setY(window()->height() - height() - pos.y()); // opengl has its Y coordinate inverted compared to qt

    // clear the viewer rectangle and just its area, not the whole OpenGL buffer
    //glScissor(pos.x(), pos.y(), width(), height());
    //glEnable(GL_SCISSOR_TEST);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    //glDisable(GL_SCISSOR_TEST);

    // set the viewer viewport
    //glViewport(pos.x(), pos.y(), width(), height());
    glViewport(0, 0, width(), height());

    glDisable(GL_DEPTH_TEST);

    //glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    myRenderShaderProgram->disableAttributeArray(0);
    myRenderShaderProgram->release();

    myFramebuffer->release();

    if(window()->renderTarget())
        window()->renderTarget()->bind();
}

void Viewer::grab()
{
//    static int init = 0;
//    if(1 == init++)
//    {
//        QImage&& image = myFramebuffer->toImage();
//        if(image.save("viewer_output.png"))
//            qDebug() << "SAVED";
//        else
//            qDebug() << "NOT SAVED";
//    }

    // draw the grabbed scene
    if(!myCompositionShaderProgram)
    {
        myCompositionShaderProgram = new QOpenGLShaderProgram();
        myCompositionShaderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/resource/composition.vs");
        myCompositionShaderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/resource/composition.fs");

        myCompositionShaderProgram->bindAttributeLocation("vertices", 0);
        myCompositionShaderProgram->link();

        myCompositionShaderProgram->setUniformValue("uTexture", 0);

        connect(window()->openglContext(), SIGNAL(aboutToBeDestroyed()),
                this, SLOT(cleanup()), Qt::DirectConnection);
    }

    myCompositionShaderProgram->bind();

    myCompositionShaderProgram->enableAttributeArray(0);

    float values[] = {
        -1, -1,
        1, -1,
        -1, 1,
        1, 1
    };
    myCompositionShaderProgram->setAttributeArray(0, GL_FLOAT, values, 2);
    glBindTexture(GL_TEXTURE_2D, myFramebuffer->texture());

    QPointF pos = mapToScene(QPointF(0.0, 0.0));
    pos.setY(window()->height() - height() - pos.y()); // OpenGL has its Y coordinate inverted compared to qt
    glViewport(pos.x(), pos.y(), width(), height());

    glDisable(GL_DEPTH_TEST);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glBindTexture(GL_TEXTURE_2D, 0);
    myCompositionShaderProgram->disableAttributeArray(0);
    myCompositionShaderProgram->release();
}

void Viewer::saveScreenshot(const QString& imagePath)
{
    QString finalImagePath = imagePath;
    if(finalImagePath.isEmpty())
        finalImagePath = QString("screenshot_") + QDate::currentDate().toString("yyyy-MM-dd_") + QTime::currentTime().toString("hh.mm.ss") + QString(".png");

    QImage&& image = myFramebuffer->toImage();
    if(image.save(finalImagePath))
        qDebug() << "SAVED";
    else
        qDebug() << "NOT SAVED";
}

void Viewer::cleanup()
{
    if (myCompositionShaderProgram)
    {
        delete myCompositionShaderProgram;
        myCompositionShaderProgram = 0;
    }

    if (myRenderShaderProgram)
    {
        delete myRenderShaderProgram;
        myRenderShaderProgram = 0;
    }
}

void Viewer::sync()
{
    m_thread_t = m_t;
}
