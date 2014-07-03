#include "Window.h"
#include <QTime>
#include <QOpenGLContext>
#include <QtGui/QImage>

Window::Window(QWindow* parent) : QQuickWindow(parent),
    myFramebuffer(0),
    myCompositionShaderProgram(0)
{
    connect(this, SIGNAL(beforeRendering()), this, SLOT(paint()), Qt::DirectConnection);
    connect(this, SIGNAL(afterRendering()), this, SLOT(grab()), Qt::DirectConnection);

    // since we draw our scene in a fbo it is not useful anymore, let qt clear the render buffer for us
    //setClearBeforeRendering(false);
}

Window::~Window()
{

}

void Window::createFrameBuffer()
{
	delete myFramebuffer;
	myFramebuffer = new QOpenGLFramebufferObject(size(), QOpenGLFramebufferObject::CombinedDepthStencil);

    setRenderTarget(myFramebuffer);
}

void Window::paint()
{
    if(!myFramebuffer || size() != myFramebuffer->size())
        createFrameBuffer();

    myFramebuffer->bind();
}

void Window::grab()
{
    myFramebuffer->release();

//    static bool init = false;
//    if(!init)
//    {
//        init = true;

//        QImage&& image = myFramebuffer->toImage();
//        if(image.save("window_output.png"))
//            qDebug() << "SAVED";
//        else
//            qDebug() << "NOT SAVED";
//    }

    // draw the grabbed scene
    if(!myCompositionShaderProgram)
    {
        myCompositionShaderProgram = new QOpenGLShaderProgram();
        myCompositionShaderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/resource/shaders/composition.vs");
        myCompositionShaderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/resource/shaders/composition.fs");

        myCompositionShaderProgram->bindAttributeLocation("vertices", 0);
        myCompositionShaderProgram->link();

        myCompositionShaderProgram->setUniformValue("uTexture", 0);

        connect(openglContext(), SIGNAL(aboutToBeDestroyed()), this, SLOT(cleanup()), Qt::DirectConnection);
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

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, width(), height());

    glDisable(GL_DEPTH_TEST);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glBindTexture(GL_TEXTURE_2D, 0);
    myCompositionShaderProgram->disableAttributeArray(0);
    myCompositionShaderProgram->release();
}

void Window::saveScreenshot(const QString& imagePath)
{
    QString finalImagePath = imagePath;
    if(finalImagePath.isEmpty())
        finalImagePath = QString("screenshot_") + QDate::currentDate().toString("yyyy-MM-dd_") + QTime::currentTime().toString("hh.mm.ss") + QString(".png");

    QImage image = myFramebuffer->toImage(); // sorry but clang version 3.4-1ubuntu1 in debug mode fails on QImage&&
    if(image.save(finalImagePath))
        qDebug() << "SAVED";
    else
        qDebug() << "NOT SAVED";
    qDebug() << "Window::saveScreenshot temporarily disabled by FF since clang34-debug issues an error on: QImage&& ";
}

void Window::cleanup()
{
    if (myCompositionShaderProgram)
    {
        delete myCompositionShaderProgram;
        myCompositionShaderProgram = 0;
    }
}
