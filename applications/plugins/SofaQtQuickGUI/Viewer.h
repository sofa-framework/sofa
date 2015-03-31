#ifndef VIEWER_H
#define VIEWER_H

#include "SofaQtQuickGUI.h"
#include <QtQuick/QQuickItem>
#include <QVector3D>
#include <QVector4D>
#include <QiMAGE>
#include <QColor>

class QOpenGLFramebufferObject;

namespace sofa
{

namespace qtquick
{

class Scene;
class Camera;

class SOFA_SOFAQTQUICKGUI_API Viewer : public QQuickItem
{
    Q_OBJECT

public:
    explicit Viewer(QQuickItem* parent = 0);
	~Viewer();

	void classBegin();
	void componentComplete();

public:
    Q_PROPERTY(sofa::qtquick::Scene* scene READ scene WRITE setScene NOTIFY sceneChanged)
    Q_PROPERTY(sofa::qtquick::Camera* camera READ camera WRITE setCamera NOTIFY cameraChanged)

    Q_PROPERTY(QColor backgroundColor READ backgroundColor WRITE setBackgroundColor NOTIFY backgroundColorChanged)
    Q_PROPERTY(QUrl backgroundImageSource READ backgroundImageSource WRITE setBackgroundImageSource NOTIFY backgroundImageSourceChanged)
    Q_PROPERTY(bool wireframe READ wireframe WRITE setWireframe NOTIFY wireframeChanged)
    Q_PROPERTY(bool culling READ culling WRITE setCulling NOTIFY cullingChanged)
    Q_PROPERTY(bool antialiasing READ antialiasing WRITE setAntialiasing NOTIFY antialiasingChanged)
    Q_PROPERTY(int texture READ texture NOTIFY textureChanged)

public:
    Scene* scene() const        {return myScene;}
    void setScene(Scene* newScene);

    Camera* camera() const      {return myCamera;}
    void setCamera(Camera* newCamera);

    QColor backgroundColor() const	{return myBackgroundColor;}
    void setBackgroundColor(QColor newBackgroundColor);

    QUrl backgroundImageSource() const	{return myBackgroundImageSource;}
    void setBackgroundImageSource(QUrl newBackgroundImageSource);

    bool wireframe() const      {return myWireframe;}
    void setWireframe(bool newWireframe);

    bool culling() const        {return myCulling;}
    void setCulling(bool newCulling);

    bool antialiasing() const        {return myAntialiasing;}
    void setAntialiasing(bool newAntialiasing);

    Q_INVOKABLE unsigned int texture() const;

    Q_INVOKABLE QVector3D mapFromWorld(const QVector3D& wsPoint);
    Q_INVOKABLE QVector3D mapToWorld(const QVector3D& ssPoint);

    Q_INVOKABLE QVector4D projectOnGeometry(const QPointF& ssPoint);    // .w == 0 => background hit ; .w == 1 => geometry hit

signals:
    void sceneChanged(sofa::qtquick::Scene* newScene);
	void scenePathChanged();
    void cameraChanged(sofa::qtquick::Camera* newCamera);
    void backgroundColorChanged(QColor newBackgroundColor);
    void backgroundImageSourceChanged(QUrl newBackgroundImageSource);
    void wireframeChanged(bool newWireframe);
    void cullingChanged(bool newCulling);
    void antialiasingChanged(bool newAntialiasing);
    void textureChanged(bool newTexture);

public slots:
    void paint();
	void viewAll();

private slots:
	void handleSceneChanged(Scene* scene);
	void handleScenePathChanged();
    void handleBackgroundImageSourceChanged(QUrl newBackgroundImageSource);
    void handleWindowChanged(QQuickWindow* window);

private:
	Scene*						myScene;
	Camera*						myCamera;
    QColor                      myBackgroundColor;
    QUrl                        myBackgroundImageSource;
    QImage                      myBackgroundImage;
    bool                        myWireframe;
    bool                        myCulling;
    bool                        myAntialiasing;
    QOpenGLFramebufferObject*   myFBO;

};

}

}

#endif // VIEWER_H
