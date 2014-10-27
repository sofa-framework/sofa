#ifndef VIEWER_H
#define VIEWER_H

#include <QtQuick/QQuickItem>
#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFramebufferObject>

class Scene;
class Camera;
class QOpenGLShaderProgram;

class Viewer : public QQuickItem
{
    Q_OBJECT

public:
	Viewer();
	~Viewer();

public:
	Q_PROPERTY(Scene* scene MEMBER myScene NOTIFY sceneChanged)

signals:
	void sceneChanged(Scene* newScene);

public slots:
    void paint();
	void viewAll();
    void sync();

private slots:
	void handleSceneChanged(Scene* scene);
    void handleWindowChanged(QQuickWindow* window);

signals:
	void requestPaint();

private:
	Scene*						myScene;
	Camera*						myCamera;
	bool						myTexturesDirty;
	QOpenGLShaderProgram*		myProgram;
};

#endif // VIEWER_H
