#ifndef VIEWER_H
#define VIEWER_H

#include <QtQuick/QQuickItem>
#include <QVector3D>

class Scene;
class Camera;

class Viewer : public QQuickItem
{
    Q_OBJECT

public:
	explicit Viewer(QQuickItem* parent = 0);
	~Viewer();

	void classBegin();
	void componentComplete();

public:
	Q_PROPERTY(Scene* scene READ scene WRITE setScene NOTIFY sceneChanged)
	Q_PROPERTY(Camera* camera READ camera WRITE setCamera NOTIFY cameraChanged)

public:
	Scene* scene() const	{return myScene;}
	void setScene(Scene* newScene);

	Camera* camera() const	{return myCamera;}
	void setCamera(Camera* newCamera);

	Q_INVOKABLE QVector3D mapFromWorld(const QVector3D& point);
	Q_INVOKABLE QVector3D mapToWorld(const QVector3D& point);
	Q_INVOKABLE double computeDepth(const QVector3D& point);
	Q_INVOKABLE QVector3D projectOnViewPlane(const QVector3D& point, double depth);

signals:
	void sceneChanged(Scene* newScene);
	void scenePathChanged();
	void cameraChanged(Camera* newCamera);

public slots:
    void paint();
	void viewAll();

private slots:
	void handleSceneChanged(Scene* scene);
	void handleScenePathChanged();
    void handleWindowChanged(QQuickWindow* window);

signals:
	void requestPaint();

private:
	Scene*						myScene;
	Camera*						myCamera;
};

#endif // VIEWER_H
