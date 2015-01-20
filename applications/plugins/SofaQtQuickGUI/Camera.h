#ifndef CAMERA_H
#define CAMERA_H

#include "SofaQtQuickGUI.h"
#include <QObject>
#include <QMatrix4x4>
#include <QVector3D>
#include <QPoint>

namespace sofa
{

namespace qtquick
{

class SOFA_SOFAQTQUICKGUI_API Camera : public QObject
{
	Q_OBJECT

public:
	explicit Camera(QObject* parent = 0);
	~Camera();

public:
	Q_PROPERTY(double zoomFactor READ zoomFactor WRITE setZoomFactor NOTIFY zoomFactorChanged)
	Q_PROPERTY(double smoothedZoomFactor READ smoothedZoomFactor WRITE setSmoothedZoomFactor NOTIFY smoothedZoomFactorChanged)
	Q_PROPERTY(bool smoothZoom READ smoothZoom WRITE setSmoothZoom NOTIFY smoothZoomChanged)
	Q_PROPERTY(double zoomSpeed MEMBER myZoomSpeed NOTIFY zoomSpeedChanged)

public:
	double zoomFactor() const				{return myZoomFactor;}
	void setZoomFactor(double newZoomFactor);

	double smoothedZoomFactor() const		{return mySmoothedZoomFactor;}
	void setSmoothedZoomFactor(double newSmoothedZoomFactor);

	double smoothZoom() const				{return mySmoothZoom;}
	void setSmoothZoom(bool newSmoothZoom);

signals:
	void zoomFactorChanged(double newZoomFactor);
	void smoothedZoomFactorChanged(double newSmoothedZoomFactor);
	void smoothZoomChanged(bool newEnableSmoothZoom);
	void zoomSpeedChanged(double newZoomSpeed);

public:
	const QMatrix4x4& projection() const;
	const QMatrix4x4& view() const;
	const QMatrix4x4& model() const;

	Q_INVOKABLE QVector3D eye() const				{return  model().column(3).toVector3D();}
	Q_INVOKABLE QVector3D direction() const			{return -model().column(2).toVector3D();}
	Q_INVOKABLE QVector3D up() const				{return  model().column(1).toVector3D();}
	Q_INVOKABLE QVector3D right() const				{return  model().column(0).toVector3D();}

public:
	void setFovY(double fovY);
	void setAspectRatio(double aspectRatio);
	void setZNear(double zNear);
	void setZFar(double zFar);

	void fit(const QVector3D& min, const QVector3D& max);

public slots:
	void move(double x, double y, double z);
	void turn(double angleAroundX, double angleAroundY, double angleAroundZ);
	void zoom(double factor, bool smooth = true);

private:
	void applyZoom();

private:
	double				myDistanceToCenter;
	double				myZoomFactor;
	double				mySmoothedZoomFactor;
	bool				mySmoothZoom;
	double				myZoomedDistanceToCenter;
	double				myZoomSpeed;

	double				myFovY;
	double				myAspectRatio;
	double				myZNear;
	double				myZFar;

	mutable QMatrix4x4	myProjection;
	QMatrix4x4			myView;
	mutable QMatrix4x4	myModel;

	mutable bool		myProjectionDirty;
	mutable bool		myModelDirty;
};

}

}

#endif // CAMERA_H
