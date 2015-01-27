#include "Camera.h"

#include <qqml.h>
#include <qmath.h>
#include <QDebug>

namespace sofa
{

namespace qtquick
{

Camera::Camera(QObject* parent) : QObject(parent),
	myDistanceToCenter(1.0),
	myZoomFactor(1.0),
	mySmoothedZoomFactor(1.0),
	mySmoothZoom(true),
	myZoomedDistanceToCenter(myDistanceToCenter),
    myZoomSpeed(1.025),
	myFovY(55.0),
	myAspectRatio(16.0 / 9.0),
	myZNear(0.1f),
	myZFar(1000.f),
	myProjection(),
	myView(),
	myModel(),
	myProjectionDirty(true),
	myModelDirty(true)
{
	connect(this, &Camera::smoothedZoomFactorChanged, this, &Camera::applyZoom);
}

Camera::~Camera()
{

}

void Camera::setZoomFactor(double newZoomFactor)
{
	if(newZoomFactor == myZoomFactor)
		return;

	myZoomFactor = newZoomFactor;

	zoomFactorChanged(newZoomFactor);
}

void Camera::setSmoothedZoomFactor(double newSmoothedZoomFactor)
{
	if(newSmoothedZoomFactor == mySmoothedZoomFactor)
		return;

	mySmoothedZoomFactor = newSmoothedZoomFactor;

	smoothedZoomFactorChanged(newSmoothedZoomFactor);
}

void Camera::setSmoothZoom(bool newSmoothZoom)
{
	if(newSmoothZoom == mySmoothZoom)
		return;

	mySmoothZoom = newSmoothZoom;

	smoothZoomChanged(newSmoothZoom);
}

const QMatrix4x4& Camera::projection() const
{
	if(myProjectionDirty) // update projection if needed
	{
		myProjection.setToIdentity();
		myProjection.perspective((float) myFovY, (float) myAspectRatio, myZNear, myZFar);
		myProjectionDirty = false;
	}

	return myProjection;
}

const QMatrix4x4& Camera::view() const
{
	return myView;
}

const QMatrix4x4& Camera::model() const
{
	if(myModelDirty)
	{
		myModel = myView.inverted();
		myModelDirty = false;
	}

	return myModel;
}

void Camera::setFovY(double fovY)
{
	if(fovY == myFovY)
		return;

	myFovY = fovY;

	myProjectionDirty = true;
}

void Camera::setAspectRatio(double aspectRatio)
{
	if(aspectRatio == myAspectRatio)
		return;

	myAspectRatio = aspectRatio;

	myProjectionDirty = true;
}

void Camera::setZNear(double zNear)
{
	if(zNear == myZNear)
		return;

	myZNear = zNear;

	myProjectionDirty = true;
}

void Camera::setZFar(double zFar)
{
	if(zFar == myZFar)
		return;

	myZFar = zFar;

	myProjectionDirty = true;
}

void Camera::fit(const QVector3D& min, const QVector3D& max)
{
	QVector3D center = (min + max) * 0.5;
	QVector3D diagonal = max - min;
	double radius = diagonal.length();
	double distance = 1.5 * radius / qTan(myFovY * M_PI / 180.0);

	QVector3D eye = center - direction().normalized() * distance;

	setZNear(qMax(0.01, distance - radius * 100));
	setZFar(distance + radius * 100);

	double distanceToCenter = (eye - center).length();
	if(distanceToCenter < 0.0001 || !(distanceToCenter == distanceToCenter)) // return if incorrect value, i.e < epsilon or nan
		return;

	myDistanceToCenter = distanceToCenter;
	myZoomedDistanceToCenter = myDistanceToCenter;

	bool smoothZoom = Camera::smoothZoom();
	setSmoothZoom(false);
	setZoomFactor(1.0);
	setSmoothZoom(smoothZoom);

	myView.setToIdentity();
	myView.lookAt(eye, center, up());
	myModelDirty = true;
}

void Camera::move(double x, double y, double z)
{
	double moveSpeed = myDistanceToCenter * 0.0133;

	myView.translate(-(right() * x + up() * y + direction() * z) * moveSpeed);

	myModelDirty = true;
}

void Camera::turn(double angleAroundX, double angleAroundY, double angleAroundZ)
{
	myView.rotate(angleAroundY, up());
	myView.rotate(angleAroundX, right());
	myView.rotate(angleAroundZ, direction());
	
	myModelDirty = true;
}

void Camera::zoom(double factor, bool smooth)
{
	factor *= myZoomFactor;

	if(factor <= 0.0)
		return;

	bool smoothZoom = Camera::smoothZoom();
	setSmoothZoom(smooth);
	setZoomFactor(factor);
	setSmoothZoom(smoothZoom);
}

void Camera::applyZoom()
{
	if(mySmoothedZoomFactor <= 0.0)
		return;

	double factor = 1.0 / mySmoothedZoomFactor;

	myView.translate(-(direction() * myZoomedDistanceToCenter - direction() * myDistanceToCenter * factor));
	myZoomedDistanceToCenter = myDistanceToCenter * factor;

	myModelDirty = true;
}

}

}
