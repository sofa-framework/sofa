#include "Camera.h"

#include <qmath.h>
#include <QDebug>

Camera::Camera(QObject* parent) : QObject(parent),
	myFovY(55.0),
	myAspectRatio(16.0 / 9.0),
	myZNear(0.1),
	myZFar(1000.0),
	myProjection(),
	myModelView(),
	myProjectionDirty(true),
	myModelViewDirty(true)
{
	
}

const QMatrix4x4& Camera::projection() const
{
	if(myProjectionDirty) // update projection if needed
	{
		myProjection.perspective(myFovY, myAspectRatio, 0.1, 1000.0);
		myProjectionDirty = false;
	}

	return myProjection;
}
const QMatrix4x4& Camera::modelView() const
{
	if(myModelViewDirty) // update modelview if needed
	{
		
		myModelViewDirty = false;
	}

	return myModelView;
}

void Camera::setFovY(float fovY)
{
	if(fovY == myFovY)
		return;

	myFovY = fovY;

	myProjectionDirty = true;
}

void Camera::setAspectRatio(float aspectRatio)
{
	if(aspectRatio == myAspectRatio)
		return;

	myAspectRatio = aspectRatio;

	myProjectionDirty = true;
}

void Camera::setZNear(float zNear)
{
	if(zNear == myZNear)
		return;

	myZNear = zNear;

	myProjectionDirty = true;
}

void Camera::setZFar(float zFar)
{
	if(zFar == myZFar)
		return;

	myZFar = zFar;

	myProjectionDirty = true;
}

void Camera::fit(const QVector3D& min, const QVector3D& max)
{
	QVector3D center = (min + max) * 0.5f;
	QVector3D diagonal = max - min;
	float radius = diagonal.length();
	float distance = 1.5f * radius / qTan(myFovY * M_PI / 180.0f);

	QVector3D eye = center - direction().normalized() * distance;

	//myZNear = newDistance - newRadius * 1.5f;
	//myZFar  = newDistance + newRadius * 1.5f;

	myModelView.setToIdentity();
	myModelView.lookAt(eye, center, up());
	myModelViewDirty = false;
}