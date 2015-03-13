#include "Camera.h"

#include <qqml.h>
#include <qmath.h>
#include <QDebug>

namespace sofa
{

namespace qtquick
{

Camera::Camera(QObject* parent) : QObject(parent),
	myFovY(55.0),
	myAspectRatio(16.0 / 9.0),
	myZNear(0.1f),
	myZFar(1000.f),
	myProjection(),
	myView(),
	myModel(),
    myTarget(),
	myProjectionDirty(true),
    myViewDirty(true)
{

}

Camera::~Camera()
{

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
    if(myViewDirty)
    {
        myView = model().inverted();
        myViewDirty = false;
    }

	return myView;
}

const QMatrix4x4& Camera::model() const
{
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
    myTarget = (min + max) * 0.5;
	QVector3D diagonal = max - min;
	double radius = diagonal.length();
    double distance = 1.5 * radius / qTan(myFovY * M_PI / 180.0);

    QVector3D eye = myTarget - direction() * distance;

    setZNear(qMax(0.01, distance - radius * 100));
    setZFar(distance + radius * 100);

    if(distance < 0.0001 || !(distance == distance)) // return if incorrect value, i.e < epsilon or nan
		return;

    myView.setToIdentity();
    myView.lookAt(eye, myTarget, up());
    myModel = myView.inverted();

    myViewDirty = false;
}

void Camera::move(double x, double y, double z)
{
    QVector3D translationVector(right() * x + up() * y + direction() * z);

    QMatrix4x4 translation;
    translation.translate(translationVector);

    myModel = translation * myModel;
    myTarget += translationVector;

    myViewDirty = true;
}

void Camera::turn(double angleAroundX, double angleAroundY, double angleAroundZ)
{
    QMatrix4x4 rotation;
    rotation.translate( target());
    rotation.rotate(angleAroundY, up());
    rotation.rotate(angleAroundX, right());
    rotation.rotate(angleAroundZ, direction()); // TODO: check rotation order
    rotation.translate(-target());

    myModel = rotation * myModel;

    myViewDirty = true;
}

void Camera::zoom(double factor)
{
	if(factor <= 0.0)
		return;

    QVector3D translationVector(target() - eye());

    factor = 1.0 / factor;
    translationVector *= (1.0 - factor);

    // limit zoom to znear
    if((eye() + translationVector - target()).length() <= myZNear)
        translationVector = (target() - eye()) + (eye() - target()).normalized() * (myZNear + std::numeric_limits<float>::epsilon());

    QMatrix4x4 translation;
    translation.translate(translationVector);

    myModel = translation * myModel;

    myViewDirty = true;
}

}

}
