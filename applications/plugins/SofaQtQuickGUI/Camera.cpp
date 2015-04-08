#include "Camera.h"

#include <qqml.h>
#include <qmath.h>
#include <QDebug>

namespace sofa
{

namespace qtquick
{

Camera::Camera(QObject* parent) : QObject(parent),
    myOrthographic(false),
    myOrthoLeft(-1.0),
    myOrthoRight(1.0),
    myOrthoBottom(-1.0),
    myOrthoTop(1.0),
    myPerspectiveFovY(55.0),
    myPerspectiveAspectRatio(16.0 / 9.0),
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

void Camera::setOrthographic(bool orthographic)
{
    if(orthographic == myOrthographic)
        return;

    myOrthographic = orthographic;

    if(myOrthographic)
        computeOrthographic();

    myProjectionDirty = true;

    orthographicChanged();
}

void Camera::setTarget(const QVector3D& target)
{
    if(target == myTarget)
        return;

    myTarget = target;

    computeModel();

    targetChanged();
}

const QMatrix4x4& Camera::projection() const
{
	if(myProjectionDirty) // update projection if needed
	{
		myProjection.setToIdentity();
        if(myOrthographic)
            myProjection.ortho(myOrthoLeft, myOrthoRight, myOrthoBottom, myOrthoTop, myZNear, myZFar);
        else
            myProjection.perspective((float) myPerspectiveFovY, (float) myPerspectiveAspectRatio, myZNear, myZFar);
		myProjectionDirty = false;
	}

	return myProjection;
}

const QMatrix4x4& Camera::view() const
{
    if(myViewDirty) // update view if needed
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

void Camera::lookAt(const QVector3D& eye, const QVector3D& target, const QVector3D& up)
{
    myView.setToIdentity();
    myView.lookAt(eye, target, up);
    myModel = myView.inverted();

    myViewDirty = false;
}

double Camera::computeDepth(const QVector3D& point)
{
    QVector4D csPosition = projection() * view() * QVector4D(point, 1.0);

    return csPosition.z() / csPosition.w();
}

QVector3D Camera::projectOnViewPlane(const QVector3D& point, double depth)
{
    QVector4D csPosition = projection() * view() * QVector4D(point, 1.0);
    QVector4D nsPosition = csPosition / csPosition.w();

    csPosition = projection().inverted() * QVector4D(nsPosition.x(), nsPosition.y(), depth, 1.0);
    QVector4D vsPosition = csPosition / csPosition.w();

    return (model() * vsPosition).toVector3D();
}

void Camera::viewFromFront()
{
    QVector3D eye = myTarget + QVector3D( 0.0, 0.0, 1.0) * (myTarget - Camera::eye()).length();

    lookAt(eye, myTarget, QVector3D( 0.0, 1.0, 0.0));
}

void Camera::viewFromBack()
{
    QVector3D eye = myTarget + QVector3D( 0.0, 0.0,-1.0) * (myTarget - Camera::eye()).length();

    lookAt(eye, myTarget, QVector3D( 0.0, 1.0, 0.0));
}

void Camera::viewFromLeft()
{
    QVector3D eye = myTarget + QVector3D(-1.0, 0.0, 0.0) * (myTarget - Camera::eye()).length();

    lookAt(eye, myTarget, QVector3D( 0.0, 1.0, 0.0));
}

void Camera::viewFromRight()
{
    QVector3D eye = myTarget + QVector3D( 1.0, 0.0, 0.0) * (myTarget - Camera::eye()).length();

    lookAt(eye, myTarget, QVector3D( 0.0, 1.0, 0.0));
}

void Camera::viewFromTop()
{
    QVector3D eye = myTarget + QVector3D( 0.0, 1.0, 0.0) * (myTarget - Camera::eye()).length();

    lookAt(eye, myTarget, QVector3D( 0.0, 0.0,-1.0));
}

void Camera::viewFromBottom()
{
    QVector3D eye = myTarget + QVector3D( 0.0,-1.0, 0.0) * (myTarget - Camera::eye()).length();

    lookAt(eye, myTarget, QVector3D( 0.0, 0.0, 1.0));
}

void Camera::viewIsometric()
{
    QVector3D eye = QVector3D( 0.0, 0.0, 1.0);

    QMatrix4x4 rotation;
    rotation.rotate(45.0, 0.0, 1.0, 0.0);
    rotation.rotate(34.0,-1.0, 0.0, 0.0); // TODO: angle ?
    eye = rotation.map(eye);

    eye = myTarget + eye * (myTarget - Camera::eye()).length();

    lookAt(eye, myTarget, QVector3D( 0.0, 1.0, 0.0));
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

    if(orthographic())
        computeOrthographic();
}

void Camera::setOrthoLeft(double left)
{
    myOrthoLeft = left;

    myProjectionDirty = true;
}

void Camera::setOrthoRight(double right)
{
    myOrthoRight = right;

    myProjectionDirty = true;
}

void Camera::setOrthoBottom(double bottom)
{
    myOrthoBottom = bottom;

    myProjectionDirty = true;
}

void Camera::setOrthoTop(double top)
{
    myOrthoTop = top;

    myProjectionDirty = true;
}

void Camera::setPerspectiveFovY(double fovY)
{
    if(fovY == myPerspectiveFovY)
		return;

    myPerspectiveFovY = fovY;

	myProjectionDirty = true;
}

void Camera::setPerspectiveAspectRatio(double aspectRatio)
{
    if(aspectRatio == myPerspectiveAspectRatio)
		return;

    myPerspectiveAspectRatio = aspectRatio;

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
    double distance = 1.5 * radius / qTan(myPerspectiveFovY * M_PI / 180.0);

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

void Camera::computeOrthographic()
{
    if(!orthographic())
        return;

    myOrthographic = false;
    myProjectionDirty = true;

    // compute the orthographic projection from the perspective one
    QMatrix4x4 perspectiveProj = Camera::projection();
    QMatrix4x4 perspectiveProjInv = perspectiveProj.inverted();

    myOrthographic = true;
    myProjectionDirty = true;

    QVector4D projectedTarget = perspectiveProj.map(view().map(QVector4D(target(), 1.0)));
    projectedTarget /= projectedTarget.w();

    QVector4D trCorner = perspectiveProjInv.map(QVector4D(1.0, 1.0, projectedTarget.z(), 1.0));
    trCorner /= trCorner.w();

    setOrthoLeft    (-trCorner.x());
    setOrthoRight   ( trCorner.x());
    setOrthoBottom  (-trCorner.y());
    setOrthoTop     ( trCorner.y());

    myProjectionDirty = true;
}

void Camera::computeModel()
{
    myView.setToIdentity();
    myView.lookAt(eye(), myTarget, up());
    myModel = myView.inverted();

    myViewDirty = false;
}

}

}
