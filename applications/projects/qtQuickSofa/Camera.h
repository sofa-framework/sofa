#ifndef CAMERA_H
#define CAMERA_H

#include <QObject>
#include <QMatrix4x4>

class Camera : public QObject
{
	Q_OBJECT

public:
	Camera(QObject* parent = 0);

public:
	const QMatrix4x4& projection() const;
	const QMatrix4x4& modelView() const;

	const QVector3D& eye() const			{return  myModelView.column(3).toVector3D();}
	const QVector3D& direction() const		{return -myModelView.column(2).toVector3D();}
	const QVector3D& up() const				{return  myModelView.column(1).toVector3D();}
	const QVector3D& right() const			{return  myModelView.column(0).toVector3D();}

public:
	void setFovY(float fovY);
	void setAspectRatio(float aspectRatio);
	void setZNear(float zNear);
	void setZFar(float zFar);

	void fit(const QVector3D& min, const QVector3D& max);

private:
	float				myFovY;
	float				myAspectRatio;
	float				myZNear;
	float				myZFar;

	mutable QMatrix4x4	myProjection;
	mutable QMatrix4x4	myModelView;

	mutable bool		myProjectionDirty;
	mutable bool		myModelViewDirty;
};

#endif // CAMERA_H
