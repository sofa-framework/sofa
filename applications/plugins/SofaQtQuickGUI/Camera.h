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
    Q_PROPERTY(bool orthographic READ orthographic WRITE setOrthographic NOTIFY orthographicChanged)

public:
    bool orthographic() const {return myOrthographic;}
    void setOrthographic(bool orthographic);

signals:
    void orthographicChanged();

public:
	const QMatrix4x4& projection() const;
	const QMatrix4x4& view() const;
	const QMatrix4x4& model() const;

    Q_INVOKABLE QVector3D eye() const				{return model().column(3).toVector3D();}
    Q_INVOKABLE QVector3D target() const			{return myTarget;}

    Q_INVOKABLE QVector3D direction() const			{return -model().column(2).toVector3D().normalized();}
    Q_INVOKABLE QVector3D up() const				{return  model().column(1).toVector3D().normalized();}
    Q_INVOKABLE QVector3D right() const				{return  model().column(0).toVector3D().normalized();}

    Q_INVOKABLE void lookAt(const QVector3D& eye, const QVector3D& target, const QVector3D& up);

    Q_INVOKABLE double computeDepth(const QVector3D& point);

public slots:
    void viewFromFront();
    void viewFromBack();
    void viewFromLeft();
    void viewFromRight();
    void viewFromTop();
    void viewFromBottom();
    void viewIsometric();

    void move(double x, double y, double z);
    void turn(double angleAroundX, double angleAroundY, double angleAroundZ);
    void zoom(double factor);

public:
    void setOrthoLeft(double left);
    void setOrthoRight(double right);
    void setOrthoBottom(double bottom);
    void setOrthoTop(double top);

    void setPerspectiveFovY(double fovY);
    void setPerspectiveAspectRatio(double aspectRatio);
	void setZNear(double zNear);
	void setZFar(double zFar);

	void fit(const QVector3D& min, const QVector3D& max);

private:
    void computeOrthographic();

private:
    bool                myOrthographic;
    double              myOrthoLeft;
    double              myOrthoRight;
    double              myOrthoBottom;
    double              myOrthoTop;
    double				myPerspectiveFovY;
    double				myPerspectiveAspectRatio;
	double				myZNear;
	double				myZFar;

	mutable QMatrix4x4	myProjection;
    mutable QMatrix4x4	myView;
    QMatrix4x4          myModel;

    QVector3D           myTarget;

	mutable bool		myProjectionDirty;
    mutable bool		myViewDirty;
};

}

}

#endif // CAMERA_H
