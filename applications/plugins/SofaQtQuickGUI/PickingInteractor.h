#ifndef PICKINGINTERACTOR_H
#define PICKINGINTERACTOR_H

#include <QObject>
#include <QQmlParserStatus>
#include <QVector3D>

class Scene;

namespace sofa {
namespace core {
namespace behavior {
	class BaseMechanicalState;
	class BaseInteractionForceField;
}
namespace objectmodel {
	class BaseNode;
}
}
}

class PickingInteractor : public QObject, public QQmlParserStatus
{
	Q_OBJECT
	Q_INTERFACES(QQmlParserStatus)

	typedef sofa::core::behavior::BaseMechanicalState			BaseMechanicalState;
	typedef sofa::core::behavior::BaseInteractionForceField		BaseInteractionForceField;
	typedef sofa::core::objectmodel::BaseNode					BaseNode;

	struct PickedPoint {
		BaseMechanicalState*	mechanicalState;
		int						index;
		QVector3D				position;
	};

public:
	PickingInteractor(QObject *parent = 0);
	~PickingInteractor();

	void classBegin();
	void componentComplete();
	
public:
	Q_PROPERTY(Scene* scene READ scene WRITE setScene NOTIFY sceneChanged);
	Q_PROPERTY(double stiffness MEMBER myStiffness NOTIFY stiffnessChanged);
	Q_PROPERTY(bool picking READ picking NOTIFY pickingChanged);
	Q_PROPERTY(QVector3D position READ position WRITE setPosition NOTIFY positionChanged);

public:
	Scene* scene() const		{return myScene;}
	void setScene(Scene* newScene);

	double stiffness() const	{return myStiffness;}

	bool picking() const		{return 0 != myPickedPoint;}

	QVector3D position() const;
	void setPosition(const QVector3D& position);
	
signals:
	void sceneChanged(Scene* newScene);
	void stiffnessChanged(double newStiffness);
	void pickingChanged(bool newPicking);
	void positionChanged(const QVector3D& newPosition);
	
public slots:
	void release();

private slots:
	void handleSceneChanged(Scene* scene);
	void computePickProperties();

public:
	Q_INVOKABLE bool pick(const QVector3D& origin, const QVector3D& ray);

	Q_INVOKABLE QVector3D pickedPointPosition() const;

private:
	Scene*									myScene;

	BaseNode*								myNode;
	BaseMechanicalState*					myMechanicalState;
	BaseInteractionForceField*				myForcefield;

	double									myDistanceToRay;
	double									myDistanceToRayGrowth;

	PickedPoint*							myPickedPoint;
	double									myStiffness;
	
};

#endif // PICKINGINTERACTOR_H