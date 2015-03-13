#ifndef SCENE_H
#define SCENE_H

#include "SofaQtQuickGUI.h"

#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/MutationListener.h>

#include <QObject>
#include <QVariant>
#include <QSet>
#include <QVector3D>
#include <QUrl>

class QTimer;

namespace sofa
{

namespace qtquick
{

class Scene;

class SceneComponent : public QObject
{
    Q_OBJECT

public:
    SceneComponent(const Scene* scene, sofa::core::objectmodel::Base* base);

    sofa::core::objectmodel::Base* base();
    const sofa::core::objectmodel::Base* base() const;

private:
    const Scene*                            myScene;
    mutable sofa::core::objectmodel::Base*  myBase;

};

class SceneData : public QObject
{
    Q_OBJECT

public:
    SceneData(const SceneComponent* sceneComponent, sofa::core::objectmodel::BaseData* data);

    Q_INVOKABLE QVariantMap object() const;
    Q_INVOKABLE void setValue(const QVariant& value);

    sofa::core::objectmodel::BaseData* data();
    const sofa::core::objectmodel::BaseData* data() const;

private:
    const SceneComponent*                       mySceneComponent;
    mutable sofa::core::objectmodel::BaseData*  myData;

};

class Scene : public QObject, private sofa::simulation::MutationListener
{
    Q_OBJECT

    friend class SceneComponent;

public:
    explicit Scene(QObject *parent = 0);
	~Scene();

public:
	Q_PROPERTY(Status status READ status WRITE setStatus NOTIFY statusChanged);
	Q_PROPERTY(QUrl source READ source WRITE setSource NOTIFY sourceChanged);
	Q_PROPERTY(QUrl sourceQML READ sourceQML WRITE setSourceQML NOTIFY sourceQMLChanged);
	Q_PROPERTY(double dt READ dt WRITE setDt NOTIFY dtChanged);
	Q_PROPERTY(bool play READ playing WRITE setPlay NOTIFY playChanged)
	Q_PROPERTY(bool asynchronous MEMBER myAsynchronous NOTIFY asynchronousChanged)
    Q_PROPERTY(bool visualDirty READ visualDirty NOTIFY visualDirtyChanged)

	Q_ENUMS(Status)
	enum Status {
		Null,
		Ready,
		Loading,
		Error
	};

public:
	Status status()	const							{return myStatus;}
	void setStatus(Status newStatus);

	const QUrl& source() const						{return mySource;}
	void setSource(const QUrl& newSource);

	const QUrl& sourceQML() const					{return mySourceQML;}
	void setSourceQML(const QUrl& newSourceQML);

	double dt() const								{return myDt;}
	void setDt(double newDt);
	
	bool playing() const							{return myPlay;}
	void setPlay(bool newPlay);

	bool isReady() const							{return Status::Ready == myStatus;}
	bool isInit() const								{return myIsInit;}

    bool visualDirty() const						{return myVisualDirty;}
    void setVisualDirty(bool newVisualDirty);

signals:
	void loaded();
    void aboutToUnload();
	void statusChanged(Status newStatus);
	void sourceChanged(const QUrl& newSource);
	void sourceQMLChanged(const QUrl& newSourceQML);
	void dtChanged(double newDt);
	void playChanged(bool newPlay);
	void asynchronousChanged(bool newAsynchronous);
    void visualDirtyChanged(bool newVisualDirty);

public:
    Q_INVOKABLE double radius() const;
    Q_INVOKABLE void computeBoundingBox(QVector3D& min, QVector3D& max) const;
    Q_INVOKABLE QString dumpGraph() const;
    Q_INVOKABLE void reinitComponent(const QString& path);

public:
    static QVariantMap dataObject(const sofa::core::objectmodel::BaseData* data);
    static QVariant dataValue(const sofa::core::objectmodel::BaseData* data);
    static void setDataValue(sofa::core::objectmodel::BaseData* data, const QVariant& value);

    QVariant dataValue(const QString& path) const;
    void setDataValue(const QString& path, const QVariant& value);

protected:
    Q_INVOKABLE QVariant onDataValue(const QString& path) const;
    Q_INVOKABLE void onSetDataValue(const QString& path, const QVariant& value);

public slots:
    void init();        // need an opengl context made current
	void reload();
	void step();
	void reset();
	void draw();

	void onKeyPressed(char key);
	void onKeyReleased(char key);

signals:
	void stepBegin();
    void stepEnd();
    void reseted();

private slots:
	void open();    

public:
	sofa::simulation::Simulation* sofaSimulation() const {return mySofaSimulation;}

protected:
    void addChild(sofa::simulation::Node* parent, sofa::simulation::Node* child);
    void removeChild(sofa::simulation::Node* parent, sofa::simulation::Node* child);
    void addObject(sofa::simulation::Node* parent, sofa::core::objectmodel::BaseObject* object);
    void removeObject(sofa::simulation::Node* parent, sofa::core::objectmodel::BaseObject* object);

private:
    Status                                  myStatus;
    QUrl                                    mySource;
    QUrl                                    mySourceQML;
    bool                                    myIsInit;
    bool                                    myVisualDirty;
    double                                  myDt;
    bool                                    myPlay;
    bool                                    myAsynchronous;

    sofa::simulation::Simulation*           mySofaSimulation;
    QTimer*                                 myStepTimer;
    QSet<sofa::core::objectmodel::Base*>    myBases;
};

}

}

#endif // SCENE_H
