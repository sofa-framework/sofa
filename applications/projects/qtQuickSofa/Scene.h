#ifndef SCENE_H
#define SCENE_H

#include <QObject>
#include <QUrl>
#include <sofa/simulation/common/Simulation.h>

class QTimer;

class Scene : public QObject
{
    Q_OBJECT

public:
    explicit Scene(QObject *parent = 0);
	~Scene();

public:
	Q_PROPERTY(Status status READ status WRITE setStatus NOTIFY statusChanged);
	Q_PROPERTY(QUrl source MEMBER mySource NOTIFY sourceChanged);
	Q_PROPERTY(double dt MEMBER myDt NOTIFY dtChanged);
	Q_PROPERTY(bool play READ playing WRITE setPlay NOTIFY playChanged)
	Q_PROPERTY(bool asynchronous MEMBER myAsynchronous NOTIFY asynchronousChanged)

	Q_ENUMS(Status)
	enum Status {
		Null,
		Ready,
		Loading,
		Error
	};

public:
	Status status()	const				{return myStatus;}
	void setStatus(Status newStatus);

	const QUrl& source() const			{return mySource;}
	double dt() const					{return myDt;}
	
	bool playing() const				{return myPlay;}
	void setPlay(bool newPlay);

	bool isReady() const				{return Status::Ready == myStatus;}

signals:
	void loaded();
	void statusChanged(Status newStatus);
	void sourceChanged(const QUrl& newSource);
	void dtChanged(double newDt);
	void playChanged(bool newPlay);
	void asynchronousChanged(bool newAsynchronous);

public slots:
	/// re-open the current scene
	void reload();
	/// apply one simulation time step, the simulation must be paused (play = false)
	void step();
	/// restart at the beginning, without reloading the file
	void reset();

signals:
	void stepBegin();
    void stepEnd();

private slots:
    /// open a scene according to the source
	void open();

public:
	sofa::simulation::Simulation* sofaSimulation() const {return mySofaSimulation;}

private:
	Status							myStatus;
	QUrl							mySource;
	double							myDt;
	bool							myPlay;
	bool							myAsynchronous;

	sofa::simulation::Simulation*	mySofaSimulation;
	QTimer*							myStepTimer;
};

#endif // SCENE_H