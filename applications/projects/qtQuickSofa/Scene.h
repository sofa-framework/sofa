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
	Q_PROPERTY(QUrl source MEMBER mySource NOTIFY sourceChanged);
	Q_PROPERTY(double dt MEMBER myDt NOTIFY dtChanged);
	Q_PROPERTY(bool play MEMBER myPlay NOTIFY playChanged);

public:
	const QUrl& source() const	{return mySource;}
	double dt() const			{return myDt;}
	bool playing() const		{return myPlay;}

signals:
	void sourceChanged(const QUrl& newSource);
	void dtChanged(double newDt);
	void playChanged(bool newPlay);

public slots:
	/// re-open the current scene
	bool reload();
	/// apply one simulation time step, the simulation must be paused (play = false)
	void step();
	/// restart at the beginning, without reloading the file
	void reset();

signals:
	void opened();
	void stepBegin();
    void stepEnd();

protected slots:
    /// open a scene according to the source
	bool open();

public:
	sofa::simulation::Simulation* sofaSimulation() const {return mySofaSimulation;}

private:
	QUrl							mySource;
	double							myDt;
	bool							myPlay;

	sofa::simulation::Simulation*	mySofaSimulation;
	QTimer*							myStepTimer;
};

#endif // SCENE_H