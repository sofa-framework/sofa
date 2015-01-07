#ifndef SCENE_H
#define SCENE_H

#include <QObject>
#include <QQmlParserStatus>
#include <QUrl>
#include <sofa/simulation/common/Simulation.h>

class QTimer;
class QVector3D;

class Scene : public QObject, public QQmlParserStatus
{
    Q_OBJECT
	Q_INTERFACES(QQmlParserStatus)

public:
    explicit Scene(QObject *parent = 0);
	~Scene();

	void classBegin();
	void componentComplete();

public:
	Q_PROPERTY(Status status READ status WRITE setStatus NOTIFY statusChanged);
	Q_PROPERTY(QUrl source READ source WRITE setSource NOTIFY sourceChanged);
	Q_PROPERTY(QUrl sourceQML READ sourceQML WRITE setSourceQML NOTIFY sourceQMLChanged);
	Q_PROPERTY(double dt READ dt WRITE setDt NOTIFY dtChanged);
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

signals:
	void loaded();
	void statusChanged(Status newStatus);
	void sourceChanged(const QUrl& newSource);
	void sourceQMLChanged(const QUrl& newSourceQML);
	void dtChanged(double newDt);
	void playChanged(bool newPlay);
	void asynchronousChanged(bool newAsynchronous);

public:
	Q_INVOKABLE double radius();
	Q_INVOKABLE void computeBoundingBox(QVector3D& min, QVector3D& max);
	Q_INVOKABLE QString dumpGraph();

public slots:
	void init();
	void reload();
	void step();
	void reset();
	void draw();

	void onKeyPressed(char key);
	void onKeyReleased(char key);

signals:
	void stepBegin();
    void stepEnd();

private slots:
	void open();

public:
	sofa::simulation::Simulation* sofaSimulation() const {return mySofaSimulation;}

private:
	Status							myStatus;
	QUrl							mySource;
	QUrl							mySourceQML;
	bool							myIsInit;
	bool							myVisualDirty;
	double							myDt;
	bool							myPlay;
	bool							myAsynchronous;

	sofa::simulation::Simulation*	mySofaSimulation;
	QTimer*							myStepTimer;
};

#endif // SCENE_H