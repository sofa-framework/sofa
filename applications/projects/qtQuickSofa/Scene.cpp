#include "Scene.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/simulation/common/xml/initXml.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/component/init.h>
#include <QTimer>
#include <QString>
#include <QUrl>
#include <QThread>
#include <QDebug>

Scene::Scene(QObject *parent) :
    QObject(parent),
	myStatus(Status::Null),
	mySource(),
	myDt(0.04),
	myPlay(false),
	myAsynchronous(true),
	mySofaSimulation(0),
	myStepTimer(new QTimer(this))
{
	sofa::core::ExecParams::defaultInstance()->setAspectID(0);
	boost::shared_ptr<sofa::core::ObjectFactory::ClassEntry> classVisualModel;
	sofa::core::ObjectFactory::AddAlias("VisualModel", "OglModel", true, &classVisualModel);

	myStepTimer->setInterval(0);
	mySofaSimulation = sofa::simulation::graph::getSimulation();

	sofa::component::init();
	sofa::simulation::xml::initXml();

	connect(this, &Scene::sourceChanged, this, &Scene::open);
	connect(this, &Scene::playChanged, myStepTimer, [&](bool newPlay) {newPlay ? myStepTimer->start() : myStepTimer->stop();});
	connect(this, &Scene::statusChanged, this, [&](Scene::Status newStatus) {if(Scene::Status::Ready == newStatus) loaded();});

    connect(myStepTimer, &QTimer::timeout, this, &Scene::step);

	if(!mySource.isEmpty())
		open();
}

Scene::~Scene()
{
	if(mySofaSimulation == sofa::simulation::getSimulation())
		sofa::simulation::setSimulation(0);
}

static bool LoaderProcess(sofa::simulation::Simulation* sofaSimulation, const QString& scenePath)
{
	if(!sofaSimulation || scenePath.isEmpty())
		return false;

	sofaSimulation->unload(sofaSimulation->GetRoot());
	if(sofaSimulation->load(scenePath.toLatin1().constData()))
	{
		sofaSimulation->init(sofaSimulation->GetRoot().get());

		if(sofaSimulation->GetRoot())
			return true;
	}

	return false;
}

class LoaderThread : public QThread
{
public:
	LoaderThread(sofa::simulation::Simulation* sofaSimulation, const QString& scenePath) :
		mySofaSimulation(sofaSimulation),
		myScenepath(scenePath),
		myIsLoaded(false)
	{

	}

	void run()
	{
		myIsLoaded = LoaderProcess(mySofaSimulation, myScenepath);
	}

	bool isLoaded() const			{return myIsLoaded;}

private:
	sofa::simulation::Simulation*	mySofaSimulation;
	QString							myScenepath;
	bool							myIsLoaded;
};

void Scene::open()
{
	if(Status::Loading == myStatus) // return now if a scene is already loading
		return;

	setStatus(Status::Loading);

	QString finalFilename = mySource.toLocalFile();
	if(finalFilename.isEmpty())
	{
		setStatus(Status::Error);
		return;
	}

	std::string filepath = finalFilename.toLatin1().constData();
	if(sofa::helper::system::DataRepository.findFile(filepath))
		finalFilename = filepath.c_str();

	if(finalFilename.isEmpty())
	{
		setStatus(Status::Error);
		return;
	}

	setPlay(false);

	if(myAsynchronous)
	{
		LoaderThread* loaderThread = new LoaderThread(mySofaSimulation, finalFilename);
		connect(loaderThread, &QThread::finished, this, [this, loaderThread]() {setStatus(loaderThread && loaderThread->isLoaded() ? Status::Ready : Status::Error);});
		connect(loaderThread, &QThread::finished, loaderThread, &QObject::deleteLater);
		loaderThread->start();
	}
	else
	{
		setStatus(LoaderProcess(mySofaSimulation, finalFilename) ? Status::Ready : Status::Error);
	}
}

void Scene::setStatus(Status newStatus)
{
	if(newStatus == myStatus)
		return;

	myStatus = newStatus;

	statusChanged(newStatus);
}

void Scene::setPlay(bool newPlay)
{
	if(newPlay == myPlay)
		return;

	myPlay = newPlay;

	playChanged(newPlay);
}

void Scene::reload()
{
	open();
}

void Scene::step()
{
	if(!mySofaSimulation->GetRoot())
		return;

	emit stepBegin();
    mySofaSimulation->animate(mySofaSimulation->GetRoot().get(), myDt);
    emit stepEnd();
}

void Scene::reset()
{
	if(!mySofaSimulation->GetRoot())
		return;

    mySofaSimulation->reset(mySofaSimulation->GetRoot().get());
}
