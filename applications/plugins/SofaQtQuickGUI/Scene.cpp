#include "Scene.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/simulation/common/xml/initXml.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <SofaComponentMain/init.h>

#include <qqml.h>
#include <QVector3D>
#include <QTimer>
#include <QString>
#include <QUrl>
#include <QThread>
#include <QDebug>

Scene::Scene(QObject *parent) : QObject(parent),
	myStatus(Status::Null),
	mySource(),
	mySourceQML(),
	myIsInit(false),
	myVisualDirty(true),
	myDt(0.04),
	myPlay(false),
	myAsynchronous(true),
	mySofaSimulation(0),
	myStepTimer(new QTimer(this))
{
	// sofa init
	sofa::core::ExecParams::defaultInstance()->setAspectID(0);
	boost::shared_ptr<sofa::core::ObjectFactory::ClassEntry> classVisualModel;
	sofa::core::ObjectFactory::AddAlias("VisualModel", "OglModel", true, &classVisualModel);

	myStepTimer->setInterval(0);
	mySofaSimulation = sofa::simulation::graph::getSimulation();

	sofa::component::init();
	sofa::simulation::xml::initXml();

	// plugins
	QVector<QString> plugins;
	plugins.append("SofaPython");

    for(const QString& plugin : plugins) {
        std::string s = plugin.toStdString();
        sofa::helper::system::PluginManager::getInstance().loadPlugin(s);
    }

	sofa::helper::system::PluginManager::getInstance().init();

	// connections
	connect(this, &Scene::sourceChanged, this, &Scene::open);
	connect(this, &Scene::playChanged, myStepTimer, [&](bool newPlay) {newPlay ? myStepTimer->start() : myStepTimer->stop();});
	connect(this, &Scene::statusChanged, this, [&](Scene::Status newStatus) {if(Scene::Status::Ready == newStatus) loaded();});

    connect(myStepTimer, &QTimer::timeout, this, &Scene::step);
}

Scene::~Scene()
{
	if(mySofaSimulation == sofa::simulation::getSimulation())
		sofa::simulation::setSimulation(0);
}

void Scene::classBegin()
{

}

void Scene::componentComplete()
{
	if(!mySource.isEmpty())
		open();
}

static bool LoaderProcess(sofa::simulation::Simulation* sofaSimulation, const QString& scenePath)
{
	if(!sofaSimulation || scenePath.isEmpty())
		return false;

	sofaSimulation->unload(sofaSimulation->GetRoot());

	sofa::core::visual::VisualParams* vparams = sofa::core::visual::VisualParams::defaultInstance();
	if(vparams)
		vparams->displayFlags().setShowVisualModels(true);

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
	setSourceQML(QUrl());

	if(Status::Loading == myStatus) // return now if a scene is already loading
		return;

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

	setStatus(Status::Loading);

	setPlay(false);
	myIsInit = false;

	std::string qmlFilepath = (finalFilename + ".qml").toLatin1().constData();
	if(!sofa::helper::system::DataRepository.findFile(qmlFilepath))
		qmlFilepath.clear();

	if(myAsynchronous)
	{
		LoaderThread* loaderThread = new LoaderThread(mySofaSimulation, finalFilename);
		connect(loaderThread, &QThread::finished, this, [this, loaderThread]() {setStatus(loaderThread && loaderThread->isLoaded() ? Status::Ready : Status::Error);});
		
		if(!qmlFilepath.empty())
			connect(loaderThread, &QThread::finished, this, [=]() {setSourceQML(QUrl::fromLocalFile(qmlFilepath.c_str()));});

		connect(loaderThread, &QThread::finished, loaderThread, &QObject::deleteLater);
		loaderThread->start();
	}
	else
	{
		setStatus(LoaderProcess(mySofaSimulation, finalFilename) ? Status::Ready : Status::Error);
		
		if(!qmlFilepath.empty())
			setSourceQML(QUrl::fromLocalFile(qmlFilepath.c_str()));
	}
}

void Scene::setStatus(Status newStatus)
{
	if(newStatus == myStatus)
		return;

	myStatus = newStatus;

	statusChanged(newStatus);
}

void Scene::setSource(const QUrl& newSource)
{
	if(newSource == mySource || Status::Loading == myStatus)
		return;

	setStatus(Status::Null);

	mySource = newSource;

	sourceChanged(newSource);
}

void Scene::setSourceQML(const QUrl& newSourceQML)
{
	if(newSourceQML == mySourceQML)
		return;

	mySourceQML = newSourceQML;

	sourceQMLChanged(newSourceQML);
}

void Scene::setDt(double newDt)
{
	if(newDt == myDt)
		return;

	myDt = newDt;

	dtChanged(newDt);
}

void Scene::setPlay(bool newPlay)
{
	if(newPlay == myPlay)
		return;

	myPlay = newPlay;

	playChanged(newPlay);
}

double Scene::radius()
{
	QVector3D min, max;
	computeBoundingBox(min, max);
	QVector3D diag = (max - min);

	return diag.length();
}

void Scene::computeBoundingBox(QVector3D& min, QVector3D& max)
{
	SReal pmin[3], pmax[3];
	mySofaSimulation->computeTotalBBox(mySofaSimulation->GetRoot().get(), pmin, pmax );

	min = QVector3D(pmin[0], pmin[1], pmin[2]);
	max = QVector3D(pmax[0], pmax[1], pmax[2]);
}

QString Scene::dumpGraph()
{
	QString dump;

	if(mySofaSimulation->GetRoot())
	{
		std::streambuf* backup(std::cout.rdbuf());

		std::ostringstream stream;
		std::cout.rdbuf(stream.rdbuf());
		mySofaSimulation->print(mySofaSimulation->GetRoot().get());
		std::cout.rdbuf(backup);

		dump += QString::fromStdString(stream.str());
	}

	return dump;
}

void Scene::init()
{
	if(!mySofaSimulation->GetRoot())
		return;

	mySofaSimulation->initTextures(mySofaSimulation->GetRoot().get());
	setDt(mySofaSimulation->GetRoot()->getDt());

	myIsInit = true;
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
	myVisualDirty = true;
    emit stepEnd();
}

void Scene::reset()
{
	if(!mySofaSimulation->GetRoot())
		return;

    mySofaSimulation->reset(mySofaSimulation->GetRoot().get());
}

void Scene::draw()
{
	if(!mySofaSimulation->GetRoot())
		return;

	if(myVisualDirty)
	{
		mySofaSimulation->updateVisual(mySofaSimulation->GetRoot().get());
		myVisualDirty = false;
	}

	mySofaSimulation->draw(sofa::core::visual::VisualParams::defaultInstance(), mySofaSimulation->GetRoot().get());
}

void Scene::onKeyPressed(char key)
{
	if(!mySofaSimulation->GetRoot())
		return;

	sofa::core::objectmodel::KeypressedEvent keyEvent(key);
	sofaSimulation()->GetRoot()->propagateEvent(sofa::core::ExecParams::defaultInstance(), &keyEvent);
}

void Scene::onKeyReleased(char key)
{
	if(!mySofaSimulation->GetRoot())
		return;

	sofa::core::objectmodel::KeyreleasedEvent keyEvent(key);
	sofaSimulation()->GetRoot()->propagateEvent(sofa::core::ExecParams::defaultInstance(), &keyEvent);
}
