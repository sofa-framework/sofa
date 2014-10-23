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
#include <QDebug>

Scene::Scene(QObject *parent) :
    QObject(parent),
	mySource(),
	myDt(0.04),
	myPlay(false),
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

    connect(myStepTimer, &QTimer::timeout, this, &Scene::step);
}

Scene::~Scene()
{
	if(mySofaSimulation == sofa::simulation::getSimulation())
		sofa::simulation::setSimulation(0);
}

bool Scene::open()
{
	QString finalFilename = mySource.toLocalFile();
	if(finalFilename.isEmpty())
		return false;

	std::string filepath = finalFilename.toLatin1().constData();
	if(sofa::helper::system::DataRepository.findFile(filepath))
		finalFilename = filepath.c_str();

	if(finalFilename.isEmpty())
		return false;

	mySofaSimulation->unload(mySofaSimulation->GetRoot());
	if(!mySofaSimulation->load(finalFilename.toLatin1().constData()))
		return false;

	mySofaSimulation->init(mySofaSimulation->GetRoot().get());

	emit opened();

	return true;
}

bool Scene::reload()
{
	return open();
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
