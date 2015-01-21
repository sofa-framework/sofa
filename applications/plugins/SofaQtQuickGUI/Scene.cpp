#include <GL/glew.h>
#include "Scene.h"

#include <sofa/component/init.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/simulation/common/xml/initXml.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/visual/DrawToolGL.h>
#include <sofa/helper/system/glut.h>

#include <qqml.h>
#include <QVector3D>
#include <QTimer>
#include <QString>
#include <QUrl>
#include <QThread>
#include <QDebug>

namespace sofa
{

namespace qtquick
{

Scene::Scene(QObject *parent) : QObject(parent), QQmlParserStatus(),
	myStatus(Status::Null),
	mySource(),
	mySourceQML(),
	myIsInit(false),
    myVisualDirty(false),
	myDt(0.04),
	myPlay(false),
	myAsynchronous(true),
	mySofaSimulation(0),
    myStepTimer(new QTimer(this)),
    myComponentCount(0)
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
/*
int	Scene::rowCount(const QModelIndex & parent) const
{
    return myComponentCount;
}

QVariant Scene::data(const QModelIndex& index, int role) const
{
    qDebug() << "data" << index.row();

    if(!index.isValid())
        return QVariant("Error");

    using sofa::core::objectmodel::BaseNode;
    BaseNode* node = static_cast<BaseNode*>(index.internalPointer());
    if(0 == node)
        return QVariant("Error");

    return QVariant::fromValue(QString(node->name.getValue().c_str()));
}

QModelIndex Scene::index(int row, int column, const QModelIndex& parent) const
{
    using sofa::core::objectmodel::BaseNode;
    std::stack<BaseNode*> nodeStack;

    int id = 0;

    nodeStack.push(sofaSimulation()->GetRoot().get());
    while(!nodeStack.empty())
    {
        BaseNode* node = nodeStack.top();
        nodeStack.pop();

        if(0 == node)
            continue;

        qDebug() << rowCount() << count;
        if(rowCount() == count)
            return createIndex(row, column, (void*) node);

        ++id;

        for(int i = 0; i < node->getChildren().size(); ++i)
        {
            int j = node->getChildren().size() - 1 - i;
            nodeStack.push(node->getChildren()[j]);
        }
    }

    return QModelIndex();
}

void Scene::update()
{
    myComponentCount = 0;

    using sofa::core::objectmodel::BaseNode;
    std::stack<BaseNode*> nodeStack;

    nodeStack.push(sofaSimulation()->GetRoot().get());
    while(!nodeStack.empty())
    {
        BaseNode* node = nodeStack.top();
        nodeStack.pop();

        if(0 == node)
            continue;

        ++myComponentCount;

        for(int i = 0; i < node->getChildren().size(); ++i)
        {
            int j = node->getChildren().size() - 1 - i;
            nodeStack.push(node->getChildren()[j]);
        }
    }

    beginInsertRows(QModelIndex(), 0, myComponentCount - 1);
//    for(int i = 0; i < mySofaSimulation->GetRoot()->getChildren().size(); ++i)
//    {
//        sofa::core::objectmodel::BaseNode* baseNode = mySofaSimulation->GetRoot()->getChildren()[i];
//        qDebug() << "base" << baseNode;
//    }
    endInsertRows();
}
*/
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

    mySofaSimulation->unload(mySofaSimulation->GetRoot());

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

void Scene::setVisualDirty(bool newVisualDirty)
{
    if(newVisualDirty == myVisualDirty)
        return;

    myVisualDirty = newVisualDirty;

    visualDirtyChanged(newVisualDirty);
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

    GLenum err = glewInit();
    if(0 != err)
        qDebug() << "GLEW Initialization failed with error code:" << err;

    // prepare the sofa visual params
    sofa::core::visual::VisualParams* visualParams = sofa::core::visual::VisualParams::defaultInstance();
    if(visualParams)
    {
        if(!visualParams->drawTool())
        {
            visualParams->drawTool() = new sofa::core::visual::DrawToolGL();
            visualParams->setSupported(sofa::core::visual::API_OpenGL);
        }
    }

#ifdef __linux__
    static bool glutInited = false;
    if(!glutInited)
    {
        int argc = 0;
        glutInit(&argc, NULL);
        glutInited = true;
    }
#endif

	mySofaSimulation->initTextures(mySofaSimulation->GetRoot().get());
	setDt(mySofaSimulation->GetRoot()->getDt());

    myIsInit = true;

    //update();
}

void Scene::reload()
{
    // TODO: ! NEED CURRENT OPENGL CONTEXT while releasing the old sofa scene
    //qDebug() << "reload - thread" << QThread::currentThread() << QOpenGLContext::currentContext() << (void*) &glLightfv;

    open();
}

void Scene::step()
{
	if(!mySofaSimulation->GetRoot())
		return;

	emit stepBegin();
    mySofaSimulation->animate(mySofaSimulation->GetRoot().get(), myDt);
    setVisualDirty(true);
    emit stepEnd();
}

void Scene::reset()
{
    if(!mySofaSimulation->GetRoot())
        return;

    // TODO: ! NEED CURRENT OPENGL CONTEXT
    mySofaSimulation->reset(mySofaSimulation->GetRoot().get());
    setVisualDirty(true);
    emit reseted();
}

void Scene::draw()
{
	if(!mySofaSimulation->GetRoot())
		return;

    // prepare the sofa visual params
    sofa::core::visual::VisualParams* visualParams = sofa::core::visual::VisualParams::defaultInstance();
    if(visualParams)
    {
        GLint _viewport[4];
        GLdouble _mvmatrix[16], _projmatrix[16];

        glGetIntegerv(GL_VIEWPORT, _viewport);
        glGetDoublev(GL_MODELVIEW_MATRIX, _mvmatrix);
        glGetDoublev(GL_PROJECTION_MATRIX, _projmatrix);

        visualParams->viewport() = sofa::helper::fixed_array<int, 4>(_viewport[0], _viewport[1], _viewport[2], _viewport[3]);
        visualParams->sceneBBox() = mySofaSimulation->GetRoot()->f_bbox.getValue();
        visualParams->setProjectionMatrix(_projmatrix);
        visualParams->setModelViewMatrix(_mvmatrix);
    }

    //qDebug() << "draw - thread" << QThread::currentThread() << QOpenGLContext::currentContext();

    if(visualDirty())
    {
        mySofaSimulation->updateVisual(mySofaSimulation->GetRoot().get());
        setVisualDirty(false);
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

}

}
