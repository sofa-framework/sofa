#include <GL/glew.h>
#include "Scene.h"

#include <sofa/component/init.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/simulation/common/xml/initXml.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/visual/DrawToolGL.h>
#include <sofa/helper/system/glut.h>
#include <sofa/defaulttype/DataTypeInfo.h>

#include <qqml.h>
#include <QVector3D>
#include <QStack>
#include <QTimer>
#include <QString>
#include <QUrl>
#include <QThread>
#include <QSequentialIterable>
#include <QJSValue>
#include <QDebug>

namespace sofa
{

namespace qtquick
{

using namespace sofa::defaulttype;
using namespace sofa::core::objectmodel;
using namespace sofa::simulation;

Scene::Scene(QObject *parent) : QObject(parent),
	myStatus(Status::Null),
	mySource(),
	mySourceQML(),
	myIsInit(false),
    myVisualDirty(false),
	myDt(0.04),
	myPlay(false),
	myAsynchronous(true),
	mySofaSimulation(0),
    myStepTimer(new QTimer(this))
{
	// sofa init
    sofa::helper::system::DataRepository.addFirstPath("../../share/");
    sofa::helper::system::DataRepository.addFirstPath("../../examples/");
    sofa::helper::system::PluginRepository.addFirstPath("../bin/");

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

    for(const QString& plugin : plugins)
    {
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

    aboutToUnload();

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
        connect(loaderThread, &QThread::finished, this, [this, loaderThread]() {setStatus(loaderThread->isLoaded() ? Status::Ready : Status::Error);});
		
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

QVariant Scene::getData(const QString& path) const
{
    return onGetData(path);
}

void Scene::setData(const QString& path, const QVariant& value)
{
    onSetData(path, value);
}

QVariant Scene::onGetData(const QString& path) const
{
    QVariant value;

    BaseData* data = 0;
    mySofaSimulation->GetRoot()->findDataLinkDest(data, path.toStdString(), 0);
    if(!data)
    {
        qWarning() << "DataPath unknown:" << path;
        return value;
    }

    const AbstractTypeInfo* typeinfo = data->getValueTypeInfo();
    const void* valueVoidPtr = data->getValueVoidPtr();

    if(!typeinfo->Container())
    {
        if(typeinfo->Text())
            value = QString::fromStdString(typeinfo->getTextValue(valueVoidPtr, 0));
        else if(typeinfo->Scalar())
            value = typeinfo->getScalarValue(valueVoidPtr, 0);
        else if(typeinfo->Integer())
            value = typeinfo->getIntegerValue(valueVoidPtr, 0);
    }
    else
    {
        int nbCols = typeinfo->size();
        int nbRows = typeinfo->size(data->getValueVoidPtr()) / nbCols;

        if(typeinfo->Text())
        {
            QVariantList values;
            values.reserve(nbRows);

            QVariantList subValues;
            subValues.reserve(nbCols);

            for(int j = 0; j < nbRows; j++)
            {
                subValues.clear();
                for(int i = 0; i < nbCols; i++)
                    subValues.append(QVariant::fromValue(QString::fromStdString((typeinfo->getTextValue(valueVoidPtr, j * nbCols + i)))));

                values.append(QVariant::fromValue(subValues));
            }

            value = values;
        }
        else if(typeinfo->Scalar())
        {
            QVariantList values;
            values.reserve(nbRows);

            QVariantList subValues;
            subValues.reserve(nbCols);

            for(int j = 0; j < nbRows; j++)
            {
                subValues.clear();
                for(int i = 0; i < nbCols; i++)
                    subValues.append(QVariant::fromValue(typeinfo->getScalarValue(valueVoidPtr, j * nbCols + i)));

                values.append(QVariant::fromValue(subValues));
            }

            value = values;
        }
        else if(typeinfo->Integer())
        {
            QVariantList values;
            values.reserve(nbRows);

            QVariantList subValues;
            subValues.reserve(nbCols);

            for(int j = 0; j < nbRows; j++)
            {
                subValues.clear();
                for(int i = 0; i < nbCols; i++)
                    subValues.append(QVariant::fromValue(typeinfo->getIntegerValue(valueVoidPtr, j * nbCols + i)));

                values.push_back(QVariant::fromValue(subValues));
            }

            value = values;
        }
    }

    return value;
}

void Scene::onSetData(const QString& path, const QVariant& value)
{
    BaseData* data = 0;
    mySofaSimulation->GetRoot()->findDataLinkDest(data, path.toStdString(), 0);

    if(!data)
    {
        qWarning() << "DataPath unknown:" << path;
        return;
    }

    const AbstractTypeInfo* typeinfo = data->getValueTypeInfo();

    if(!value.isNull())
    {
        QVariant finalValue = value;
        if(finalValue.userType() == qMetaTypeId<QJSValue>())
            finalValue = finalValue.value<QJSValue>().toVariant();

        if(QVariant::List == finalValue.type())
        {
            QSequentialIterable valueIterable = finalValue.value<QSequentialIterable>();
            if(1 == valueIterable.size())
                finalValue = valueIterable.at(0);
        }

        if(QVariant::List == finalValue.type())
        {
            QSequentialIterable valueIterable = finalValue.value<QSequentialIterable>();

            int nbCols = typeinfo->size();
            int nbRows = typeinfo->size(data->getValueVoidPtr()) / nbCols;

            if(!typeinfo->Container())
            {
                qWarning("Trying to set a list of values on a non-container data");
                return;
            }

            if(valueIterable.size() != nbRows)
            {
                if(typeinfo->FixedSize())
                {
                    qWarning("The new data should have the same size");
                    return;
                }

                typeinfo->setSize(data, valueIterable.size());
            }

            if(typeinfo->Scalar())
            {
                QString dataString;
                for(int i = 0; i < valueIterable.size(); ++i)
                {
                    QVariant subFinalValue = valueIterable.at(i);
                    if(QVariant::List == subFinalValue.type())
                    {
                        QSequentialIterable subValueIterable = subFinalValue.value<QSequentialIterable>();
                        if(subValueIterable.size() != nbCols)
                        {
                            qWarning("The new sub data should have the same size");
                            return;
                        }

                        for(int j = 0; j < subValueIterable.size(); ++j)
                        {
                            dataString += QString::number(subValueIterable.at(j).toDouble());
                            if(subValueIterable.size() - 1 != j)
                                dataString += ' ';
                        }
                    }
                    else
                    {
                        dataString += QString::number(subFinalValue.toDouble());
                    }

                    if(valueIterable.size() - 1 != i)
                        dataString += ' ';
                }

                data->read(dataString.toStdString());
            }
            else if(typeinfo->Integer())
            {
                QString dataString;
                for(int i = 0; i < valueIterable.size(); ++i)
                {
                    QVariant subFinalValue = valueIterable.at(i);
                    if(QVariant::List == subFinalValue.type())
                    {
                        QSequentialIterable subValueIterable = subFinalValue.value<QSequentialIterable>();
                        if(subValueIterable.size() != nbCols)
                        {
                            qWarning("The new sub data should have the same size");
                            return;
                        }

                        for(int j = 0; j < subValueIterable.size(); ++j)
                        {
                            dataString += QString::number(subValueIterable.at(j).toLongLong());
                            if(subValueIterable.size() - 1 != j)
                                dataString += ' ';
                        }
                    }
                    else
                    {
                        dataString += QString::number(subFinalValue.toLongLong());
                    }

                    if(valueIterable.size() - 1 != i)
                        dataString += ' ';
                }

                data->read(dataString.toStdString());
            }
        }
        else if(QVariant::Map == finalValue.type())
        {
            qWarning("Map type are not supported");
        }
        else
        {
            if(typeinfo->Text())
                data->read(value.toString().toStdString());
            else if(typeinfo->Scalar())
                data->read(QString::number(value.toDouble()).toStdString());
            else if(typeinfo->Integer())
                data->read(QString::number(value.toLongLong()).toStdString());
        }
    }
}

void Scene::init()
{
	if(!mySofaSimulation->GetRoot())
		return;

    GLenum err = glewInit();
    if(0 != err)
        qWarning() << "GLEW Initialization failed with error code:" << err;

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

SceneListModel::SceneListModel(QObject* parent) : QAbstractListModel(parent), QQmlParserStatus(), MutationListener(),
    myItems(),
    myUpdatedCount(0),
    myScene(0)
{

}

SceneListModel::~SceneListModel()
{

}

void SceneListModel::classBegin()
{

}

void SceneListModel::componentComplete()
{
    if(!myScene)
        setScene(qobject_cast<Scene*>(parent()));
    else
        handleSceneChange(myScene);
}

void SceneListModel::update()
{
    int changeNum = myItems.size() - myUpdatedCount;
    if(changeNum > 0)
    {
        beginInsertRows(QModelIndex(), myUpdatedCount, myItems.size() - 1);
        endInsertRows();
    }
    else if(changeNum < 0)
    {
        beginRemoveRows(QModelIndex(), myItems.size(), myUpdatedCount - 1);
        endRemoveRows();
    }

    dataChanged(createIndex(0, 0), createIndex(myItems.size() - 1, 0));

    myUpdatedCount = myItems.size();
}

void SceneListModel::handleSceneChange(Scene* newScene)
{
    if(myScene)
    {
        clear();
        if(myScene->isReady())
            addChild(0, myScene->sofaSimulation()->GetRoot().get());

        connect(myScene, &Scene::loaded, [this]() {addChild(0, myScene->sofaSimulation()->GetRoot().get()); update();});
        connect(myScene, &Scene::aboutToUnload, this, &SceneListModel::clear);
    }
}

void SceneListModel::clear()
{
    myItems.clear();
    update();
}

SceneListModel::Item SceneListModel::buildNodeItem(Item* parent, BaseNode* node)
{
    SceneListModel::Item item;

    item.parent = parent;
    item.depth = parent ? parent->depth + 1 : 0;
    item.visibility = !parent ? Visible : (((Hidden | Collapsed) & parent->visibility) ? Hidden : Visible);
    item.base = node;
    item.object = 0;
    item.node = node;

    return item;
}

SceneListModel::Item SceneListModel::buildObjectItem(Item* parent, BaseObject* object)
{
    SceneListModel::Item item;

    item.parent = parent;
    item.depth = parent ? parent->depth + 1 : 0;
    item.visibility = !parent ? Visible : (((Hidden | Collapsed) & parent->visibility) ? Hidden : Visible);
    item.base = object;
    item.object = object;
    item.node = parent->node;

    return item;
}

void SceneListModel::setScene(Scene* newScene)
{
    if(newScene == myScene)
        return;

    if(myScene)
        myScene->disconnect(this);

    myScene = newScene;

    handleSceneChange(myScene);

    sceneChanged(newScene);
}

int	SceneListModel::rowCount(const QModelIndex & parent) const
{
    return myItems.size();
}

QVariant SceneListModel::data(const QModelIndex& index, int role) const
{
    if(!index.isValid())
    {
        qWarning("Invalid index");
        return QVariant("");
    }

    if(index.row() >= myItems.size())
        return QVariant("");

    const Item& item = myItems[index.row()];
    int depth = item.depth;
    int visibility = item.visibility;
    Base* base = item.base;
    BaseObject* object = item.object;

    if(0 == base)
    {
        qWarning("Item is empty");
        return QVariant("");
    }

    switch(role)
    {
    case NameRole:
        return QVariant::fromValue(QString(base->name.getValue().c_str()));
    case DepthRole:
        return QVariant::fromValue(depth);
    case VisibilityRole:
        return QVariant::fromValue(visibility);
    case TypeRole:
        return QVariant::fromValue(QString(base->getClass()->className.c_str()));
    case IsNodeRole:
        return QVariant::fromValue(0 == object);
    default:
        qWarning("Role unknown");
    }

    return QVariant("");
}

QHash<int,QByteArray> SceneListModel::roleNames() const
{
    QHash<int,QByteArray> roles;

    roles[NameRole]         = "name";
    roles[DepthRole]        = "depth";
    roles[VisibilityRole]   = "visibility";
    roles[TypeRole]         = "type";
    roles[IsNodeRole]       = "isNode";

    return roles;
}

void SceneListModel::setCollapsed(int row, bool collapsed)
{
    if(-1 == row)
    {
        qWarning("Invalid index");
        return;
    }

    if(row >= myItems.size())
        return;

    Item& item = myItems[row];

    int collapsedFlag = collapsed ? Collapsed : Visible;
    if(collapsedFlag == item.visibility)
        return;

    item.visibility = collapsed;

    QStack<Item*> children;
    for(int i = 0; i < item.children.size(); ++i)
        children.append(item.children[i]);

    while(!children.isEmpty())
    {
        Item* child = children.pop();
        if(1 == collapsed)
            child->visibility |= Hidden;
        else
            child->visibility ^= Hidden;

        if(!(Collapsed & child->visibility))
            for(int i = 0; i < child->children.size(); ++i)
                children.append(child->children[i]);
    }

    dataChanged(createIndex(0, 0), createIndex(myItems.size() - 1, 0));
}

// TODO: prefer iteration to recursivity
//static bool isAncestor(BaseNode* ancestor, BaseNode* node)
//{
//    if(!node)
//        return false;

//    if(0 == ancestor && 0 == node->getParents().size())
//        return true;

//    BaseNode::Parents parents = node->getParents();
//    for(int i = 0; i != parents.size(); ++i)
//    {
//        BaseNode* parent = parents[i];
//        if(ancestor == parent)
//        {
//            return true;
//        }
//        else
//        {
//            bool status = isAncestor(ancestor, parent);
//            if(status)
//                return true;
//        }
//    }

//    return false;
//}

void SceneListModel::addChild(Node* parent, Node* child)
{
    if(!child)
        return;

    if(!parent)
    {
        myItems.append(buildNodeItem(0, child));
    }
    else
    {
        QList<Item>::iterator parentItemIt = myItems.begin();
        while(parentItemIt != myItems.end())
        {
            if(parent == parentItemIt->base)
            {
                Item* parentItem = &*parentItemIt;

                QList<Item>::iterator itemIt = parentItemIt;
                while(++itemIt != myItems.end())
                    if(parent != itemIt->node)
                        break;

                QList<Item>::iterator childItemIt = myItems.insert(itemIt, buildNodeItem(parentItem, child));
                parentItem->children.append(&*childItemIt);

                parentItemIt = childItemIt;
            }
            else
                ++parentItemIt;
        }
    }

    MutationListener::addChild(parent, child);
}

void SceneListModel::removeChild(Node* parent, Node* child)
{
    if(!child)
        return;

    MutationListener::removeChild(parent, child);

    QList<Item>::iterator itemIt = myItems.begin();
    while(itemIt != myItems.end())
    {
        if(child == itemIt->node)
        {
            Item* parentItem = itemIt->parent;
            if((!parent && !parentItem) || (parent && parentItem && parent == parentItem->base))
            {
                if(parentItem)
                {
                    int index = parentItem->children.indexOf(&*itemIt);
                    if(-1 != index)
                        parentItem->children.remove(index);
                }

                itemIt = myItems.erase(itemIt);
            }
            else
                ++itemIt;
        }
        else
        {
            ++itemIt;
        }
    }
}

//void SceneListModel::moveChild(Node* previous, Node* parent, Node* child)
//{

//}

void SceneListModel::addObject(Node* parent, BaseObject* object)
{
    if(!object || !parent)
        return;

    QList<Item>::iterator parentItemIt = myItems.begin();
    while(parentItemIt != myItems.end())
    {
        if(parent == parentItemIt->base)
        {
            Item* parentItem = &*parentItemIt;

            QList<Item>::iterator itemIt = parentItemIt;
            while(++itemIt != myItems.end())
                if(parent != itemIt->node)
                    break;

            QList<Item>::iterator childItemIt = myItems.insert(itemIt, buildObjectItem(parentItem, object));
            parentItem->children.append(&*childItemIt);

            parentItemIt = childItemIt;
        }
        else
            ++parentItemIt;
    }

    MutationListener::addObject(parent, object);
}

void SceneListModel::removeObject(Node* parent, BaseObject* object)
{
    if(!object || !parent)
        return;

    MutationListener::removeObject(parent, object);

    QList<Item>::iterator itemIt = myItems.begin();
    while(itemIt != myItems.end())
    {
        if(object == itemIt->base)
        {
            Item* parentItem = itemIt->parent;
            if(parentItem && parent == parentItem->base)
            {
                if(parentItem)
                {
                    int index = parentItem->children.indexOf(&*itemIt);
                    if(-1 != index)
                        parentItem->children.remove(index);
                }

                itemIt = myItems.erase(itemIt);
            }
            else
                ++itemIt;
        }
        else
        {
            ++itemIt;
        }
    }
}

//void SceneListModel::moveObject(Node* previous, Node* parent, BaseObject* object)
//{

//}

void SceneListModel::addSlave(BaseObject* master, BaseObject* slave)
{
    MutationListener::addSlave(master, slave);
}

void SceneListModel::removeSlave(BaseObject* master, BaseObject* slave)
{
    MutationListener::removeSlave(master, slave);
}

}

}
