#include <GL/glew.h>
#include "Scene.h"

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
#include <sofa/defaulttype/DataTypeInfo.h>

#include <qqml.h>
#include <QVector3D>
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

Scene::Scene(QObject *parent) : QAbstractListModel(parent), MutationListener(),
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
    mySceneModelItems()
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

int	Scene::rowCount(const QModelIndex & parent) const
{
    return mySceneModelItems.size();
}

QVariant Scene::data(const QModelIndex& index, int role) const
{
    if(!index.isValid() || index.row() >= mySceneModelItems.size())
    {
        qWarning("Invalid index");
        return QVariant();
    }

    const SceneModelItem& sceneModelItem = mySceneModelItems[index.row()];

    int parent = sceneModelItem.parentIndex;
    int depth = sceneModelItem.depth;
    Base* base = sceneModelItem.base;
    BaseObject* object = sceneModelItem.object;

    if(0 == base)
    {
        qWarning("Item empty");
        return QVariant();
    }

    switch(role)
    {
    case NameRole:
        return QVariant::fromValue(QString(base->name.getValue().c_str()));
    case ParentIndexRole:
        return QVariant::fromValue(parent);
    case DepthRole:
        return QVariant::fromValue(depth);
    case TypeRole:
        return QVariant::fromValue(QString(base->getClass()->className.c_str()));
    case IsNodeRole:
        return QVariant::fromValue(!object);
    }

    qWarning("Role unknown");
    return QVariant();
}

QHash<int,QByteArray> Scene::roleNames() const
{
    QHash<int,QByteArray> roles;

    roles[NameRole]         = "name";
    roles[ParentIndexRole]  = "parentIndex";
    roles[DepthRole]        = "depth";
    roles[TypeRole]         = "type";
    roles[IsNodeRole]       = "isNode";

    return roles;
}

// TODO: use the MutationListener interface instead
void Scene::update()
{
    // TODO: warning - we are modifying the model but we are not between a begin / end
    mySceneModelItems.clear();

    struct NodeIterator
    {
        int         parent;
        int         depth;
        BaseNode*   node;
    };
    std::stack<NodeIterator> nodeIterarorStack;
	NodeIterator nodeObject;
	nodeObject.parent = -1;
	nodeObject.depth = 0;
	nodeObject.node = sofaSimulation()->GetRoot().get();
    nodeIterarorStack.push(nodeObject);
    while(!nodeIterarorStack.empty())
    {
        NodeIterator& nodeIterator = nodeIterarorStack.top();
        int parent = nodeIterator.parent;
        int depth = nodeIterator.depth;
        BaseNode* node = nodeIterator.node;
        nodeIterarorStack.pop();

        if(0 == node)
            continue;

        int id = mySceneModelItems.size();
        BaseContext* context = dynamic_cast<BaseContext*>(node);

        // node
        SceneModelItem sceneModelItem;
        sceneModelItem.parentIndex  = parent;
        sceneModelItem.depth        = depth;
        sceneModelItem.base         = node;
        sceneModelItem.object       = 0;
        sceneModelItem.context      = context;
        sceneModelItem.node         = node;

        mySceneModelItems.push_back(sceneModelItem);

        // objects
        std::vector<BaseObject*> objects;
        context->get<BaseObject>(&objects, BaseContext::Local);

        for(int i = 0; i < objects.size(); ++i)
        {
            int j = objects.size() - 1 - i;
            BaseObject* object = objects[j];

            if(0 == object)
                continue;

            SceneModelItem sceneModelItem;
            sceneModelItem.parentIndex  = id;
            sceneModelItem.depth        = depth + 1;
            sceneModelItem.base         = object;
            sceneModelItem.object       = object;
            sceneModelItem.context      = context;
            sceneModelItem.node         = node;

            mySceneModelItems.push_back(sceneModelItem);
        }

        // nodes
        for(int i = 0; i < node->getChildren().size(); ++i)
		{
			NodeIterator nodeTmp;
			int j = node->getChildren().size() - 1 - i;
			nodeTmp.parent=id; nodeTmp.depth=depth + 1; nodeTmp.node=node->getChildren()[j];
            nodeIterarorStack.push(nodeTmp);
        }
    }

    beginInsertRows(QModelIndex(), 0, mySceneModelItems.size() - 1);
    endInsertRows();
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

    if(mySofaSimulation->GetRoot())
        removeChild(0, mySofaSimulation->GetRoot().get());

	std::string qmlFilepath = (finalFilename + ".qml").toLatin1().constData();
	if(!sofa::helper::system::DataRepository.findFile(qmlFilepath))
		qmlFilepath.clear();

    mySofaSimulation->unload(mySofaSimulation->GetRoot());

	if(myAsynchronous)
	{
		LoaderThread* loaderThread = new LoaderThread(mySofaSimulation, finalFilename);
        connect(loaderThread, &QThread::finished, this, [this, loaderThread]() {
            bool loaded = loaderThread && loaderThread->isLoaded();
            if(loaded)
                addChild(0, mySofaSimulation->GetRoot().get());
            setStatus(loaded ? Status::Ready : Status::Error);}
        );
		
		if(!qmlFilepath.empty())
			connect(loaderThread, &QThread::finished, this, [=]() {setSourceQML(QUrl::fromLocalFile(qmlFilepath.c_str()));});

		connect(loaderThread, &QThread::finished, loaderThread, &QObject::deleteLater);
		loaderThread->start();
	}
	else
	{
        bool loaded = LoaderProcess(mySofaSimulation, finalFilename);
        if(loaded)
            addChild(0, mySofaSimulation->GetRoot().get());
        setStatus(loaded ? Status::Ready : Status::Error);
		
		if(!qmlFilepath.empty())
			setSourceQML(QUrl::fromLocalFile(qmlFilepath.c_str()));
	}
}

int Scene::findItemIndex(Base* base) const
{
    for(int i = 0; i != mySceneModelItems.size(); ++i)
        if(base == mySceneModelItems[i].base)
            return i;

    return mySceneModelItems.size();
}

int Scene::findItemIndex(BaseNode* parent, Base* base) const
{
    for(int i = 0; i != mySceneModelItems.size(); ++i)
        if(base == mySceneModelItems[i].base && parent == mySceneModelItems[i].parent)
            return i;

    return mySceneModelItems.size();
}

// TODO: prefer iteration over recursivity
bool Scene::isAncestor(BaseNode* ancestor, BaseNode* node) const
{
    if(0 == ancestor && 0 == node->getParents().size())
        return true;

    BaseNode::Parents parents = node->getParents();
    for(int i = 0; i != parents.size(); ++i)
    {
        BaseNode* parent = parents[i];
        if(ancestor == parent)
        {
            return true;
        }
        else
        {
            bool status = isAncestor(ancestor, parent);
            if(status)
                return true;
        }
    }

    return false;
}

void Scene::addChild(Node* parent, Node* child)
{
    int i = findItemIndex(parent);

    int depth = 0;
    if(i != mySceneModelItems.size())
        depth = mySceneModelItems[i].depth + 1;

    SceneModelItem sceneModelItem;
    sceneModelItem.parentIndex  = i;
    sceneModelItem.depth        = depth;
    sceneModelItem.base         = child;
    sceneModelItem.object       = 0;
    sceneModelItem.context      = dynamic_cast<BaseContext*>(child);
    sceneModelItem.node         = child;
    sceneModelItem.parent       = parent;

    for(; i != mySceneModelItems.size(); ++i)
    {
        int j = i + 1;
        if(j == mySceneModelItems.size())
            break;

        if(!mySceneModelItems[j].object && !isAncestor(parent, mySceneModelItems[j].parent))
            break;
    }

    if(i != mySceneModelItems.size())
        ++i;

    beginInsertRows(QModelIndex(), i, i);
    mySceneModelItems.insert(i, sceneModelItem);
    endInsertRows();

    MutationListener::addChild(parent, child);
}

void Scene::removeChild(Node* parent, Node* child)
{
    MutationListener::removeChild(parent, child);

    int i = findItemIndex(parent, child);
    if(i == mySceneModelItems.size())
    {
        qWarning("Trying to remove an unknown node");
        return;
    }

    beginRemoveRows(QModelIndex(), i, i);
    mySceneModelItems.remove(i);
    endRemoveRows();
}

void Scene::addObject(Node* parent, BaseObject* object)
{
//    int i = findItemIndex(parent);

//    int depth = 0;
//    if(i != mySceneModelItems.size())
//        depth = mySceneModelItems[i].depth + 1;

//    SceneModelItem sceneModelItem;
//    sceneModelItem.parentIndex  = i;
//    sceneModelItem.depth        = depth;
//    sceneModelItem.base         = object;
//    sceneModelItem.object       = object;
//    sceneModelItem.context      = dynamic_cast<BaseContext*>(parent);
//    sceneModelItem.node         = parent;
//    sceneModelItem.parent       = parent;

//    for(; i != mySceneModelItems.size(); ++i)
//    {
//        int j = i + 1;
//        if(j == mySceneModelItems.size())
//            break;

//        if(parent != mySceneModelItems[j].node)
//            break;
//    }

//    if(i != mySceneModelItems.size())
//        ++i;

//    beginInsertRows(QModelIndex(), i, i);
//    mySceneModelItems.insert(i, sceneModelItem);
//    endInsertRows();

//    MutationListener::addObject(parent, object);
}

void Scene::removeObject(Node* parent, BaseObject* object)
{
//    MutationListener::removeObject(parent, object);

//    int i = findItemIndex(parent, object);
//    if(i == mySceneModelItems.size())
//    {
//        qWarning("Trying to remove an unknown object");
//        return;
//    }

//    beginRemoveRows(QModelIndex(), i, i);
//    mySceneModelItems.remove(i);
//    endRemoveRows();
}

void Scene::addSlave(BaseObject* master, BaseObject* slave)
{
    MutationListener::addSlave(master, slave);
}

void Scene::removeSlave(BaseObject* master, BaseObject* slave)
{
    MutationListener::removeSlave(master, slave);
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
