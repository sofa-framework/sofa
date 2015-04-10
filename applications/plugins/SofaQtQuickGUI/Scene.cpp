#include <GL/glew.h>
#include "Scene.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/GUIEvent.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/simulation/common/xml/initXml.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/visual/DrawToolGL.h>
#include <sofa/helper/system/glut.h>

#include <sstream>
#include <qqml.h>
#include <QtCore/QCoreApplication>
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

SceneComponent::SceneComponent(const Scene* scene, const sofa::core::objectmodel::Base* base) : QObject(),
    myScene(scene),
    myBase(base)
{

}

Base* SceneComponent::base()
{
    return const_cast<Base*>(static_cast<const SceneComponent*>(this)->base());
}

const Base* SceneComponent::base() const
{
    // check object existence
    if(myScene && myBase)
        if(myScene->myBases.contains(myBase))
            return myBase;

    myBase = 0;
    return myBase;
}

const Scene* SceneComponent::scene() const
{
    return myScene;
}

SceneData::SceneData(const SceneComponent* sceneComponent, const sofa::core::objectmodel::BaseData* data) : QObject(),
    myScene(sceneComponent->scene()),
    myBase(sceneComponent->base()),
    myData(data)
{

}

SceneData::SceneData(const Scene* scene, const sofa::core::objectmodel::Base* base, const sofa::core::objectmodel::BaseData* data) : QObject(),
    myScene(scene),
    myBase(base),
    myData(data)
{

}

QVariantMap SceneData::object() const
{
    const BaseData* data = SceneData::data();
    if(data)
        return Scene::dataObject(data);

    return QVariantMap();
}

bool SceneData::setValue(const QVariant& value)
{
    BaseData* data = SceneData::data();
    if(data)
        return Scene::setDataValue(data, value);

    return false;
}

bool SceneData::setLink(const QString& path)
{
    BaseData* data = SceneData::data();
    if(data)
    {
        std::streambuf* backup(std::cerr.rdbuf());

        std::ostringstream stream;
        std::cerr.rdbuf(stream.rdbuf());
        bool status = Scene::setDataLink(data, path);
        std::cerr.rdbuf(backup);

        return status;
    }

    return false;
}

BaseData* SceneData::data()
{
    return const_cast<BaseData*>(static_cast<const SceneData*>(this)->data());
}

const BaseData* SceneData::data() const
{
    // check if the base still exists hence if the data is still valid
    const Base* base = 0;
    if(myScene && myBase)
        if(myScene->myBases.contains(myBase))
            base = myBase;

    myBase = base;
    if(!myBase)
        myData = 0;

    return myData;
}

Scene::Scene(QObject *parent) : QObject(parent),
	myStatus(Status::Null),
	mySource(),
    mySourceQML(),
    myPathQML(),
	myIsInit(false),
    myVisualDirty(false),
	myDt(0.04),
	myPlay(false),
	myAsynchronous(true),
	mySofaSimulation(0),
    myStepTimer(new QTimer(this)),
    myBases()
{
	sofa::core::ExecParams::defaultInstance()->setAspectID(0);
	boost::shared_ptr<sofa::core::ObjectFactory::ClassEntry> classVisualModel;
	sofa::core::ObjectFactory::AddAlias("VisualModel", "OglModel", true, &classVisualModel);

	myStepTimer->setInterval(0);
	mySofaSimulation = sofa::simulation::graph::getSimulation();

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
    connect(this, &Scene::statusChanged, this, &Scene::handleStatusChange);
    connect(this, &Scene::loaded, this, [&]() {addChild(0, mySofaSimulation->GetRoot().get());});
    connect(this, &Scene::aboutToUnload, this, [&]() {myBases.clear();});

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
        if(sofaSimulation->GetRoot()) {
            sofaSimulation->init(sofaSimulation->GetRoot().get());
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
    myPathQML.clear();
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

	finalFilename.replace("\\", "/");

    aboutToUnload();

	setStatus(Status::Loading);

	setPlay(false);
	myIsInit = false;

    std::string qmlFilepath = (finalFilename + ".qml").toLatin1().constData();
    if(!sofa::helper::system::DataRepository.findFile(qmlFilepath))
        qmlFilepath.clear();

    myPathQML = QString::fromStdString(qmlFilepath);

    mySofaSimulation->unload(mySofaSimulation->GetRoot());

	if(myAsynchronous)
	{
        LoaderThread* loaderThread = new LoaderThread(mySofaSimulation, finalFilename);

        connect(loaderThread, &QThread::finished, this, [this, loaderThread, qmlFilepath]() {                    
            if(!loaderThread->isLoaded())
                setStatus(Status::Error);
            else
                myIsInit = true;

            loaderThread->deleteLater();
        });

		loaderThread->start();
	}
    else
	{
        if(!LoaderProcess(mySofaSimulation, finalFilename))
            setStatus(Status::Error);
        else
            myIsInit = true;
	}
}

void Scene::handleStatusChange(Scene::Status newStatus)
{
    switch(newStatus)
    {
    case Status::Null:
        break;
    case Status::Ready:
        loaded();
        break;
    case Status::Loading:
        break;
    case Status::Error:
        break;
    default:
        qWarning() << "Scene status unknown";
        break;
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

double Scene::radius() const
{
	QVector3D min, max;
	computeBoundingBox(min, max);
	QVector3D diag = (max - min);

	return diag.length();
}

void Scene::computeBoundingBox(QVector3D& min, QVector3D& max) const
{
	SReal pmin[3], pmax[3];
    mySofaSimulation->computeTotalBBox(mySofaSimulation->GetRoot().get(), pmin, pmax);

	min = QVector3D(pmin[0], pmin[1], pmin[2]);
	max = QVector3D(pmax[0], pmax[1], pmax[2]);
}

QString Scene::dumpGraph() const
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

void Scene::reinitComponent(const QString& path)
{
    QStringList pathComponents = path.split("/");

    Node::SPtr node = mySofaSimulation->GetRoot();
    unsigned int i = 0;
    while(i < pathComponents.size()-1) {
        if (pathComponents[i]=="@") {
            ++i;
            continue;
        }

        node = node->getChild(pathComponents[i].toStdString());
        if (!node) {
            qWarning() << "Object path unknown:" << path;
            return;
        }
        ++i;
    }
    BaseObject* object = node->get<BaseObject>(pathComponents[i].toStdString());
    if(!object) {
        qWarning() << "Object path unknown:" << path;
        return;
    }
    object->reinit();
}

void Scene::sendGUIEvent(const QString& controlID, const QString& valueName, const QString& value)
{
    if(!mySofaSimulation->GetRoot())
        return;

    sofa::core::objectmodel::GUIEvent event(controlID.toUtf8().constData(), valueName.toUtf8().constData(), value.toUtf8().constData());
    mySofaSimulation->GetRoot()->propagateEvent(sofa::core::ExecParams::defaultInstance(), &event);
}

QVariantMap Scene::dataObject(const sofa::core::objectmodel::BaseData* data)
{
    QVariantMap object;

    if(!data)
    {
        object.insert("name", "Invalid");
        object.insert("description", "");
        object.insert("type", "");
        object.insert("group", "");
        object.insert("properties", "");
        object.insert("link", "");
        object.insert("value", "");

        return object;
    }

    // TODO:
    QString type;
    const AbstractTypeInfo* typeinfo = data->getValueTypeInfo();

    QVariantMap properties;

    if(typeinfo->Text())
    {
        type = "string";
    }
    else if(typeinfo->Scalar())
    {
        type = "number";
        properties.insert("step", 0.1);
        properties.insert("decimals", 3);
    }
    else if(typeinfo->Integer())
    {
        if(std::string::npos != typeinfo->name().find("bool"))
        {
            type = "boolean";
            properties.insert("autoUpdate", true);
        }
        else
        {
            type = "number";
            properties.insert("decimals", 0);
            if(std::string::npos != typeinfo->name().find("unsigned"))
                properties.insert("min", 0);
        }
    }
    else
    {
        type = QString::fromStdString(data->getValueTypeString());
    }

    if(typeinfo->Container())
    {
        type = "array";
        int nbCols = typeinfo->size();

        properties.insert("cols", nbCols);
        if(typeinfo->FixedSize())
            properties.insert("static", true);

        const AbstractTypeInfo* baseTypeinfo = typeinfo->BaseType();
        if(baseTypeinfo->FixedSize())
            properties.insert("innerStatic", true);
    }

    QString widget(data->getWidget());
    if(!widget.isEmpty())
        type = widget;

    properties.insert("readOnly", false);

    object.insert("name", data->getName().c_str());
    object.insert("description", data->getHelp());
    object.insert("type", type);
    object.insert("group", data->getGroup());
    object.insert("properties", properties);
    object.insert("link", QString::fromStdString(data->getLinkPath()));
    object.insert("value", dataValue(data));

    return object;
}

QVariant Scene::dataValue(const BaseData* data)
{
    QVariant value;

    if(!data)
        return value;

    const AbstractTypeInfo* typeinfo = data->getValueTypeInfo();
    const void* valueVoidPtr = data->getValueVoidPtr();

    if(!typeinfo->Container())
    {
        if(typeinfo->Text())
            value = QString::fromStdString(typeinfo->getTextValue(valueVoidPtr, 0));
        else if(typeinfo->Scalar())
            value = typeinfo->getScalarValue(valueVoidPtr, 0);
        else if(typeinfo->Integer())
        {
            if(std::string::npos != typeinfo->name().find("bool"))
                value = 0 != typeinfo->getIntegerValue(valueVoidPtr, 0) ? true : false;
            else
                value = typeinfo->getIntegerValue(valueVoidPtr, 0);
        }
        else
        {
            value = QString::fromStdString(data->getValueString());
        }
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

            bool isBool = false;
            if(std::string::npos != typeinfo->name().find("bool"))
                isBool = true;

            for(int j = 0; j < nbRows; j++)
            {
                subValues.clear();

                if(isBool)
                    for(int i = 0; i < nbCols; i++)
                        subValues.append(QVariant::fromValue(0 != typeinfo->getIntegerValue(valueVoidPtr, j * nbCols + i) ? true : false));
                else
                    for(int i = 0; i < nbCols; i++)
                        subValues.append(QVariant::fromValue(typeinfo->getIntegerValue(valueVoidPtr, j * nbCols + i)));

                values.push_back(QVariant::fromValue(subValues));
            }

            value = values;
        }
        else
        {
            value = QString::fromStdString(data->getValueString());
        }
    }

    return value;
}

// TODO: WARNING : do not use data->read anymore but directly the correct set*Type*Value(...)
bool Scene::setDataValue(BaseData* data, const QVariant& value)
{
    if(!data)
        return false;

    const AbstractTypeInfo* typeinfo = data->getValueTypeInfo();

    if(!value.isNull())
    {
        QVariant finalValue = value;
        if(finalValue.userType() == qMetaTypeId<QJSValue>())
            finalValue = finalValue.value<QJSValue>().toVariant();

        if(QVariant::List == finalValue.type())
        {
            QSequentialIterable valueIterable = finalValue.value<QSequentialIterable>();

            int nbCols = typeinfo->size();
            int nbRows = typeinfo->size(data->getValueVoidPtr()) / nbCols;

            if(!typeinfo->Container())
            {
                qWarning("Trying to set a list of values on a non-container data");
                return false;
            }

            if(valueIterable.size() != nbRows)
            {
                if(typeinfo->FixedSize())
                {
                    qWarning() << "The new data should have the same size, should be" << nbRows << ", got" << valueIterable.size();
                    return false;
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
                            qWarning() << "The new sub data should have the same size, should be" << nbCols << ", got" << subValueIterable.size() << "- data size is:" << valueIterable.size();
                            return false;
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
                            qWarning() << "The new sub data should have the same size, should be" << nbCols << ", got" << subValueIterable.size() << "- data size is:" << valueIterable.size();
                            return false;
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
            else if(typeinfo->Text())
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
                            qWarning() << "The new sub data should have the same size, should be" << nbCols << ", got" << subValueIterable.size() << "- data size is:" << valueIterable.size();
                            return false;
                        }

                        for(int j = 0; j < subValueIterable.size(); ++j)
                        {
                            dataString += subValueIterable.at(j).toString();
                            if(subValueIterable.size() - 1 != j)
                                dataString += ' ';
                        }
                    }
                    else
                    {
                        dataString += subFinalValue.toString();
                    }

                    if(valueIterable.size() - 1 != i)
                        dataString += ' ';
                }

                data->read(dataString.toStdString());
            }
            else
                data->read(value.toString().toStdString());
        }
        else if(QVariant::Map == finalValue.type())
        {
            qWarning("Map type is not supported");
            return false;
        }
        else
        {
            if(typeinfo->Text())
                data->read(value.toString().toStdString());
            else if(typeinfo->Scalar())
                data->read(QString::number(value.toDouble()).toStdString());
            else if(typeinfo->Integer())
                data->read(QString::number(value.toLongLong()).toStdString());
            else
                data->read(value.toString().toStdString());
        }
    }
    else
        return false;

    return true;
}

bool Scene::setDataLink(BaseData* data, const QString& link)
{
    if(!data)
        return false;

    if(link.isEmpty())
        data->setParent(0);
    else
        data->setParent(link.toStdString());

    return data->getParent();
}

QVariant Scene::dataValue(const QString& path) const
{
    return onDataValue(path);
}

void Scene::setDataValue(const QString& path, const QVariant& value)
{
    onSetDataValue(path, value);
}

static BaseData* FindDataHelper(BaseNode* node, const QString& path)
{
    BaseData* data = 0;
    std::streambuf* backup(std::cerr.rdbuf());

    std::ostringstream stream;
    std::cerr.rdbuf(stream.rdbuf());
    node->findDataLinkDest(data, path.toStdString(), 0);
    std::cerr.rdbuf(backup);

    return data;
}

SceneData* Scene::data(const QString& path) const
{
    BaseData* data = FindDataHelper(mySofaSimulation->GetRoot().get(), path);
    if(!data)
        return 0;

    Base* base = data->getOwner();
    if(!base)
        return 0;

    return new SceneData(this, base, data);
}

SceneComponent* Scene::component(const QString& path) const
{
    BaseData* data = FindDataHelper(mySofaSimulation->GetRoot().get(), path + ".name"); // search for the "name" data of the component (this data is always present if the component exist)

    if(!data)
        return 0;

    Base* base = data->getOwner();
    if(!base)
        return 0;

    return new SceneComponent(this, base);
}

QVariant Scene::onDataValue(const QString& path) const
{
    BaseData* data = FindDataHelper(mySofaSimulation->GetRoot().get(), path);

    if(!data)
    {
        qWarning() << "DataPath unknown:" << path;
        return QVariant();
    }

    return dataValue(data);
}

void Scene::onSetDataValue(const QString& path, const QVariant& value)
{
    BaseData* data = FindDataHelper(mySofaSimulation->GetRoot().get(), path);

    if(!data)
    {
        qWarning() << "DataPath unknown:" << path;
    }
    else
    {
        if(!value.isNull())
        {
            QVariant finalValue = value;
            if(finalValue.userType() == qMetaTypeId<QJSValue>())
                finalValue = finalValue.value<QJSValue>().toVariant();

            // arguments from JS are packed in an array, we have to unpack it
            if(QVariant::List == finalValue.type())
            {
                QSequentialIterable valueIterable = finalValue.value<QSequentialIterable>();
                if(1 == valueIterable.size())
                    finalValue = valueIterable.at(0);
            }

            setDataValue(data, finalValue);
        }
    }
}

void Scene::initGraphics()
{
    if(!myIsInit)
        return;

    if(!mySofaSimulation->GetRoot())
    {
        setStatus(Status::Error);
		return;
    }

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

    // WARNING: some plugins like "image" need a valid OpenGL Context during init because they are initing textures during init instead of initTextures ...
	mySofaSimulation->initTextures(mySofaSimulation->GetRoot().get());
	setDt(mySofaSimulation->GetRoot()->getDt());

    setStatus(Status::Ready);

    if(!myPathQML.isEmpty())
        setSourceQML(QUrl::fromLocalFile(myPathQML));
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

void Scene::addChild(Node* parent, Node* child)
{
    if(!child)
        return;

    myBases.insert(child);

    MutationListener::addChild(parent, child);
}

void Scene::removeChild(Node* parent, Node* child)
{
    if(!child)
        return;

    MutationListener::removeChild(parent, child);

    myBases.remove(child);
}

void Scene::addObject(Node* parent, BaseObject* object)
{
    if(!object || !parent)
        return;

    myBases.insert(object);

    MutationListener::addObject(parent, object);
}

void Scene::removeObject(Node* parent, BaseObject* object)
{
    if(!object || !parent)
        return;

    MutationListener::removeObject(parent, object);

    myBases.remove(object);
}

}

}
