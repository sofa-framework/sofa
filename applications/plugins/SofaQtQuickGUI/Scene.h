#ifndef SCENE_H
#define SCENE_H

#include "SofaQtQuickGUI.h"

#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/MutationListener.h>

#include <QObject>
#include <QUrl>
#include <QAbstractListModel>
#include <QQmlParserStatus>
#include <QList>

class QTimer;
class QVector3D;

namespace sofa
{

namespace qtquick
{

class SceneListModel;

class Scene : public QObject
{
    Q_OBJECT

public:
    explicit Scene(QObject *parent = 0);
	~Scene();

public:
	Q_PROPERTY(Status status READ status WRITE setStatus NOTIFY statusChanged);
	Q_PROPERTY(QUrl source READ source WRITE setSource NOTIFY sourceChanged);
	Q_PROPERTY(QUrl sourceQML READ sourceQML WRITE setSourceQML NOTIFY sourceQMLChanged);
	Q_PROPERTY(double dt READ dt WRITE setDt NOTIFY dtChanged);
	Q_PROPERTY(bool play READ playing WRITE setPlay NOTIFY playChanged)
	Q_PROPERTY(bool asynchronous MEMBER myAsynchronous NOTIFY asynchronousChanged)
    Q_PROPERTY(bool visualDirty READ visualDirty NOTIFY visualDirtyChanged)

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

    bool visualDirty() const						{return myVisualDirty;}
    void setVisualDirty(bool newVisualDirty);

signals:
	void loaded();
    void aboutToUnload();
	void statusChanged(Status newStatus);
	void sourceChanged(const QUrl& newSource);
	void sourceQMLChanged(const QUrl& newSourceQML);
	void dtChanged(double newDt);
	void playChanged(bool newPlay);
	void asynchronousChanged(bool newAsynchronous);
    void visualDirtyChanged(bool newVisualDirty);

public:
	Q_INVOKABLE double radius();
	Q_INVOKABLE void computeBoundingBox(QVector3D& min, QVector3D& max);
    Q_INVOKABLE QString dumpGraph();

public:
    QVariant getData(const QString& path) const;
    void setData(const QString& path, const QVariant& value);

protected:
    Q_INVOKABLE QVariant onGetData(const QString& path) const;
    Q_INVOKABLE void onSetData(const QString& path, const QVariant& value);

public slots:
    void init();        // need an opengl context made current
	void reload();
	void step();
	void reset();
	void draw();

	void onKeyPressed(char key);
	void onKeyReleased(char key);

signals:
	void stepBegin();
    void stepEnd();
    void reseted();

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

class SceneListModel : public QAbstractListModel, public QQmlParserStatus, private sofa::simulation::MutationListener
{
    Q_OBJECT
    Q_INTERFACES(QQmlParserStatus)

public:
    SceneListModel(QObject* parent = 0);
    ~SceneListModel();

    void classBegin();
    void componentComplete();

    Q_INVOKABLE void update();

    Q_ENUMS(Visibility)
    enum Visibility {
        Visible     = 0,
        Collapsed   = 1,
        Hidden      = 2
    };

public slots:
    void handleSceneChange(Scene* newScene);
    void clear();

public:
    Q_PROPERTY(sofa::qtquick::Scene* scene READ scene WRITE setScene NOTIFY sceneChanged);

public:
    Scene* scene() const		{return myScene;}
    void setScene(Scene* newScene);

protected:
    int	rowCount(const QModelIndex & parent = QModelIndex()) const;
    QVariant data(const QModelIndex & index, int role = Qt::DisplayRole) const;
    QHash<int,QByteArray> roleNames() const;

    Q_INVOKABLE void setCollapsed(int row, bool value);

signals:
    void sceneChanged(sofa::qtquick::Scene* newScene);

protected:
    void addChild(sofa::simulation::Node* parent, sofa::simulation::Node* child);
    void removeChild(sofa::simulation::Node* parent, sofa::simulation::Node* child);
    //void moveChild(sofa::simulation::Node* previous, sofa::simulation::Node* parent, sofa::simulation::Node* child);
    void addObject(sofa::simulation::Node* parent, sofa::core::objectmodel::BaseObject* object);
    void removeObject(sofa::simulation::Node* parent, sofa::core::objectmodel::BaseObject* object);
    //void moveObject(sofa::simulation::Node* previous, sofa::simulation::Node* parent, sofa::core::objectmodel::BaseObject* object);
    void addSlave(sofa::core::objectmodel::BaseObject* master, sofa::core::objectmodel::BaseObject* slave);
    void removeSlave(sofa::core::objectmodel::BaseObject* master, sofa::core::objectmodel::BaseObject* slave);
    //void moveSlave(sofa::core::objectmodel::BaseObject* previousMaster, sofa::core::objectmodel::BaseObject* master, sofa::core::objectmodel::BaseObject* slave);
    //void sleepChanged(sofa::simulation::Node* node);

private:
    enum {
        NameRole = Qt::UserRole + 1,
        DepthRole,
        VisibilityRole,
        TypeRole,
        IsNodeRole
    };

    struct Item
    {
        Item() :
            parent(0),
            children(0),
            depth(0),
            visibility(0),
            base(0),
            object(0),
            context(0),
            node(0)
        {

        }

        Item*                                   parent;
        QVector<Item*>                          children;
        int                                     depth;
        int                                     visibility;

        sofa::core::objectmodel::Base*          base;
        sofa::core::objectmodel::BaseObject*    object;
        sofa::core::objectmodel::BaseContext*   context;
        sofa::core::objectmodel::BaseNode*      node;
    };

    Item buildNodeItem(Item* parent, sofa::core::objectmodel::BaseNode* node);
    Item buildObjectItem(Item* parent, sofa::core::objectmodel::BaseObject* object);

private:
    QList<Item>                     myItems;
    int                             myUpdatedCount;
    Scene*                          myScene;

};

}

}

#endif // SCENE_H
