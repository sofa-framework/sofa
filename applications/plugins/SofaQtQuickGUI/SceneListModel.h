#ifndef SCENELISTMODEL_H
#define SCENELISTMODEL_H

#include "SofaQtQuickGUI.h"
#include "Scene.h"

#include <sofa/simulation/common/MutationListener.h>

#include <QAbstractListModel>
#include <QQmlParserStatus>

namespace sofa
{

namespace qtquick
{

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
    Q_PROPERTY(sofa::qtquick::Scene* scene READ scene WRITE setScene NOTIFY sceneChanged)

public:
    Scene* scene() const		{return myScene;}
    void setScene(Scene* newScene);

protected:
    int	rowCount(const QModelIndex & parent = QModelIndex()) const;
    QVariant data(const QModelIndex & index, int role = Qt::DisplayRole) const;
    QHash<int, QByteArray> roleNames() const;

    Q_INVOKABLE QVariant get(int row) const;
    Q_INVOKABLE sofa::qtquick::SceneComponent* getComponentById(int row) const;
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

    Item buildNodeItem(Item* parent, sofa::core::objectmodel::BaseNode* node) const;
    Item buildObjectItem(Item* parent, sofa::core::objectmodel::BaseObject* object) const;


private:
    QList<Item>                     myItems;
    int                             myUpdatedCount;
    Scene*                          myScene;

};

}

}

#endif // SCENELISTMODEL_H
