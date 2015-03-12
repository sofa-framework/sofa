#ifndef DATALISTMODEL_H
#define DATALISTMODEL_H

#include "SofaQtQuickGUI.h"
#include "Scene.h"
#include "SceneListModel.h"

#include <sofa/core/objectmodel/Base.h>

#include <QAbstractListModel>
#include <QList>

class QTimer;
class QVector3D;

namespace sofa
{

namespace qtquick
{

class DataListModel : public QAbstractListModel
{
    Q_OBJECT

public:
    DataListModel(QObject* parent = 0);
    ~DataListModel();

    Q_INVOKABLE void update();

public:
    Q_PROPERTY(sofa::qtquick::SceneComponent* sceneComponent READ sceneComponent WRITE setSceneComponent NOTIFY sceneComponentChanged);

public:
    SceneComponent* sceneComponent() const		{return mySceneComponent;}
    void setSceneComponent(SceneComponent* newSceneComponent);

protected:
    int	rowCount(const QModelIndex & parent = QModelIndex()) const;
    QVariant data(const QModelIndex & index, int role = Qt::DisplayRole) const;
    QHash<int,QByteArray> roleNames() const;

    Q_INVOKABLE sofa::qtquick::SceneData* getDataById(int row) const;
signals:
    void sceneComponentChanged(SceneComponent* newSceneComponent) const;

private:
    enum {
        NameRole = Qt::UserRole + 1,
        GroupRole,
        ValueRole
    };

    struct Item
    {
        Item() :
            data(0)
        {

        }

        sofa::core::objectmodel::BaseData*      data;
    };

    Item buildDataItem(BaseData* data) const;

private:
    QList<Item>             myItems;
    int                     myUpdatedCount;
    mutable SceneComponent* mySceneComponent;

};

}

}

#endif // DATALISTMODEL_H
