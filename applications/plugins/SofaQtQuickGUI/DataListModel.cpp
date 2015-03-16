#include "DataListModel.h"

#include <QStack>
#include <QDebug>

namespace sofa
{

namespace qtquick
{

using namespace sofa::defaulttype;
using namespace sofa::core::objectmodel;
using namespace sofa::simulation;

DataListModel::DataListModel(QObject* parent) : QAbstractListModel(parent),
    myItems(),
    myUpdatedCount(0),
    mySceneComponent(0)
{
    connect(this, &DataListModel::sceneComponentChanged, &DataListModel::update);
}

DataListModel::~DataListModel()
{

}

void DataListModel::update()
{
    myItems.clear();

    if(mySceneComponent)
    {
        const Base* base = mySceneComponent->base();
        if(base)
        {
            sofa::helper::vector<BaseData*> dataFields = base->getDataFields();
            for(unsigned int i = 0; i < dataFields.size(); ++i)
                myItems.append(buildDataItem(dataFields[i]));

            qStableSort(myItems.begin(), myItems.end(), [](const Item& a, const Item& b) {return QString::compare(a.data->getGroup(), b.data->getGroup()) < 0;});
        }
        else
        {
            setSceneComponent(0);
        }
    }

    /*int changeNum = myItems.size() - myUpdatedCount;
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

    dataChanged(createIndex(0, 0), createIndex(myItems.size() - 1, 0));*/

    // TODO: why the old system does not work ?

    beginRemoveRows(QModelIndex(), 0, myUpdatedCount - 1);
    endRemoveRows();

    beginInsertRows(QModelIndex(), 0, myItems.size() - 1);
    endInsertRows();

    myUpdatedCount = myItems.size();
}

DataListModel::Item DataListModel::buildDataItem(BaseData* data) const
{
    DataListModel::Item item;

    item.data = data;

    return item;
}

void DataListModel::setSceneComponent(SceneComponent* newSceneComponent)
{
    if(newSceneComponent == mySceneComponent)
        return;

    mySceneComponent = newSceneComponent;

    sceneComponentChanged(newSceneComponent);
}

int	DataListModel::rowCount(const QModelIndex & /*parent*/) const
{
    return myItems.size();
}

QVariant DataListModel::data(const QModelIndex& index, int role) const
{
    if(!index.isValid())
    {
        qWarning("Invalid index");
        return QVariant("");
    }

    if(index.row() >= myItems.size())
        return QVariant("");

    if(!mySceneComponent->base())
    {
        //the base is not valid anymore, neither is its data
        mySceneComponent = 0;
        sceneComponentChanged(0);

        return QVariant("");
    }

    const Item& item = myItems[index.row()];
    BaseData* data = item.data;

    switch(role)
    {
    case NameRole:
        return QVariant::fromValue(QString(data->getName().c_str()));
    case GroupRole:
    {
        QString group = data->getGroup();
        if(group.isEmpty())
            group = "Base";

        return QVariant::fromValue(group);
    }
    case ValueRole:
        return QVariant::fromValue(Scene::dataValue(data));
    default:
        qWarning("Role unknown");
    }

    return QVariant("");
}

QHash<int,QByteArray> DataListModel::roleNames() const
{
    QHash<int,QByteArray> roles;

    roles[NameRole]         = "name";
    roles[GroupRole]        = "group";
    roles[ValueRole]        = "value";

    return roles;
}

SceneData* DataListModel::getDataById(int row) const
{
    if(row < 0 || row >= myItems.size())
        return 0;

    return new SceneData(mySceneComponent, myItems.at(row).data);
}

}

}
