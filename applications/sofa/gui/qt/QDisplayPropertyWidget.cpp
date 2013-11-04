/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "QDisplayPropertyWidget.h"
#include "ModifyObject.h"
#include "DataWidget.h"
#include "QDisplayDataWidget.h"
#include "QDataDescriptionWidget.h"
#include "QTabulationModifyObject.h"

// uncomment to show traces of GUI operations in this file
//#define DEBUG_GUI

namespace sofa
{

namespace gui
{

namespace qt
{

QDisplayTreeItemWidget::QDisplayTreeItemWidget(QWidget* parent, QTreeWidgetItem* item) : QWidget(parent)
    , treeWidgetItem(item)
{

}

QDisplayTreeItemWidget::~QDisplayTreeItemWidget()
{

}

void QDisplayTreeItemWidget::updateDirtyWidget()
{
    const QObjectList& parentList = children();
    for(QObjectList::const_iterator itParent = parentList.begin(); itParent != parentList.end(); ++itParent)
    {
        QWidget* parentWidget = dynamic_cast<QWidget*>(*itParent);
        if(parentWidget)
        {
            const QObjectList& childList = parentWidget->children();
            for(QObjectList::const_iterator itChild = childList.begin(); itChild != childList.end(); ++itChild)
            {
                QWidget* childWidget = dynamic_cast<QWidget*>(*itChild);
                if(childWidget)
                {
                    childWidget->adjustSize();
                    childWidget->setHidden(false);
                }
            }
            parentWidget->adjustSize();
            parentWidget->setHidden(false);
        }
    }

    adjustSize();
    setHidden(false);
    treeWidgetItem->setHidden(false);
}

void QDisplayPropertyWidget::updateListViewItem()
{
    std::map<QTreeWidgetItem*, std::pair<core::objectmodel::Base*, Q3ListViewItem*> >::iterator objectIterator;
    for(objectIterator = objects.begin(); objectIterator != objects.end(); ++objectIterator)
    {
        core::objectmodel::Base* object = objectIterator->second.first;
        Q3ListViewItem* item = objectIterator->second.second;

        if (/*simulation::Node *node=*/dynamic_cast< simulation::Node *>(object))
        {
            item->setText(0,object->getName().c_str());
            //emit nodeNameModification(node);
        }
        else
        {
            QString currentName = item->text(0);

            std::string name=item->text(0).ascii();
            std::string::size_type pos = name.find(' ');
            if(pos != std::string::npos)
                name = name.substr(0,pos);
            name += "  ";
            name += object->getName();
            QString newName(name.c_str());
            if(newName != currentName)
                item->setText(0,newName);
        }
    }
}

void QDisplayPropertyWidget::updateDirtyWidget()
{

}

const QString QDisplayPropertyWidget::defaultGroup("Property");

QDisplayPropertyWidget::QDisplayPropertyWidget(QWidget* parent) : QTreeWidget(parent)
    , pinIcon()
{
	std::string filename = "textures/pin.png";
	sofa::helper::system::DataRepository.findFile(filename);
	pinIcon = QIcon(filename.c_str());

    setColumnCount(2);
    //setIndentation(10);

    headerItem()->setText(0, QString("Property"));
    headerItem()->setText(1, QString("Value"));
    setSelectionMode(QAbstractItemView::NoSelection);
    //setSelectionBehavior(QAbstractItemView::SelectItems);
    setDragEnabled(false);
    setAcceptDrops(false);
    setDropIndicatorShown(false);
    //setDragDropMode(QAbstractItemView::InternalMove);
    setIndentation(0);

    setFocusPolicy(Qt::NoFocus);
    setAutoFillBackground(false);
}

QDisplayPropertyWidget::~QDisplayPropertyWidget()
{

}

void QDisplayPropertyWidget::addComponent(const QString& component, core::objectmodel::Base* base, Q3ListViewItem* listItem, bool clear)
{
    if(clear)
        this->clear();

    // return now if the component to add is empty or if it is already in the tree
    if(component.isEmpty() || !base || findComponent(component))
        return;

    const sofa::core::objectmodel::Base::VecData& fields = base->getDataFields();

    // return now if the component has no field
    if(fields.empty())
        return;

    // finally, add the group
    QTreeWidgetItem *componentItem = new QTreeWidgetItem(this);
    QFont *font = new QFont();
    font->setBold(true);
    componentItem->setFont(0, *font);
    componentItem->setText(0, component);
    QPushButton* pin = new QPushButton(this);
    pin->setFixedSize(QSize(18, 18));
    pin->setCheckable(true);
    pin->setIcon(pinIcon);
    setItemWidget(componentItem, 1, pin);
    componentItem->setExpanded(true);
    QBrush *backgroundBrush = new QBrush(QColor(20, 20, 20));
    QBrush *foregroundBrush = new QBrush(QColor(255, 255, 255));
    componentItem->setBackground(0, *backgroundBrush);
    componentItem->setForeground(0, *foregroundBrush);
    componentItem->setTextAlignment(0, Qt::AlignLeft);
    componentItem->setBackground(1, *backgroundBrush);
    componentItem->setForeground(1, *foregroundBrush);
    componentItem->setTextAlignment(1, Qt::AlignRight);

    objects[componentItem] = std::pair<core::objectmodel::Base*, Q3ListViewItem*>(base, listItem);

    for(sofa::core::objectmodel::Base::VecData::const_iterator it = fields.begin(); it != fields.end(); ++it)
    {
        core::objectmodel::BaseData *data = *it;

        // ignore unnamed data
        if(data->getName().empty())
            continue;

        // for each data of the current object we determine where it belongs
        QString group = data->getGroup();

        // use the default group if data does not belong to any group
        if(group.isEmpty())
            group = defaultGroup;

        // finally, add the data
        addData(component, group, data);
    }
}

void QDisplayPropertyWidget::addGroup(const QString& component, const QString& group)
{
    // return now if the component does not exist
    QTreeWidgetItem *componentItem = NULL;
    componentItem = findComponent(component);
    if(!componentItem)
        return;

    // return now if the group component already exist
    QTreeWidgetItem *groupItem = NULL;
    groupItem = findGroup(component, group);
    if(groupItem)
        return;

    // assign the default label if group is an empty string
    QString groupLabel = group;
    if(group.isEmpty())
        groupLabel = defaultGroup;

    // finally, add the group
    groupItem = new QTreeWidgetItem(componentItem);
    QFont *font = new QFont();
    font->setBold(true);
    groupItem->setFont(0, *font);
    groupItem->setText(0, groupLabel);
    groupItem->setExpanded(true);
    QBrush *backgroundBrush = new QBrush(QColor(160, 160, 160));
    QBrush *foregroundBrush = new QBrush(QColor(255, 255, 255));
    groupItem->setBackground(0, *backgroundBrush);
    groupItem->setForeground(0, *foregroundBrush);
    groupItem->setBackground(1, *backgroundBrush);
    groupItem->setForeground(1, *foregroundBrush);

    /*if(groupLabel == defaultGroup)
    {
    	//groupItem->setChildIndicatorPolicy(QTreeWidgetItem::DontShowIndicator);
    	//groupItem->setExpanded(true);
    	groupItem->setSizeHint(0, QSize(0,0));
    	groupItem->setSizeHint(1, QSize(0,0));
    }*/
}

void QDisplayPropertyWidget::addData(const QString& component, const QString& group, sofa::core::objectmodel::BaseData *data)
{
    if(!data || !data->isDisplayed())
        return;

    addGroup(component, group);
    QTreeWidgetItem *groupItem = NULL;
    groupItem = findGroup(component, group);

    if(!groupItem)
        return;

    QTreeWidgetItem *dataItem = new QTreeWidgetItem(groupItem);
    QBrush *brush = NULL;
    if(groupItem->childCount() % 2 == 0)
        brush = new QBrush(QColor(255, 255, 191));
    else
        brush = new QBrush(QColor(255, 255, 222));
    dataItem->setBackground(0, *brush);
    dataItem->setBackground(1, *brush);

    data->setDisplayed(true);

    QDisplayTreeItemWidget *widget = new QDisplayTreeItemWidget(this, dataItem);
    QHBoxLayout *layout = new QHBoxLayout(widget);

    QString name = data->getName().c_str();
    dataItem->setText(0, name);
    ModifyObjectFlags modifyObjectFlags = ModifyObjectFlags();
    modifyObjectFlags.setFlagsForModeler();
    modifyObjectFlags.PROPERTY_WIDGET_FLAG = true;
    QDisplayDataWidget *displayDataWidget = new QDisplayDataWidget(widget, data, modifyObjectFlags);
    layout->addWidget(displayDataWidget);

    connect(displayDataWidget, SIGNAL(DataOwnerDirty(bool)), this, SLOT( updateListViewItem() ) );
    connect(displayDataWidget, SIGNAL(WidgetDirty(bool)), widget, SLOT(updateDirtyWidget()));
    connect(displayDataWidget, SIGNAL(WidgetDirty(bool)), this, SLOT(updateDirtyWidget()));

    widget->setContentsMargins(0, 0, 0, 0);
    widget->layout()->setContentsMargins(0, 0, 0, 0);
    widget->layout()->setSpacing(0);
    setItemWidget(dataItem, 1, widget);

    dataItem->setExpanded(true);
}

void QDisplayPropertyWidget::clear()
{
    QTreeWidgetItem *item = NULL;
    QPushButton* pin = NULL;
    for(unsigned int i = 0; (item = topLevelItem(i));)
    {
        pin = static_cast<QPushButton*>(itemWidget(item, 1));
        if(pin && !pin->isChecked())
        {
            objects.erase(objects.find(item));
            takeTopLevelItem(i);
        }
        else
            ++i;
    }
}

void QDisplayPropertyWidget::clearAll()
{
    QTreeWidget::clear();
}

QTreeWidgetItem* QDisplayPropertyWidget::findComponent(const QString& component) const
{
    QTreeWidgetItem *componentItem = NULL;
    for(unsigned int i = 0; (componentItem = topLevelItem(i)); ++i)
        if(componentItem->text(0) == component)
            break;

    return componentItem;
}

QTreeWidgetItem* QDisplayPropertyWidget::findGroup(const QString& component, const QString& group) const
{
    QTreeWidgetItem *componentItem = NULL;
    componentItem = findComponent(component);
    if(!componentItem)
        return NULL;

    QTreeWidgetItem *groupItem = NULL;
    for(unsigned int i = 0; (groupItem = componentItem->child(i)); ++i)
        if(groupItem->text(0) == group)
            break;

    return groupItem;
}

/*void QDisplayPropertyWidget::dragEnterEvent(QDragEnterEvent *event)
{
	QModelIndex index = indexAt(event->pos());
	if(!index.isValid() && !index.parent().isValid())
		return;

	std::cout << index.row() << " - " << index.column() << std::endl;
	QTreeWidgetItem* source = itemFromIndex(index);
	if(source->checkState(0) == Qt::Unchecked)
		return;

	QTreeWidget::dragEnterEvent(event);
}

void QDisplayPropertyWidget::dropEvent(QDropEvent *event)
{
	QModelIndex index = indexAt(event->pos());
	if(!index.isValid() && !index.parent().isValid())
		return;

	std::cout << index.row() << " - " << index.column() << std::endl;
	QTreeWidgetItem* target = itemFromIndex(index);
	if(target->checkState(0) == Qt::Unchecked)
		return;

	QTreeWidget::dropEvent(event);
}*/

/*void QDisplayPropertyWidget::dragMoveEvent(QDragMoveEvent *event)
{
	QModelIndex index = indexAt(event->pos());
	QTreeWidgetItem* source = itemFromIndex(index);
	if(source->checkState(0) == Qt::Checked)
		event->accept();
}

void QDisplayPropertyWidget::dragLeaveEvent(QDragLeaveEvent *event)
{
//	QModelIndex index = indexAt(event->pos());
// 	QTreeWidgetItem* source = itemFromIndex(index);
// 	if(source->checkState(0) == Qt::Checked)
// 		event->accept();
}

Qt::DropActions QDisplayPropertyWidget::supportedDropActions() const
{
	return Qt::CopyAction | Qt::MoveAction;
}*/

} // namespace qt

} // namespace gui

} // namespace sofa
