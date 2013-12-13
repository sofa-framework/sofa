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
#include "QDisplayLinkWidget.h"
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

QDisplayPropertyWidget::QDisplayPropertyWidget(const ModifyObjectFlags& modifyFlags, QWidget* parent) : QTreeWidget(parent)
	, objects()
    , pinIcon()
	, modifyObjectFlags(modifyFlags)
{
	modifyObjectFlags.PROPERTY_WIDGET_FLAG = true;

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

	// add data
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
            group = DefaultDataGroup();

        // finally, add the data
        addData(component, group, data);
    }

	bool notImplementedYet = true;
	if(!notImplementedYet)
	{
		// add links
		const sofa::core::objectmodel::Base::VecLink& links = base->getLinks();
		for(sofa::core::objectmodel::Base::VecLink::const_iterator it = links.begin(); it != links.end(); ++it)
		{
			core::objectmodel::BaseLink *link = *it;

			// ignore unnamed link
			if(link->getName().empty())
				continue;

			if(!link->storePath() && 0 == link->getSize())
				continue;

			// use the default link group
			QString group = DefaultLinkGroup();

			// finally, add the data
			addLink(component, group, link);
		}

		// add info
		{
			// use the default info group
			QString group = DefaultInfoGroup();

			setDescription(component, group, base);
		}

		// add console
		/*{
			updateConsole();
			if (outputTab)
			{
				dialogTab->addTab(outputTab,  QString("Logs"));
			}
			if (warningTab)
			{
				dialogTab->addTab(warningTab, QString("Warnings"));
			}
		}*/
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
        groupLabel = DefaultDataGroup();

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

    /*if(groupLabel == DefaultDataGroup())
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

    dataItem->setText(0, data->getName().c_str());
	dataItem->setToolTip(0, data->getHelp());
    QDisplayDataWidget *displayDataWidget = new QDisplayDataWidget(widget, data, modifyObjectFlags);
    layout->addWidget(displayDataWidget);

    connect(displayDataWidget, SIGNAL(DataOwnerDirty(bool)), this, SLOT( updateListViewItem() ) );
    connect(displayDataWidget, SIGNAL(WidgetDirty(bool)), widget, SLOT(updateDirtyWidget()));

    widget->setContentsMargins(0, 0, 0, 0);
	if(widget->layout())
	{
		widget->layout()->setContentsMargins(0, 0, 0, 0);
		widget->layout()->setSpacing(0);
	}
    setItemWidget(dataItem, 1, widget);
	dataItem->setToolTip(1, data->getHelp());

    dataItem->setExpanded(true);
}

void QDisplayPropertyWidget::addLink(const QString& component, const QString& group, sofa::core::objectmodel::BaseLink *link)
{
    if(!link)
        return;

    addGroup(component, group);
    QTreeWidgetItem *groupItem = NULL;
    groupItem = findGroup(component, group);

    if(!groupItem)
        return;

    QTreeWidgetItem *linkItem = new QTreeWidgetItem(groupItem);
    QBrush *brush = NULL;
    if(groupItem->childCount() % 2 == 0)
        brush = new QBrush(QColor(255, 255, 191));
    else
        brush = new QBrush(QColor(255, 255, 222));
    linkItem->setBackground(0, *brush);
    linkItem->setBackground(1, *brush);

    QDisplayTreeItemWidget *widget = new QDisplayTreeItemWidget(this, linkItem);
    QHBoxLayout *layout = new QHBoxLayout(widget);

    linkItem->setText(0, link->getName().c_str());
	linkItem->setToolTip(0, link->getHelp());
    QDisplayLinkWidget *displayLinkWidget = new QDisplayLinkWidget(widget, link, modifyObjectFlags);
    layout->addWidget(displayLinkWidget);

    connect(displayLinkWidget, SIGNAL(LinkOwnerDirty(bool)), this, SLOT( updateListViewItem() ) );
    connect(displayLinkWidget, SIGNAL(WidgetDirty(bool)), widget, SLOT(updateDirtyWidget()));

    widget->setContentsMargins(0, 0, 0, 0);
	if(widget->layout())
	{
		widget->layout()->setContentsMargins(0, 0, 0, 0);
		widget->layout()->setSpacing(0);
	}
    setItemWidget(linkItem, 1, widget);
	linkItem->setToolTip(1, link->getHelp());

    linkItem->setExpanded(true);
}

void QDisplayPropertyWidget::setDescription(const QString& component, const QString& group, sofa::core::objectmodel::Base *base)
{
	if(!base)
        return;

    addGroup(component, group);
    QTreeWidgetItem *groupItem = NULL;
    groupItem = findGroup(component, group);

    if(!groupItem)
        return;

    QTreeWidgetItem *descriptionItem = new QTreeWidgetItem(groupItem);
    QBrush *brush = NULL;
    if(groupItem->childCount() % 2 == 0)
        brush = new QBrush(QColor(255, 255, 191));
    else
        brush = new QBrush(QColor(255, 255, 222));
    descriptionItem->setBackground(0, *brush);
    descriptionItem->setBackground(1, *brush);

	QDataDescriptionWidget* description=new QDataDescriptionWidget(this, base);
	if(description->layout())
	{
		description->layout()->setContentsMargins(0, 0, 0, 0);
		description->layout()->setSpacing(0);
	}
	setItemWidget(descriptionItem, 1, description);

	descriptionItem->setExpanded(true);
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
