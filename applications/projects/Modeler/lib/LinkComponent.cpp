/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "LinkComponent.h"

#include <sofa/helper/set.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QCompleter>
#include <QPushButton>
#include <QSpacerItem>
#include <QGroupBox>
#include <QHeaderView>

namespace sofa
{

namespace gui
{

namespace qt
{


LinkComponent::LinkComponent(GraphModeler* mg, const std::vector<QTreeWidgetItem*>& items, QTreeWidgetItem* sel) :
    listView(NULL),
    mainGraph(mg),
    items2components(),
    selectedComponent(sel),
    loaderNum(0)
{
    setWindowTitle(QString("Link Component"));
    //setCaption(QString("Link Component"));

    if(!selectedComponent)
        return;

    //Creation of the GUI
    QVBoxLayout *globalLayout = new QVBoxLayout(this);

    //***********************************************************************************
    //Selection of the Loader Name
    QWidget *loaderNameWidget = new QWidget(this);
    QVBoxLayout *loaderNameLayout = new QVBoxLayout(loaderNameWidget);

    listView = new QTreeWidget(loaderNameWidget);
    listView->setAcceptDrops(false);
    listView->setSortingEnabled(false);
    listView->header()->hide();
    listView->setSelectionMode(QAbstractItemView::SingleSelection);
    //listView->addColumn("");

//    QString text;
    QTreeWidgetItem *item = NULL;
    QTreeWidgetItem *childItem = NULL;
    QTreeWidgetItem *parentItem = NULL;
    for(std::vector<QTreeWidgetItem*>::const_reverse_iterator it = items.rbegin(); it != items.rend();)
    {
        QTreeWidgetItem const * const & cur = *it++;
        QTreeWidgetItem const * next = NULL;
        if(it != items.rend())
            next = *it;

        if(!item)
        {
            item = new QTreeWidgetItem(listView);
            item->setText(0, cur->text(0));//??
            item->setIcon(0, cur->icon(0));
            item->setFlags(Qt::ItemIsEnabled);
            item->setSelected(false);
        }

        childItem = NULL;

        for(int i=0 ; i<cur->childCount();i++)
        {
            QTreeWidgetItem* curChild = cur->child(i);

            // is the object a Loader
            bool isLoader = false;
            if(dynamic_cast<sofa::core::loader::BaseLoader*>(mainGraph->getObject(curChild)))
                isLoader = true;

            // is the object a valid Node
            bool isValidNode = false;
            if(dynamic_cast<Node*>(mainGraph->getComponent(curChild)) && curChild == next)
                isValidNode = true;

            // is the object the selected object
            bool isSelectedObject = false;
            if(curChild == sel)
                isSelectedObject = true;

            // if the component is not a valid component
            if(	!isLoader && !isValidNode && !isSelectedObject)
                continue;

            QTreeWidgetItem* previousItem = childItem;
            childItem = new QTreeWidgetItem(item);
            childItem->setText(0, curChild->text(0));
            childItem->setIcon(0, curChild->icon(0));
            childItem->setExpanded(true);
            if(isLoader)
            {
                items2components[childItem] = dynamic_cast<sofa::core::loader::BaseLoader*>(mainGraph->getObject(curChild));
                ++loaderNum;
            }
            else
                childItem->setFlags(Qt::ItemIsEnabled);

            if(previousItem)
            {
                QTreeWidgetItem* parent = curChild->parent();
                int index = parent->indexOfChild(childItem);
                QTreeWidgetItem* tmpItem = parent->takeChild(index);
                parent->insertChild(parent->indexOfChild(previousItem), tmpItem);
            }

            if(curChild == next)
                parentItem = childItem;

            if(curChild == sel)
            {
                childItem->setFlags(Qt::ItemIsSelectable);
                break;
            }
        }
        item = parentItem;
    }

    if(loaderNum == 0)
        return;

    loaderNameLayout->addWidget(new QLabel(QString("Select the Loader to link with"),loaderNameWidget));
    loaderNameLayout->addWidget(listView);

    //***********************************************************************************
    //Button Panel
    QWidget *buttonWidget = new QWidget(this);
    QHBoxLayout *buttonLayout = new QHBoxLayout(buttonWidget);

    QPushButton *linkButton = new QPushButton(QString("Link"), buttonWidget);
    QPushButton *cancelButton = new QPushButton(QString("Cancel"), buttonWidget);

    linkButton->setAutoDefault(true);

    buttonLayout->addItem(new QSpacerItem(1,1,QSizePolicy::Expanding,QSizePolicy::Minimum));
    buttonLayout->addWidget(linkButton);
    buttonLayout->addWidget(cancelButton);

    //***********************************************************************************
    //Control Panel

    globalLayout->addWidget(loaderNameWidget);
    globalLayout->addWidget(buttonWidget);


    //***********************************************************************************
    //Do the connections
    connect(listView, SIGNAL(returnPressed(QTreeWidgetItem*)), this, SLOT(applyLinkComponent()));
    connect(linkButton, SIGNAL(clicked()), this, SLOT(applyLinkComponent()));
    connect(cancelButton, SIGNAL(clicked()), this, SLOT(close()));
}

LinkComponent::~LinkComponent()
{
    delete listView;
}

unsigned int LinkComponent::loaderNumber() const
{
    return loaderNum;
}

void LinkComponent::applyLinkComponent()
{
    QList<QTreeWidgetItem*> selectedLoader = listView->selectedItems();

    if(selectedLoader.count() < 1 || !items2components[selectedLoader[0]])
    {
        const std::string message="You did not select any loader, the component cannot be linked";
        emit displayMessage(message);
        return;
    }

    sofa::core::loader::BaseLoader* loader = items2components[selectedLoader[0]];

    //compute depth for the 2 elements as depth() disappeared starting from Qt4
    int depthComponent=0;
    int depthLoader=0;
    QTreeWidgetItem* currentItem = selectedComponent;
    while(currentItem->parent())
    {
        currentItem = currentItem->parent();
        depthComponent++;
    }
    currentItem = selectedLoader[0];
    while(currentItem->parent())
    {
        currentItem = currentItem->parent();
        depthLoader++;
    }

    int depthDiff = depthComponent - depthLoader;

    std::string loaderPath = "@";
    for(int i = 0; i < depthDiff; ++i)
        loaderPath += "../";

    loaderPath += loader->getName();

    // link the component with the loader with setSrc
    mainGraph->getObject(selectedComponent)->setSrc(loaderPath, loader, NULL);

    // close the dialog box
    close();
}

}
}
}

