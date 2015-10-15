/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
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

namespace sofa
{

namespace gui
{

namespace qt
{


LinkComponent::LinkComponent(GraphModeler* mg, const std::vector<Q3ListViewItem*>& items, Q3ListViewItem* sel) :
    listView(NULL),
    mainGraph(mg),
    items2components(),
    selectedComponent(sel),
    loaderNum(0)
{
    setCaption(QString("Link Component"));

    if(!selectedComponent)
        return;

    //Creation of the GUI
    QVBoxLayout *globalLayout = new QVBoxLayout(this);

    //***********************************************************************************
    //Selection of the Loader Name
    QWidget *loaderNameWidget = new QWidget(this);
    QVBoxLayout *loaderNameLayout = new QVBoxLayout(loaderNameWidget);

    listView = new Q3ListView(loaderNameWidget);
    listView->setAcceptDrops(false);
    listView->setSorting(-1);
    listView->header()->hide();
    listView->setSelectionMode(Q3ListView::Single);
    listView->addColumn("");

    QString text;
    Q3ListViewItem *item = NULL;
    Q3ListViewItem *childItem = NULL;
    Q3ListViewItem *parentItem = NULL;
    for(std::vector<Q3ListViewItem*>::const_reverse_iterator it = items.rbegin(); it != items.rend();)
    {
        Q3ListViewItem const * const & cur = *it++;
        Q3ListViewItem const * next = NULL;
        if(it != items.rend())
            next = *it;

        if(!item)
        {
            item = new Q3ListViewItem(listView, cur->text(0));
            item->setPixmap(0, *cur->pixmap(0));
            item->setOpen(true);
            item->setSelectable(false);
        }

        childItem = NULL;
        for(Q3ListViewItem* curChild = cur->firstChild(); curChild != NULL; curChild = curChild->nextSibling())
        {
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

            Q3ListViewItem* previousItem = childItem;
            childItem = new Q3ListViewItem(item, curChild->text(0));
            childItem->setPixmap(0, *curChild->pixmap(0));
            childItem->setOpen(true);
            if(isLoader)
            {
                items2components[childItem] = dynamic_cast<sofa::core::loader::BaseLoader*>(mainGraph->getObject(curChild));
                ++loaderNum;
            }
            else
                childItem->setSelectable(false);

            if(previousItem)
                childItem->moveItem(previousItem);

            if(curChild == next)
                parentItem = childItem;

            if(curChild == sel)
            {
                childItem->setEnabled(false);
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
    connect(listView, SIGNAL(returnPressed(Q3ListViewItem*)), this, SLOT(applyLinkComponent()));
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
    Q3ListViewItem* selectedLoader = listView->selectedItem();
    sofa::core::loader::BaseLoader* loader = items2components[selectedLoader];

    if(!selectedLoader || !loader)
    {
        const std::string message="You did not select any loader, the component cannot be linked";
        emit displayMessage(message);
        return;
    }

    int depthDiff = selectedComponent->depth() - selectedLoader->depth();

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

