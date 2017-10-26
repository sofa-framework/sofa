/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "QComponentTreeLibrary.h"

#include <QToolTip>

namespace sofa
{

namespace gui
{

namespace qt
{
QComponentTreeLibrary::QComponentTreeLibrary(QWidget *parent, QTreeWidgetItem* category, const std::string &componentN, const std::string &categoryN, ClassEntry::SPtr e, const std::vector< std::string > &exampleFiles): QWidget(parent), ComponentLibrary(componentN,categoryN,e, exampleFiles)
{
    setObjectName(componentN.c_str());
    //Create Tree component
    tree=(QTreeWidget*) parent;
    componentTree = new QTreeWidgetItem( category, QTreeWidgetItem::UserType );
    category->addChild(componentTree);

    //Create button and label
    label     = new ComponentLabel( QString(this->getName().c_str()), this);
    label->setFlat(false);
    connect( label, SIGNAL(pressed()), this, SLOT( componentPressed() ));
    templates = new ComponentTemplates(this);

    //Add Documentation tool tip
    std::string tooltipText = entry->description.substr(0, entry->description.size()-1);
    label->setToolTip(QString(tooltipText.c_str()));
}

QComponentTreeLibrary::~QComponentTreeLibrary()
{
    delete label;
    delete templates;
}

void QComponentTreeLibrary::endConstruction()
{
    tree->setItemWidget(componentTree,0,label);
    if (templateName.empty())
    {
        templates->hide();
        return;
    }

    tree->setItemWidget(componentTree,1,templates);
    for (unsigned int i=0; i<templateName.size(); ++i)
    {
        templates->addItem(QString(templateName[i].c_str()));
    }
}

void QComponentTreeLibrary::setDisplayed(bool b)
{
    componentTree->setHidden(!b);
}

//*********************//
// SLOTS               //
//*********************//
void QComponentTreeLibrary::componentPressed()
{
    std::string tName;
    if (!templateName.empty()) tName = templates->currentText().toStdString();

    emit( componentDragged( description, tName, entry));
    label->setDown(false);
}

}
}
}
