/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include "QDataDescriptionWidget.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseNode.h>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>



namespace sofa
{

namespace gui
{
namespace qt
{
void QDataDescriptionWidget::addRow(QGridLayout* grid, const std::string& title,
                                    const std::string& value, unsigned int row,
                                    unsigned int /*minimumWidth*/)
{
    QLabel* tmplabel;
    grid->addWidget(new QLabel(QString(title.c_str())), row, 0);
    tmplabel = (new QLabel(QString(value.c_str())));
    tmplabel->setMinimumWidth(20);
    grid->addWidget(tmplabel, row, 1);
}

QDataDescriptionWidget::QDataDescriptionWidget(QWidget* parent, core::objectmodel::Base* object)
    :QWidget(parent)
{

    QVBoxLayout* tabLayout = new QVBoxLayout(this);
    tabLayout->setMargin(0);
    tabLayout->setSpacing(1);
    tabLayout->setObjectName("tabInfoLayout");

    //Instance
    {
        QGroupBox *box = new QGroupBox(this);
        tabLayout->addWidget(box);
        QGridLayout* boxLayout = new QGridLayout();
        box->setLayout(boxLayout);

        box->setTitle(QString("Instance"));

        addRow(boxLayout, "Name", object->getName(), 0);
        addRow(boxLayout, "Class", object->getClassName(), 1);

        std::string namespacename = core::objectmodel::BaseClass::decodeNamespaceName(typeid(*object));

        int nextRow = 2;
        if (!namespacename.empty())
        {
            addRow(boxLayout, "Namespace", namespacename, nextRow, 20);
            nextRow++;
        }
        if (!object->getTemplateName().empty())
        {
            addRow(boxLayout, "Template", object->getTemplateName(), nextRow, 20);
            nextRow++;
        }

        core::objectmodel::BaseNode* node = object->toBaseNode(); // Node
        if (node && node->getNbParents()>1) // MultiNode
        {
            addRow(boxLayout, "Path", node->getPathName(), nextRow, 20);
            nextRow++;
         }

        tabLayout->addWidget( box );
    }


    //Class description
    core::ObjectFactory::ClassEntry entry = core::ObjectFactory::getInstance()->getEntry(object->getClassName());
    if (! entry.creatorMap.empty())
    {
        QGroupBox *box = new QGroupBox(this);
        tabLayout->addWidget(box);
        QGridLayout* boxLayout = new QGridLayout();
        box->setLayout(boxLayout);
        box->setTitle(QString("Class"));

        int nextRow = 0;
        if (!entry.description.empty() && entry.description != std::string("TODO"))
        {
            addRow(boxLayout, "Description", entry.description, nextRow, 20);
            nextRow++;
        }
        core::ObjectFactory::CreatorMap::iterator it = entry.creatorMap.find(object->getTemplateName());
        if (it != entry.creatorMap.end() && *it->second->getTarget())
        {
            addRow(boxLayout, "Provided by",it->second->getTarget(), nextRow, 20);
            nextRow++;
        }

        if (!entry.authors.empty() && entry.authors != std::string("TODO"))
        {
            addRow(boxLayout, "Authors", entry.authors, nextRow, 20);
            nextRow++;
        }
        if (!entry.license.empty() && entry.license != std::string("TODO"))
        {
            addRow(boxLayout, "License", entry.license, nextRow, 20);
            nextRow++;
        }
        tabLayout->addWidget( box );
    }

    tabLayout->addStretch();
}




} // qt
} //gui
} //sofa

