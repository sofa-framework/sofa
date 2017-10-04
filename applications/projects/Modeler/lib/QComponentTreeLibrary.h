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
#ifndef SOFA_QCOMPONENTTREELIBRARY_H
#define SOFA_QCOMPONENTTREELIBRARY_H

#include <sofa/core/ComponentLibrary.h>

#include <QTreeWidget>
#include <QPushButton>
#include <QComboBox>


namespace sofa
{

namespace gui
{

namespace qt
{

using sofa::core::ComponentLibrary;
typedef sofa::core::ObjectFactory::ClassEntry ClassEntry;

class QComponentTreeLibrary : public QWidget, public ComponentLibrary
{

    Q_OBJECT
public:
    typedef QPushButton ComponentLabel;
    typedef QComboBox   ComponentTemplates;
public:
    QComponentTreeLibrary(QWidget *parent, QTreeWidgetItem* category,const std::string &componentName, const std::string &categoryName, ClassEntry::SPtr entry, const std::vector< std::string > &exampleFiles);
    ~QComponentTreeLibrary();

    void endConstruction();

    void setDisplayed(bool b);

    QWidget *getQWidget() { return this;};
protected:
    //--------------------------------------------
    //Qt Data
    ComponentLabel     *label;
    ComponentTemplates *templates;

    QTreeWidgetItem *componentTree;
    QTreeWidget *tree;

public slots:
    void componentPressed();

signals:
    void componentDragged( std::string description, std::string templateName, ClassEntry::SPtr entry);
};

}
}
}

#endif
