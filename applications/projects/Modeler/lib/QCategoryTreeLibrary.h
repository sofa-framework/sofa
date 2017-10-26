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
#ifndef SOFA_QCATEGORYLIBRARY_H
#define SOFA_QCATEGORYLIBRARY_H

#include <sofa/core/CategoryLibrary.h>

#include <QTreeWidget>
#include <QGridLayout>

namespace sofa
{

namespace gui
{

namespace qt
{


using sofa::core::CategoryLibrary;
using sofa::core::ComponentLibrary;

typedef sofa::core::ObjectFactory::ClassEntry ClassEntry;

//***************************************************************
class QCategoryTreeLibrary : public QWidget, public CategoryLibrary
{

    Q_OBJECT
public:
    typedef QGridLayout CategoryLayout;
public:
    QCategoryTreeLibrary(QWidget *parent, const std::string &categoryName, unsigned int numCom);
    ~QCategoryTreeLibrary();

    ComponentLibrary *addComponent(const std::string &componentName, ClassEntry::SPtr entry, const std::vector< std::string > &exampleFiles);
    void endConstruction();

    void setDisplayed(bool b);

    QTreeWidgetItem *getQWidget() { return categoryTree;};
protected:
    ComponentLibrary *createComponent(const std::string &componentName, ClassEntry::SPtr entry, const std::vector< std::string > &exampleFiles);

    QTreeWidgetItem *categoryTree;
    QTreeWidget *tree;

public slots:
    void componentDraggedReception( std::string description, std::string templateName, ClassEntry::SPtr componentEntry);

signals:
    void componentDragged( std::string description, std::string categoryName, std::string templateName, ClassEntry::SPtr entry);
};



}
}
}

#endif
