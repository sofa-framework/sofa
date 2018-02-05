/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_QSOFATREELIBRARY_H
#define SOFA_QSOFATREELIBRARY_H

#include <sofa/core/SofaLibrary.h>
#include "FilterLibrary.h"

#include <QTreeWidget>



namespace sofa
{

namespace gui
{

namespace qt
{

using sofa::core::SofaLibrary;
using sofa::core::CategoryLibrary;

typedef sofa::core::ObjectFactory::ClassEntry ClassEntry;

//***************************************************************
//Library using QToolBox
class QSofaTreeLibrary : public QTreeWidget, public SofaLibrary
{
    Q_OBJECT
public:
    typedef QTreeWidget LibraryContainer;
public:
    QSofaTreeLibrary(QWidget *parent);


    void build(const std::vector< std::string >& examples=std::vector< std::string >());
    void filter(const FilterQuery &f);
    void clear() {QTreeWidget::clear(); SofaLibrary::clear();}

    LibraryContainer* getContainer() {return toolbox;};

    QWidget *getQWidget() {return this;};
protected:
    CategoryLibrary *createCategory(const std::string &category, unsigned int numComponent);
    void addCategory(CategoryLibrary *category);


    LibraryContainer *toolbox;

public slots:
    void componentDraggedReception( std::string description, std::string categoryName, std::string templateName, ClassEntry::SPtr componentEntry);

signals:
    void componentDragged( std::string description, std::string categoryName, std::string templateName, ClassEntry::SPtr entry);
};

}
}
}

#endif
