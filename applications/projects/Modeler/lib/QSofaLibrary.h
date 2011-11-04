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
#ifndef SOFA_QSOFALIBRARY_H
#define SOFA_QSOFALIBRARY_H

#include <sofa/core/SofaLibrary.h>
#include "FilterLibrary.h"

#ifdef SOFA_QT4
#include <Q3Header>
#include <QToolBox>
#else
#include <qheader.h>
#include <qtoolbox.h>
#endif



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
class QSofaLibrary : virtual public QToolBox, public SofaLibrary
{
    Q_OBJECT
public:
    typedef QToolBox LibraryContainer;
public:
    QSofaLibrary(QWidget *parent);

    void build(const std::vector< std::string >& examples=std::vector< std::string >());
    void filter(const FilterQuery &f);
    void clear()
    {
        const int n=this->count();
        for ( int i=n; i>=0; --i)
        {
            removeItem(item(i));
        }
        SofaLibrary::clear();
    }

    LibraryContainer* getContainer() {return toolbox;};

    QWidget *getQWidget() {return this;};
protected:
    CategoryLibrary *createCategory(const std::string &category, unsigned int numComponent);
    void addCategory(CategoryLibrary *category);


    LibraryContainer *toolbox;

public slots:
    void componentDraggedReception( std::string description, std::string categoryName, std::string templateName, ClassEntry* componentEntry);

signals:
    void componentDragged( std::string description, std::string categoryName, std::string templateName, ClassEntry* entry);
};

}
}
}

#endif
