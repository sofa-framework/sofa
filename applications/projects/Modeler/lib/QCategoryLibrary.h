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
#ifndef SOFA_QCATEGORYLIBRARY_H
#define SOFA_QCATEGORYLIBRARY_H

#include <sofa/core/CategoryLibrary.h>

#ifdef SOFA_QT4
#include <Q3Header>
#include <QGridLayout>
#else
#include <qheader.h>
#include <qlayout.h>
#endif

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
class QCategoryLibrary : virtual public QWidget, public CategoryLibrary
{

    Q_OBJECT
public:
    typedef QGridLayout CategoryLayout;
public:
    QCategoryLibrary(QWidget *parent, const std::string &categoryName, unsigned int numCom);
    ~QCategoryLibrary();

    ComponentLibrary *addComponent(const std::string &componentName, ClassEntry* entry, const std::vector< std::string > &exampleFiles);
    void endConstruction();

    void setDisplayed(bool b);

    QWidget *getQWidget() { return this;};
protected:
    ComponentLibrary *createComponent(const std::string &componentName, ClassEntry* entry, const std::vector< std::string > &exampleFiles);

    CategoryLayout *layout;

public slots:
    void componentDraggedReception( std::string description, std::string templateName, ClassEntry* componentEntry);

signals:
    void componentDragged( std::string description, std::string categoryName, std::string templateName, ClassEntry* entry);
};



}
}
}

#endif
