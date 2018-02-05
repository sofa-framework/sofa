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
#ifndef SOFA_GUI_QT_TABLEDATAWIDGET_H
#define SOFA_GUI_QT_TABLEDATAWIDGET_H

#include "SimpleDataWidget.h"
#include "StructDataWidget.h"
#include <SofaBaseTopology/TopologyData.h>

#include "QModelViewTableDataContainer.h"


namespace sofa
{

namespace gui
{

namespace qt
{

template<class T, int FLAGS = TABLE_NORMAL>
class TableDataWidget : public SimpleDataWidget<T, table_data_widget_container< T , FLAGS > >
{
public:
    typedef T data_type;
    typedef SimpleDataWidget<T, table_data_widget_container< T , FLAGS > > Inherit;
    typedef sofa::core::objectmodel::Data<T> MyData;
public:
    TableDataWidget(QWidget* parent,const char* name, MyData* d) : Inherit(parent,name,d) {}
    virtual unsigned int sizeWidget() {return 8;}
    virtual unsigned int numColumnWidget() { return 1; }
};


} // namespace qt

} // namespace gui

} // namespace sofa

#endif
