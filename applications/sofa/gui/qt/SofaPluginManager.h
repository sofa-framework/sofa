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
#ifndef SOFA_PLUGINMANAGER_H
#define SOFA_PLUGINMANAGER_H

#include "ui_PluginManager.h"
#include "SofaGUIQt.h"

#include <Q3ListViewItem>

#include <set>

namespace sofa
{
namespace gui
{
namespace qt
{

#ifndef SOFA_QT4
typedef QListViewItem Q3ListViewItem;
#endif

class SofaPluginManager: public QDialog, public Ui_PluginManager
{
    Q_OBJECT
public:

    SofaPluginManager();

signals:

    void libraryAdded();
    void libraryRemoved();

public slots:

    void addLibrary();
    void removeLibrary();

    void updateComponentList(Q3ListViewItem*);
    void updateDescription(Q3ListViewItem*);

private :
    void updatePluginsListView();
    void initPluginListView();

};


}
}
}

#endif
