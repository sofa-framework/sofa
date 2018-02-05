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
#ifndef SOFA_PLUGINMANAGER_H
#define SOFA_PLUGINMANAGER_H

#include <ui_PluginManager.h>
#include "SofaGUIQt.h"
#include <QTreeWidgetItem>

#include <set>


namespace sofa
{
namespace gui
{
namespace qt
{

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

    void updateComponentList();
    void updateDescription();
public:
    void updatePluginsListView();
private:
    void savePluginsToIniFile();
    void loadPluginsFromIniFile();
};


}
}
}

#endif
