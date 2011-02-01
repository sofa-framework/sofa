/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_PLUGINMANAGER_H
#define SOFA_PLUGINMANAGER_H

#include "PluginManager.h"
#include "SofaGUIQt.h"
#ifdef SOFA_QT4
#include <Q3ListViewItem>
#else
#include <qlistview.h>
#endif
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

class SofaPluginManager: public PluginManager
{
    Q_OBJECT
public:

    SofaPluginManager();
    void initPluginList();

    static SofaPluginManager* getInstance()
    {
        static SofaPluginManager instance;
        return &instance;
    }
    template <typename OutIterator >
    void getPluginList( OutIterator out )
    {
        std::set<std::string>::const_iterator it;
        for ( it = pluginList.begin(); it != pluginList.end(); ++it)
        {
            *out = *it;
            out++;
        }
    }

signals:
    void libraryAdded();
    void libraryRemoved();

public slots:

    void addLibrary();
    void removeLibrary();
#ifdef SOFA_QT4
    void updateComponentList(Q3ListViewItem*);
    void updateDescription(Q3ListViewItem*);
#else
    void updateComponentList(QListViewItem*);
    void updateDescription(QListViewItem*)
#endif

private :
    void transferPluginsToNewPath();
    std::set< std::string > pluginList;
};


}
}
}

#endif
