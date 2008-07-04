/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This program is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU General Public License as published by the Free   *
* Software Foundation; either version 2 of the License, or (at your option)    *
* any later version.                                                           *
*                                                                              *
* This program is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for     *
* more details.                                                                *
*                                                                              *
* You should have received a copy of the GNU General Public License along with *
* this program; if not, write to the Free Software Foundation, Inc., 51        *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                    *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/

#ifndef GRAPHLISTENERQLISTVIEW_H
#define GRAPHLISTENERQLISTVIEW_H




#ifdef SOFA_QT4
#include <Q3ListViewItem>
#include <Q3CheckListItem>
#include <Q3ListView>
#include <QWidget>
#include <Q3PopupMenu>

#else
#include <qlistview.h>
#include <qwidget.h>
#include <qpopupmenu.h>


#include <qlabel.h>
#include <qcheckbox.h>
#include <qpushbutton.h>
#include <qimage.h>
#include <qspinbox.h>

#endif


#include <sofa/simulation/tree/Simulation.h>
#include <sofa/simulation/tree/MutationListener.h>

#include "WFloatLineEdit.h"


namespace sofa
{

namespace gui
{

namespace qt
{
using sofa::simulation::tree::GNode;
using sofa::simulation::tree::Simulation;
using sofa::simulation::tree::MutationListener;

#ifdef SOFA_QT4
typedef Q3ListView QListView;
typedef Q3PopupMenu QPopupMenu;
#else
typedef QListViewItem Q3ListViewItem;
typedef QCheckListItem Q3CheckListItem;
typedef QListView Q3ListView;
typedef QPopupMenu Q3PopupMenu;
#endif

QPixmap* getPixmap(core::objectmodel::Base* obj);

class GraphListenerQListView : public MutationListener
{
public:
    Q3ListView* widget;
    bool frozen;
    std::map<core::objectmodel::Base*, Q3ListViewItem* > items;
    GraphListenerQListView(Q3ListView* w)
        : widget(w), frozen(false)
    {
    }


    /*****************************************************************************************************************/
    Q3ListViewItem* createItem(Q3ListViewItem* parent);
    void addChild(GNode* parent, GNode* child);
    void removeChild(GNode* parent, GNode* child);
    void moveChild(GNode* previous, GNode* parent, GNode* child);
    void addObject(GNode* parent, core::objectmodel::BaseObject* object);
    void removeObject(GNode* /*parent*/, core::objectmodel::BaseObject* object);
    void moveObject(GNode* previous, GNode* parent, core::objectmodel::BaseObject* object);
    void freeze(GNode* groot);
    void unfreeze(GNode* groot);
};

}
}
}
#endif
