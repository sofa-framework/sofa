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
#ifndef DISPLAYFLAGWIDGET_H
#define DISPLAYFLAGWIDGET_H

#include <iostream>
#include <map>
#include <vector>

#ifdef SOFA_QT4
#include <Q3ListView>
#include <Q3CheckListItem>
#include <Q3ListViewItem>
#include <Q3Header>
#include <QMouseEvent>
#include <Q3Frame>
#else
#include <qlistview.h>
#include <qheader.h>
typedef QListView Q3ListView;
typedef QCheckListItem Q3CheckListItem;
typedef QListViewItem Q3ListViewItem;
#endif
namespace sofa
{

namespace gui
{

namespace qt
{

class DisplayFlagWidget : public Q3ListView
{
    Q_OBJECT
public:

    DisplayFlagWidget(QWidget* parent, const char* name= 0, Qt::WFlags f= 0 );

    bool getFlag(int idx) {return itemShowFlag[idx]->isOn();}
    void setFlag(int idx, bool value) {itemShowFlag[idx]->setOn(value);}
signals:
    void change(int,bool);
    void clicked();

protected:
    virtual void contentsMousePressEvent ( QMouseEvent * e );

    void findChildren(Q3CheckListItem *, std::vector<Q3CheckListItem* > &children);


    Q3CheckListItem* itemShowFlag[10];
    std::map<  Q3CheckListItem*, int > mapFlag;
};

} // namespace qt

} // namespace gui

} // namespace sofa

#endif
