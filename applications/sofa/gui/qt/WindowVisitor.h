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
#ifndef SOFA_WINDOWVISITOR_H
#define SOFA_WINDOWVISITOR_H

#include "VisitorGUI.h"

#ifdef SOFA_QT4
#include <Q3ListViewItem>
#include <Q3TextDrag>
#include <QPixmap>
#else
#include <qlistview.h>
#include <qdragobject.h>
#include <qpixmap.h>
#endif


#include <iostream>

namespace sofa
{

namespace gui
{

namespace qt
{

#ifndef SOFA_QT4
typedef QListViewItem Q3ListViewItem;
#endif

class WindowVisitor: public VisitorGUI
{
    Q_OBJECT
public:
    enum typeComponent {NODE, COMMENT, COMPONENT, OTHER};
    WindowVisitor();

    void collapseNode(Q3ListViewItem* item);
    void expandNode(Q3ListViewItem* item);
public slots:
#ifdef SOFA_QT4
    void rightClick(Q3ListViewItem *, const QPoint &, int );
#else
    void rightClick(QListViewItem *, const QPoint &, int );
#endif
    void collapseNode();
    void expandNode();
    static QPixmap* getPixmap(typeComponent t) {return icons[t];}
    void closeEvent( QCloseEvent* )
    {
        emit(WindowVisitorClosed(false));
        hide();
    }
public slots:
    void clearGraph() {graphView->clear();}
signals:
    void WindowVisitorClosed(bool);
protected:
    static QPixmap *icons[OTHER+1];
};
}
}
}

#endif
