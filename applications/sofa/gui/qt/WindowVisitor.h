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
#include "PieWidget.h"
#include "QVisitorControlPanel.h"

#ifdef SOFA_QT4
#include <Q3ListViewItem>
#include <Q3TextDrag>
#include <QPixmap>
#include <QTableWidget>
#include <QComboBox>
#else
#include <qlistview.h>
#include <qdragobject.h>
#include <qpixmap.h>
#include <qtable.h>
#include <qcombobox.h>
#endif


#include <iostream>

namespace sofa
{

namespace gui
{

namespace qt
{

#ifndef SOFA_QT4
typedef QListView Q3ListView;
typedef QListViewItem Q3ListViewItem;
#endif

class WindowVisitor: public VisitorGUI
{
    Q_OBJECT
public:
    enum componentType {NODE, COMMENT, COMPONENT, VECTOR, OTHER};
    WindowVisitor();

    void collapseNode(Q3ListViewItem* item);
    void expandNode(Q3ListViewItem* item);

    void setCharts(std::vector< dataTime >&latestC, std::vector< dataTime >&maxTC, std::vector< dataTime >&totalC,
            std::vector< dataTime >&latestV, std::vector< dataTime >&maxTV, std::vector< dataTime >&totalV);


public slots:
    void setCurrentCharts(int);

#ifdef SOFA_QT4
    void rightClick(Q3ListViewItem *, const QPoint &, int );
#else
    void rightClick(QListViewItem *, const QPoint &, int );
#endif
    void collapseNode();
    void expandNode();

    void focusOn(QString focus);

    static componentType getComponentType(std::string name)
    {
        if (name == "Node")
            return NODE;
        else if (name == "Component")
            return COMPONENT;
        else if (name == "Vector")
            return VECTOR;
        else
            return OTHER;
    }

    static QPixmap* getPixmap(componentType t) {return icons[t];}

    void closeEvent( QCloseEvent* )
    {
        emit(WindowVisitorClosed(false));
        hide();
        clearGraph();
    }

    void clearGraph()
    {
        graphView->clear();
        chartsComponent->clear();
        chartsVisitor->clear();

        componentsTime.clear();
        componentsTimeMax.clear();
        componentsTimeTotal.clear();


        visitorsTime.clear();
        visitorsTimeMax.clear();
        visitorsTimeTotal.clear();
    }

signals:
    void WindowVisitorClosed(bool);
public:

    QWidget *statsWidget;
protected:
    bool setFocusOn(Q3ListViewItem *item, QString text);

    static QPixmap *icons[OTHER+1];

    std::vector< dataTime > componentsTime;
    std::vector< dataTime > visitorsTime;

    std::vector< dataTime > componentsTimeTotal;
    std::vector< dataTime > visitorsTimeTotal;

    std::vector< dataTime > componentsTimeMax;
    std::vector< dataTime > visitorsTimeMax;

    QVisitorControlPanel *controlPanel;
    ChartsWidget *chartsComponent;
    ChartsWidget *chartsVisitor;
    QComboBox *typeOfCharts;
};
}
}
}

#endif
