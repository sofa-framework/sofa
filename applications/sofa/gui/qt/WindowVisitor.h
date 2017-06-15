/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_WINDOWVISITOR_H
#define SOFA_WINDOWVISITOR_H

#include <ui_VisitorGUI.h>
#include "PieWidget.h"
#include "QVisitorControlPanel.h"

#include <QTreeWidgetItem>
#include <QDrag>
#include <QPixmap>
#include <QTableWidget>
#include <QComboBox>

#include <iostream>

namespace sofa
{

namespace gui
{

namespace qt
{

class WindowVisitor: public QWidget, public Ui_VisitorGUI
{
    Q_OBJECT
public:
    enum componentType {NODE, COMMENT, COMPONENT, VECTOR, OTHER};
    WindowVisitor();

    void collapseNode(QTreeWidgetItem* item);
    void expandNode(QTreeWidgetItem* item);

    void setCharts(std::vector< dataTime >&latestC, std::vector< dataTime >&maxTC, std::vector< dataTime >&totalC,
            std::vector< dataTime >&latestV, std::vector< dataTime >&maxTV, std::vector< dataTime >&totalV);


public slots:
    void setCurrentCharts(int);

    void rightClick( const QPoint& point);

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
    bool setFocusOn(QTreeWidgetItem *item, QString text);

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
