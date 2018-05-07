/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#ifndef SOFA_PIEWIDGET_H
#define SOFA_PIEWIDGET_H

#include <QWidget>
#include <QPainter>
#include <QTableWidget>


#include <vector>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace gui
{

namespace qt
{

struct  dataTime
{
    dataTime(double t,
            std::string n,
            std::string ty=std::string(),std::string address=std::string()):time(t), name(n), type(ty), ptr(address) {}
    bool operator== (const dataTime& other)
    {
        if (ptr.empty()) return  name == other.name;
        else return ptr == other.ptr;
    }
    double time;
    std::string name;
    std::string type;
    std::string ptr;
};


class PieWidget: public QWidget
{
public:

    PieWidget(QWidget *parent);

    void paintEvent( QPaintEvent* );

    void setChart( std::vector< dataTime >& value, unsigned int s);
    void clear();
    static defaulttype::Vec<3,int> getColor(int i);
    static std::vector< defaulttype::Vec<3,int> > colorArray;
protected:
    std::vector< dataTime > data;

    unsigned int selection;
    double totalTime;
    int sizePie;
};

class ChartsWidget: public QWidget
{
public:
    ChartsWidget(const std::string &name, QWidget *parent);

    void setChart( std::vector< dataTime >& value, unsigned int s);
    void clear();
protected:

    unsigned int selection;

    PieWidget* pie;
    QTableWidget *table;
};


}
}
}
#endif
