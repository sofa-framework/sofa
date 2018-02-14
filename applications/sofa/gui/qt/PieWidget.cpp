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
#include "PieWidget.h"
#include <iostream>

#include <QGridLayout>
#include <QStringList>
#include <QHeaderView>
#include <QSplitter>

namespace sofa
{

namespace gui
{

namespace qt
{

std::vector< defaulttype::Vec<3,int> > PieWidget::colorArray;

defaulttype::Vec<3,int> PieWidget::getColor(int i)
{
    defaulttype::Vec<3,int> res=PieWidget::colorArray[i%PieWidget::colorArray.size()];
    float factor=1.0/(1.0+(0.3*(i/PieWidget::colorArray.size())));
    res[0] = (int)(res[0]*factor);
    res[1] = (int)(res[1]*factor);
    res[2] = (int)(res[2]*factor);
    return res;
}

PieWidget::PieWidget(QWidget *parent): QWidget(parent)
{
    if (PieWidget::colorArray.empty())
    {
        colorArray.push_back(  defaulttype::Vec<3,int>(250,125,70) );
        colorArray.push_back(  defaulttype::Vec<3,int>(120,220,110) );
        colorArray.push_back(  defaulttype::Vec<3,int>(215,90,215) );
        colorArray.push_back(  defaulttype::Vec<3,int>(255,210,40) );
        colorArray.push_back(  defaulttype::Vec<3,int>(75,210,210) );
    }
}
void PieWidget::paintEvent( QPaintEvent* )
{
    sizePie = (int)(std::min(this->width(),this->height())*0.95);
    if (data.empty()) return;

    QPainter p( this );

    int initDraw=0;
    int startAngle=0;

    p.setBrush(Qt::SolidPattern);

    for (unsigned int i=0; i<data.size() && i<selection; ++i)
    {
        defaulttype::Vec<3,int> c=PieWidget::getColor(i);
        QColor color(c[0],c[1],c[2]);
        p.setBrush(color);
        int spanAngle=(int)(16*360*data[i].time/totalTime);
        p.drawPie(initDraw,initDraw,sizePie,sizePie,startAngle,spanAngle);
        startAngle+= spanAngle;
    }
}

void PieWidget::setChart( std::vector< dataTime >& value, unsigned int s)
{
    data=value;
    selection=s;
    totalTime=0;
    for (unsigned int i=0; i<value.size() && i<selection; ++i)
    {
        totalTime +=data[i].time;
    }
}

void PieWidget::clear()
{
    data.clear();
    repaint();
}

ChartsWidget::ChartsWidget(const std::string &name, QWidget *parent): QWidget(parent)
{
    QSplitter *splitter=new QSplitter(this);
    splitter->setOrientation(Qt::Horizontal);
    QGridLayout *grid = new QGridLayout(this);
    pie = new PieWidget(splitter);

    table = new QTableWidget(0,3,splitter);

#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
    table->horizontalHeader()->setResizeMode(0,QHeaderView::Fixed);
    table->horizontalHeader()->setResizeMode(1,QHeaderView::ResizeToContents);
    table->horizontalHeader()->setResizeMode(2,QHeaderView::ResizeToContents);
#else
    table->horizontalHeader()->setSectionResizeMode(0,QHeaderView::Fixed);
    table->horizontalHeader()->setSectionResizeMode(1,QHeaderView::ResizeToContents);
    table->horizontalHeader()->setSectionResizeMode(2,QHeaderView::ResizeToContents);
#endif // QT_VERSION < QT_VERSION_CHECK(5, 0, 0)

    table->horizontalHeader()->resizeSection(0,30);

    QStringList list; list<<"Id" << name.c_str() << "Time";
    table->setHorizontalHeaderLabels(list);

    grid->addWidget(splitter,0,0);

}


void ChartsWidget::clear()
{
    int rows=table->rowCount();

    for (int i=0; i<rows; ++i) table->removeRow(0);
    pie->clear();
}

void ChartsWidget::setChart( std::vector< dataTime >& value, unsigned int s)
{
    clear();
    pie->setChart(value,s);
    selection=s;
    for (unsigned int i=0; i<value.size() && i<selection; ++i)
    {
        table->insertRow(i);

        defaulttype::Vec<3,int> c=PieWidget::getColor(i);
        QColor color(c[0],c[1],c[2]);

        QString text(value[i].name.c_str());
        QString time= QString::number(value[i].time);
        time += QString(" ms");
        if (!value[i].type.empty())
        {
            text+="(";
            text+= QString(value[i].type.c_str());
            text+=")";
        }

        QTableWidgetItem *itemColor = new QTableWidgetItem();
        itemColor->setBackgroundColor(color);
        QTableWidgetItem *item = new QTableWidgetItem();
        QTableWidgetItem *itemTime = new QTableWidgetItem();
        table->setItem(i,0, itemColor);
        item->setText(text);
        table->setItem(i,1, item);
        itemTime->setText(time);
        table->setItem(i,2, itemTime);
        table->resizeColumnToContents(1);
        itemColor->setFlags(0);
        item->setFlags(0);
        itemTime->setFlags(0);

    }
    pie->repaint();

}

}
}
}
