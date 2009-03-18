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
#include "PieWidget.h"
#include <iostream>

#ifdef SOFA_QT4
#include <QGridLayout>
#include <QStringList>
#include <QHeaderView>
#include <QSplitter>
#else
#include <qheader.h>
#include <qlayout.h>
#include <qsplitter.h>
#endif

namespace sofa
{

namespace gui
{

namespace qt
{

PieWidget::PieWidget(QWidget *parent): QWidget(parent)
{
}
void PieWidget::paintEvent( QPaintEvent* )
{
    sizePie = std::min(this->width(),this->height())*0.95;
    if (data.empty()) return;

    QPainter p( this );

    int initDraw=0;
    int startAngle=0;

    p.setBrush(Qt::SolidPattern);

    for (unsigned int i=0; i<data.size() && i<selection; ++i)
    {
        QColor color(100*(i%6 == 2)+255*(i%6 == 0 || i%6 == 3 || i%6 == 4)/(1+(0.4*i/6)),
                100*(i%6 == 2)+255*(i%6 == 1 || i%6 == 3 || i%6 == 5)/(1+(0.4*i/6)),
                255*(i%6 == 2 || i%6 == 4 || i%6 == 5)/(1+(0.4*i/6)));

        p.setBrush(color);
        int spanAngle=16*360*data[i].time/totalTime;
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

ChartsWidget::ChartsWidget(QWidget *parent): QWidget(parent)
{
    QSplitter *splitter=new QSplitter(this);
    splitter->setOrientation(Qt::Horizontal);
    QGridLayout *grid = new QGridLayout(this);
    pie = new PieWidget(splitter);
#ifdef SOFA_QT4
    table = new QTableWidget(0,3,splitter);
    table->horizontalHeader()->setResizeMode(0,QHeaderView::Fixed);
    table->horizontalHeader()->resizeSection(0,30);
    table->horizontalHeader()->setResizeMode(1,QHeaderView::ResizeToContents);
    table->horizontalHeader()->setResizeMode(2,QHeaderView::Stretch);
    QStringList list; list<<"Id" << "Name" << "Time";
    table->setHorizontalHeaderLabels(list);
#else
    table = new QTableWidget(0,2,splitter);
    table->horizontalHeader()->setLabel(0,QString("Name"));
    table->horizontalHeader()->setStretchEnabled(true,0);
    table->horizontalHeader()->setLabel(1,QString("Time"));
    table->horizontalHeader()->setStretchEnabled(true,1);
#endif

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

        QColor color(100*(i%6 == 2)+255*(i%6 == 0 || i%6 == 3 || i%6 == 4)/(1+(0.4*i/6)),
                100*(i%6 == 2)+255*(i%6 == 1 || i%6 == 3 || i%6 == 5)/(1+(0.4*i/6)),
                255*(i%6 == 2 || i%6 == 4 || i%6 == 5)/(1+(0.4*i/6)));

        QString text(value[i].name.c_str());
        QString time= QString::number(value[i].time);
        time += QString(" ms");
        if (!value[i].type.empty())
        {
            text+="(";
            text+= QString(value[i].type.c_str());
            text+=")";
        }

#ifdef SOFA_QT4
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
        itemColor->setFlags(Qt::NoItemFlags);
        item->setFlags(Qt::NoItemFlags);
        itemTime->setFlags(Qt::NoItemFlags);
#else
        QTableWidgetItem *item = new QTableWidgetItem(table);
        QPixmap p(10,10); p.fill(color);
        item->setPixmap(p);
        QTableWidgetItem *itemTime = new QTableWidgetItem(table);
        item->setText(text);
        table->setItem(i,0, item);
        itemTime->setText(time);
        table->setItem(i,1, itemTime);
        item->setEnabled(false);
        itemTime->setEnabled(false);
#endif
    }
    this->update();
    this->repaint();

}

}
}
}
