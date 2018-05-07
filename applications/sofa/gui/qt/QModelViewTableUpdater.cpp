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

#include "QModelViewTableUpdater.h"

#include <QDebug>
#include <QSpinBox>
#include <QTableView>
#include <QStandardItemModel>
#include <QApplication>

namespace sofa
{
namespace gui
{
namespace qt
{

QTableViewUpdater::QTableViewUpdater(QWidget * parent )
    : QTableView(parent)
{
    setAutoFillBackground(true);
}

void QTableViewUpdater::setDisplayed(bool b)
{
    this->setVisible(b);
}


QTableModelUpdater::QTableModelUpdater ( int numRows, int numCols, QWidget * parent , const char * /*name*/ ):
    QStandardItemModel(numRows, numCols, parent)
{}

Qt::ItemFlags QTableModelUpdater::flags(const QModelIndex&) const
{
    if(m_isReadOnly)
        return (Qt::ItemIsSelectable) ;
    return Qt::ItemIsSelectable | Qt::ItemIsEditable | Qt::ItemIsEnabled ;
}

QVariant QTableModelUpdater::data(const QModelIndex &index, int role) const
{
    if(m_isReadOnly){
        switch(role){
        case Qt::BackgroundRole:
            return QApplication::palette().color(QPalette::Disabled, QPalette::Background) ;
        case Qt::ForegroundRole:
            return QApplication::palette().color(QPalette::Disabled, QPalette::Text);
        }
    }
    return QStandardItemModel::data(index,role) ;
}

void QTableModelUpdater::setReadOnly(const bool isReadOnly)
{
    m_isReadOnly=isReadOnly;
}

void QTableModelUpdater::resizeTableV( int number )
{
    QSpinBox *spinBox = (QSpinBox *) sender();
    QString header;
    if( spinBox == NULL)
    {
        return;
    }
    if (number != rowCount())
    {
        int previousRows=rowCount();
        setRowCount(number);
        if (number > previousRows)
        {
            for (int i=previousRows; i<number; ++i)
            {
                QStandardItem* header=verticalHeaderItem(i);
                if (!header) setVerticalHeaderItem(i, new QStandardItem(QString::number(i)));
                else header->setText(QString::number(i));
            }
        }
    }
}

void QTableModelUpdater::resizeTableH( int number )
{
    QSpinBox *spinBox = (QSpinBox *) sender();
    QString header;
    if( spinBox == NULL)
    {
        return;
    }
    if (number != columnCount())
    {
        int previousColumns=columnCount();
        setColumnCount(number);
        if (number > previousColumns)
        {
            for (int i=previousColumns; i<number; ++i)
            {
                QStandardItem* header=horizontalHeaderItem(i);
                if (!header) setHorizontalHeaderItem(i, new QStandardItem(QString::number(i)));
                else header->setText(QString::number(i));
            }
        }
    }
}


}
}
}

