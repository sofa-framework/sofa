/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef SOFA_GUI_QT_QMODELVIEWTABLEUPDATER_H
#define SOFA_GUI_QT_QMODELVIEWTABLEUPDATER_H


#include "SofaGUIQt.h"

#include <QDebug>
#include <QSpinBox>
#include <QTableView>
#include <QStandardItemModel>


namespace sofa
{
namespace gui
{
namespace qt
{

class QTableViewUpdater : public QTableView
{
    Q_OBJECT

public:
    QTableViewUpdater (QWidget * parent = 0):
        QTableView(parent)
    {
        setAutoFillBackground(true);
    };

public slots:
    void setDisplayed(bool b)
    {
        this->setShown(b);
    }

};

class QTableModelUpdater : public QStandardItemModel
{
    Q_OBJECT

public:
    QTableModelUpdater ( int numRows, int numCols, QWidget * parent = 0, const char * /*name*/ = 0 ):
        QStandardItemModel(numRows, numCols, parent)
    {};
public slots:
    void resizeTableV( int number )
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

    void resizeTableH( int number )
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
};

}
}
}
#endif
