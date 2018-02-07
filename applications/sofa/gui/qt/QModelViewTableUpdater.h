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
    QTableViewUpdater (QWidget * parent = 0);

public slots:
    void setDisplayed(bool b);

};

class QTableModelUpdater : public QStandardItemModel
{
    Q_OBJECT
    bool m_isReadOnly ;
public:
    QTableModelUpdater ( int numRows, int numCols, QWidget * parent = 0, const char * /*name*/ = 0 );

    virtual Qt::ItemFlags flags(const QModelIndex&) const override ;
    virtual QVariant data(const QModelIndex &index, int role) const override;

    void setReadOnly(const bool isReadOnly) ;

public slots:
    void resizeTableV( int number );

    void resizeTableH( int number );
};

}
}
}
#endif
