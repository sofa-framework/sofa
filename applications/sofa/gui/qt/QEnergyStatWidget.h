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
#ifndef SOFA_GUI_QT_QENERGYSTATWIDGET_H
#define SOFA_GUI_QT_QENERGYSTATWIDGET_H

#include "QGraphStatWidget.h"

#include <sofa/simulation/MechanicalComputeEnergyVisitor.h>

namespace sofa
{
namespace gui
{
namespace qt
{

class QEnergyStatWidget : public QGraphStatWidget
{

    Q_OBJECT

    sofa::simulation::MechanicalComputeEnergyVisitor *m_energyVisitor;


public:

    QEnergyStatWidget( QWidget* parent, simulation::Node* node );

    ~QEnergyStatWidget();

    void step();

};


} // qt
} // gui
} //sofa

#endif // SOFA_GUI_QT_QDATADESCRIPTIONWIDGET_H

