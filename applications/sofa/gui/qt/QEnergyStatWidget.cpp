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
#include "QEnergyStatWidget.h"


namespace sofa
{
namespace gui
{
namespace qt
{

QEnergyStatWidget::QEnergyStatWidget( QWidget* parent, simulation::Node* node )
    : QGraphStatWidget( parent, node, "Energy", 3 )
{
    setCurve( 0, "Kinetic", Qt::red );
    setCurve( 1, "Potential", Qt::green );
    setCurve( 2, "Mechanical", Qt::blue );

    m_energyVisitor   = new sofa::simulation::MechanicalComputeEnergyVisitor(core::MechanicalParams::defaultInstance());
}

QEnergyStatWidget::~QEnergyStatWidget()
{
    delete m_energyVisitor;
}

void QEnergyStatWidget::step()
{
    //Add Time
    QGraphStatWidget::step();

    m_energyVisitor->execute( _node->getContext() );

    _YHistory[0].push_back( m_energyVisitor->getKineticEnergy() );
    _YHistory[1].push_back( m_energyVisitor->getPotentialEnergy() );

    //Add Mechanical Energy
    _YHistory[2].push_back( _YHistory[0].back() + _YHistory[1].back() );
}


} // qt
} // gui
} //sofa


