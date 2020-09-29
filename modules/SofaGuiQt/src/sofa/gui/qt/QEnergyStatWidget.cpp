/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
    : QGraphStatWidget( parent, node, "Energy", 3, 500 )
{
    setCurve( 0, "Kinetic", Qt::red );
    setCurve( 1, "Potential", Qt::green );
    setCurve( 2, "Mechanical", Qt::blue );

    m_energyVisitor   = new sofa::simulation::MechanicalComputeEnergyVisitor(core::MechanicalParams::defaultInstance());
    m_yMin = 0;
}

QEnergyStatWidget::~QEnergyStatWidget()
{
    delete m_energyVisitor;
}

void QEnergyStatWidget::stepImpl()
{
    if (m_curves.size() != 3) {
        msg_warning("QEnergyStatWidget") << "Wrong number of curves: " << m_curves.size() << ", should be 3.";
        return;
    }
    m_energyVisitor->execute(m_node->getContext() );

    // Update series
    SReal time = m_node->getTime();
    SReal kinectic = m_energyVisitor->getKineticEnergy();
    SReal potential = m_energyVisitor->getPotentialEnergy();

    m_curves[0]->append(time, kinectic);
    m_curves[1]->append(time, potential);
    //Add Mechanical Energy
    m_curves[2]->append(time, kinectic + potential);

    if (potential > m_yMax)
    {
        m_yMax = potential;
        m_axisY->setRange(0, m_yMax*1.1);
    }
}


} // qt
} // gui
} //sofa


