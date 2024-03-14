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
#include <sofa/gui/qt/QEnergyStatWidget.h>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalComputeEnergyVisitor.h>

#include <QLineSeries>

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
using namespace QtCharts;
#endif

namespace sofa::gui::qt
{

QEnergyStatWidget::QEnergyStatWidget( QWidget* parent, simulation::Node* node )
    : QGraphStatWidget( parent, node, "Energy", 3, 500 )
{
    setCurve( 0, "Kinetic", Qt::red );
    setCurve( 1, "Potential", Qt::green );
    setCurve( 2, "Mechanical", Qt::blue );

    m_energyVisitor   = new sofa::simulation::mechanicalvisitor::MechanicalComputeEnergyVisitor(core::mechanicalparams::defaultInstance());
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
    const SReal time = m_node->getTime();
    const SReal kinectic = m_energyVisitor->getKineticEnergy();
    const SReal potential = m_energyVisitor->getPotentialEnergy();

    m_curves[0]->append(time, kinectic);
    m_curves[1]->append(time, potential);
    //Add Mechanical Energy
    m_curves[2]->append(time, kinectic + potential);

    // update maxY
    updateYAxisBounds(kinectic + potential);
    
    // update minY
    if (kinectic < potential)
        updateYAxisBounds(kinectic);
    else
        updateYAxisBounds(potential);
}


} //namespace sofa::gui::qt
