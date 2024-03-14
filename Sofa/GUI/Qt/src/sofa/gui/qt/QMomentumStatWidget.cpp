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

#include <sofa/gui/qt/QMomentumStatWidget.h>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalGetMomentumVisitor.h>

namespace sofa::gui::qt
{

QMomentumStatWidget::QMomentumStatWidget( QWidget* parent, simulation::Node* node ) : QGraphStatWidget( parent, node, "Momenta", 6, 500 )
{
    setCurve( 0, "Linear X", Qt::red );
    setCurve( 1, "Linear Y", Qt::green );
    setCurve( 2, "Linear Z", Qt::blue );
    setCurve( 3, "Angular X", Qt::cyan );
    setCurve( 4, "Angular Y", Qt::magenta );
    setCurve( 5, "Angular Z", Qt::yellow );

    m_momentumVisitor = new simulation::mechanicalvisitor::MechanicalGetMomentumVisitor(core::mechanicalparams::defaultInstance());
}

QMomentumStatWidget::~QMomentumStatWidget()
{
    delete m_momentumVisitor;
}

void QMomentumStatWidget::stepImpl()
{
    if (m_curves.size() != 6) {
        msg_warning("QMomentumStatWidget") << "Wrong number of curves: " << m_curves.size() << ", should be 3.";
        return;
    }

    m_momentumVisitor->execute( m_node->getContext() );

    const type::Vec6& momenta = m_momentumVisitor->getMomentum();

    // Update series
    const SReal time = m_node->getTime();
    SReal min = 100000;
    SReal max = -100000;
    for (unsigned i = 0; i < 6; ++i)
    {
        m_curves[i]->append(time, momenta[i]);
        if (momenta[i] < min)
            min = momenta[i];
        if (momenta[i] > max)
            max = momenta[i];
    }

    // update minY
    updateYAxisBounds(min);
    // update maxY
    updateYAxisBounds(max);
}



} //namespace sofa::gui::qt
