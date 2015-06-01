/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GUI_QT_QMOMENTUMSTATWIDGET_H
#define SOFA_GUI_QT_QMOMENTUMSTATWIDGET_H

#include "QGraphStatWidget.h"

#include <sofa/simulation/common/MechanicalGetMomentumVisitor.h>

namespace sofa
{
namespace gui
{
namespace qt
{

class QMomentumStatWidget : public QGraphStatWidget
{

    Q_OBJECT


    simulation::MechanicalGetMomentumVisitor *m_momentumVisitor;

public:

    QMomentumStatWidget( QWidget* parent, simulation::Node* node ) : QGraphStatWidget( parent, node, "Momenta", 6 )
    {
        setCurve( 0, "Linear X", Qt::red );
        setCurve( 1, "Linear Y", Qt::green );
        setCurve( 2, "Linear Z", Qt::blue );
        setCurve( 3, "Angular X", Qt::cyan );
        setCurve( 4, "Angular Y", Qt::magenta );
        setCurve( 5, "Angular Z", Qt::yellow );

        m_momentumVisitor = new simulation::MechanicalGetMomentumVisitor(core::MechanicalParams::defaultInstance());
    }

    virtual void step()
    {
        QGraphStatWidget::step(); // time history

        m_momentumVisitor->execute( _node->getContext() );

        const defaulttype::Vector6& momenta = m_momentumVisitor->getMomentum();

        // Add Momentum
        for( unsigned i=0 ; i<6 ; ++i ) _YHistory[i].push_back( momenta[i] );
    }

};


} // qt
} // gui
} //sofa

#endif // SOFA_GUI_QT_QMOMENTUMSTATWIDGET_H

