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
#include "QEnergyStatWidget.h"

#include <qwt_legend.h>

#ifdef SOFA_QT4
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <Q3GroupBox>
#include <QLabel>
#else
#include <qlayout.h>
#include <qlabel.h>
#include <qgroupbox.h>
#endif



namespace sofa
{

namespace gui
{
namespace qt
{
QEnergyStatWidget::QEnergyStatWidget(QWidget* parent, simulation::Node* n):QWidget(parent), node(n)
{
    QVBoxLayout *tabMassStatsLayout = new QVBoxLayout( this, 0, 1, "tabMassStats");


#ifdef SOFA_QT4
    graphEnergy = new QwtPlot(QwtText("Energy"),this);
#else
    graphEnergy = new QwtPlot(this,"Energy");
#endif

    energy_curve[0] = new QwtPlotCurve("Kinetic");	        energy_curve[0]->attach(graphEnergy);
    energy_curve[1] = new QwtPlotCurve("Potential");	energy_curve[1]->attach(graphEnergy);
    energy_curve[2] = new QwtPlotCurve("Mechanical");	energy_curve[2]->attach(graphEnergy);

    energy_curve[0]->setPen(QPen(Qt::red));
    energy_curve[1]->setPen(QPen(Qt::green));
    energy_curve[2]->setPen(QPen(Qt::blue));

    graphEnergy->setAxisTitle(QwtPlot::xBottom, "Time/seconds");
    graphEnergy->setTitle("Energy Graph");
    graphEnergy->insertLegend(new QwtLegend(), QwtPlot::BottomLegend);

    tabMassStatsLayout->addWidget(graphEnergy);
}

void QEnergyStatWidget::step()
{
    //Add Time
    history.push_back(node->getTime());

    //Add Kinetic Energy
    unsigned int index = energy_history[0].size();
    if (node->mass)
        energy_history[0].push_back(node->mass->getKineticEnergy());
    else
        energy_history[0].push_back(0);

    //Add Potential Energy
    double potentialEnergy=0;
    typedef sofa::simulation::Node::Sequence<core::behavior::BaseForceField> SeqFF;
    for (SeqFF::iterator it=node->forceField.begin(); it!=node->forceField.end(); ++it)
    {
        potentialEnergy += (*it)->getPotentialEnergy();
    }
    energy_history[1].push_back(potentialEnergy);

    //Add Mechanical Energy
    energy_history[2].push_back(energy_history[0][index] + energy_history[1][index]);
}

void QEnergyStatWidget::updateVisualization()
{
    energy_curve[0]->setRawSamples(&history[0],&(energy_history[0][0]), history.size());
    energy_curve[1]->setRawSamples(&history[0],&(energy_history[1][0]), history.size());
    energy_curve[2]->setRawSamples(&history[0],&(energy_history[2][0]), history.size());
    graphEnergy->replot();
}


} // qt
} //gui
} //sofa

