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
#ifndef SOFA_GUI_QT_QENERGYSTATWIDGET_H
#define SOFA_GUI_QT_QENERGYSTATWIDGET_H

#include <sofa/simulation/common/Node.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/BaseMass.h>

#ifdef SOFA_QT4
#include <QWidget>
#include <QTextEdit>
#include <Q3GroupBox>
#else
#include <qwidget.h>
#include <qtextedit.h>
#include <qgroupbox.h>
#endif


#include <qwt_plot.h>
#include <qwt_plot_curve.h>

#ifndef SOFA_QT4
typedef QGroupBox Q3GroupBox;
typedef QTextEdit   Q3TextEdit;
#endif

namespace sofa
{
namespace gui
{
namespace qt
{

class QEnergyStatWidget : public QWidget
{
    Q_OBJECT
public:
    QEnergyStatWidget(QWidget* parent, simulation::Node* node);
    void step();
    void updateVisualization();

protected:
    simulation::Node *node;

    std::vector< double > history;
    std::vector< double > energy_history[3];


    QwtPlot *graphEnergy;
    QwtPlotCurve *energy_curve[3];
};


} // qt
} // gui
} //sofa

#endif // SOFA_GUI_QT_QDATADESCRIPTIONWIDGET_H

