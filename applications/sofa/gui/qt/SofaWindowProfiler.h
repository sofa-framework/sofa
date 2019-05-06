/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_WINDOWPROFILER_H
#define SOFA_WINDOWPROFILER_H

#include <ui_WindowProfiler.h>
#include "PieWidget.h"
#include "QVisitorControlPanel.h"

#include <QTreeWidgetItem>
#include <QDrag>
#include <QPixmap>
#include <QTableWidget>
#include <QComboBox>

#include <QDialog>
#include <QPainter>
#include <QTableWidget>

#include <QtCharts/QChart>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>

#include <iostream>
#include <sofa/helper/AdvancedTimer.h>
#include <deque>

namespace sofa
{

namespace gui
{

namespace qt
{

class SofaWindowProfiler: public QDialog, public Ui_WindowProfiler
{
    Q_OBJECT
public:
    enum componentType {NODE, COMMENT, COMPONENT, VECTOR, OTHER};
    SofaWindowProfiler(QWidget* parent);

    void pushStepData();


    class AnimationSubStepData
    {
    public:
        AnimationSubStepData(){}

        int m_level;
        std::string m_subStepName;
        SReal m_totalMs;
        SReal m_totalPercent;
        SReal m_selfMs;
        SReal m_selfPercent;

        sofa::helper::vector<AnimationSubStepData*> m_children;
    };

    class AnimationStepData
    {
    public:
        AnimationStepData()
            : m_stepIteration(-1)
            , m_totalMs(0.0)
            , m_totalPercent(0.0)
        {}

        AnimationStepData(int step, helper::vector<helper::AdvancedTimer::IdStep> _steps, std::map<sofa::helper::AdvancedTimer::IdStep, sofa::helper::StepData> _stepData);

        virtual ~AnimationStepData();
        int m_stepIteration;
        SReal m_totalMs;
        SReal m_totalPercent;

        sofa::helper::vector<AnimationSubStepData*> m_subSteps;
    };

public slots:
    void closeEvent( QCloseEvent* )
    {
        emit(closeWindow(false));
    }

    void updateSummaryLabels(int step);
    void updateTree(int step);

signals:
    void closeWindow(bool);

protected:
    void updateChart();
    void createChart();

    QtCharts::QChart *m_chart;
    QtCharts::QChartView* m_chartView;
    //sofa::helper::vector<AnimationStepData> m_profilingData;
    int m_step;
    int m_bufferSize;
    float m_maxFps;
    float m_fpsMaxAxis;
    std::deque<AnimationStepData> m_profilingData;
    QtCharts::QLineSeries *m_series;
    sofa::helper::system::thread::ctime_t totalMs;
};
}
}
}

#endif // SOFA_WINDOWPROFILER_H
