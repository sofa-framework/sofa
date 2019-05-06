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
#include "SofaWindowProfiler.h"

#include <QHeaderView>
#include <QMenu>
#include <QMessageBox>

#include <QGridLayout>



namespace sofa
{

namespace gui
{

namespace qt
{
//QPixmap *WindowVisitor::icons[WindowVisitor::OTHER+1];

using namespace sofa::helper;

SofaWindowProfiler::AnimationStepData::AnimationStepData(int step, std::map<AdvancedTimer::IdStep, std::string> _steps, std::map<AdvancedTimer::IdStep, sofa::helper::StepData> _stepData)
    : m_stepIteration(step)
    , m_totalMs(0.0)
{
    std::map<AdvancedTimer::IdStep, std::string>::iterator itM;
    //std::cout << " --------------- " << std::endl;
    static SReal timer_freqd = SReal(CTime::getTicksPerSec());
    for (itM = _steps.begin(); itM != _steps.end(); ++itM)
    {
        std::string stepName = (*itM).second;
        StepData& data = _stepData[(*itM).first];

        //std::cout << "Data: lvl: " << data.level << " ->  " << stepName << std::endl;
        if (data.level == 0) // main info
        {
            m_totalMs = 1000.0 * SReal(data.ttotal) / timer_freqd;
            //std::cout << "Data: lvl: " << data.level << " ->  " << stepName << " -> ms: " << data.ttotal << std::endl;
        }

    }
}

SofaWindowProfiler::AnimationStepData::~AnimationStepData()
{
    for (unsigned int i=0; i<m_subSteps.size(); ++i)
    {
        delete m_subSteps[i];
        m_subSteps[i] = nullptr;
    }
    m_subSteps.clear();
}


using namespace QtCharts;

SofaWindowProfiler::SofaWindowProfiler(QWidget *parent)
    : QDialog(parent)
    , m_step(0)
    , totalMs(0.0)
    , m_bufferSize(100)
    , m_maxFps(0)
{
    setupUi(this);

    m_profilingData.resize(m_bufferSize);
    createChart();
}


void SofaWindowProfiler::pushStepData()
{
    m_profilingData.pop_front();
    m_profilingData.push_back(AnimationStepData(m_step, sofa::helper::AdvancedTimer::getSteps("Animate", true), sofa::helper::AdvancedTimer::getStepData("Animate")));
    m_step++;
    //sofa::helper::AdvancedTimer::clearData("Animate");
    updateChart();
}


void SofaWindowProfiler::createChart()
{
    m_series = new QLineSeries();  

    for(unsigned int i=0; i<m_bufferSize; i++)
    {
        m_series->append(i, 0.0f);
    }

    m_chart = new QChart();
    m_chart->legend()->hide();
    m_chart->addSeries(m_series);
    m_chart->createDefaultAxes();
    m_chart->setTitle("Animation step duration (in ms)");
    m_chart->axisY()->setRange(0, 1000);

    m_chartView = new QChartView(m_chart);
    m_chartView->setRenderHint(QPainter::Antialiasing);
    graph_layout->addWidget(m_chartView);

   // m_chartView = new QChartView(chart);
    //m_chartView->setRenderHint(QPainter::Antialiasing);
}


void SofaWindowProfiler::updateChart()
{
    bool updateAxis = false;

    // Need to slide all the serie. Sure this could be optimised with deeper knowledge in QLineSeries/QChart
    int cpt = 0;
    for (auto stepData : m_profilingData)
    {
        m_series->replace(cpt, cpt, stepData.m_totalMs);
        if (m_maxFps < stepData.m_totalMs){
            m_maxFps = stepData.m_totalMs;
            updateAxis = true;
        }
        cpt++;
    }

    if (updateAxis)
        m_chart->axisY()->setRange(0, m_maxFps*1.1);

    m_chartView->update();
}


}
}
}
