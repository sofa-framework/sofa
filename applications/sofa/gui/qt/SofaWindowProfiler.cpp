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

using namespace QtCharts;

SofaWindowProfiler::SofaWindowProfiler()
    : m_step(0)
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
    m_profilingData[m_step].m_steps = sofa::helper::AdvancedTimer::getSteps("Animate");
    m_profilingData[m_step].m_stepData = sofa::helper::AdvancedTimer::getStepData("Animate");

    std::cout << "stepData: " << m_profilingData[m_step].m_stepData.size() << std::endl;

    //m_series->append(m_step, m_profilingData[m_step].m_stepData[sofa::helper::AdvancedTimer::IdStep()].ttotal);
    float fps = m_profilingData[m_step].m_stepData[sofa::helper::AdvancedTimer::IdStep()].ttotal;
    float diff = fps - totalMs;
    totalMs = fps;

    std::cout << "m_step: " << m_step << " ->fps: "<< diff << " / " << totalMs << std::endl;
    m_stepFps.pop_front();
    m_stepFps.push_back(diff);

    if (m_maxFps < diff)
        m_maxFps = diff;
//    m_chartView->addSeries(m_series);
//    m_chartView->update();
    m_step++;

    if (m_step == m_bufferSize) // loop over the buffer size
    {
        m_step = 0;
        m_chart->axisY()->setRange(0, m_maxFps*1.1);
    }
    updateChart();
}

//void WindowVisitor::setCharts(std::vector< dataTime >&latestC, std::vector< dataTime >&maxTC, std::vector< dataTime >&totalC,
//        std::vector< dataTime >&latestV, std::vector< dataTime >&maxTV, std::vector< dataTime >&totalV)
//{
//    componentsTime=latestC;
//    componentsTimeMax=maxTC;
//    componentsTimeTotal=totalC;
//    visitorsTime=latestV;
//    visitorsTimeMax=maxTV;
//    visitorsTimeTotal=totalV;
//    setCurrentCharts(typeOfCharts->currentIndex());
//}


void SofaWindowProfiler::createChart()
{
    m_series = new QLineSeries();  

    for(unsigned int i=0; i<m_bufferSize; i++)
    {
        m_stepFps.push_back(0.0f);
        m_series->append(i, 0.0f);
    }
//    series->append(0, 6);
//    series->append(2, 4);
//    series->append(3, 8);
//    series->append(7, 4);
//    series->append(10, 5);

    m_chart = new QChart();
    m_chart->legend()->hide();
    m_chart->addSeries(m_series);
    m_chart->createDefaultAxes();
    m_chart->setTitle("Simple line chart example");
    m_chart->axisY()->setRange(0, 1000);

    m_chartView = new QChartView(m_chart);
    m_chartView->setRenderHint(QPainter::Antialiasing);
    graph_layout->addWidget(m_chartView);
    //->addWidget(chartView);

   // m_chartView = new QChartView(chart);
    //m_chartView->setRenderHint(QPainter::Antialiasing);
}


void SofaWindowProfiler::updateChart()
{
    int cpt = 0;
    //std::deque<float>::iterator itQ;
    for (auto fps : m_stepFps)
    //for (itQ = m_stepFps.begin(); itQ != m_stepFps.end(); ++itQ)
    {
        m_series->replace(cpt, cpt, fps);
        cpt++;
    }

    m_chart->addSeries(m_series);
    m_chartView->update();
}




}
}
}
