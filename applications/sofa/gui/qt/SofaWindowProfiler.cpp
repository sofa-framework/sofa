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
#include <QDebug>
#include <QValueAxis>

namespace sofa
{

namespace gui
{

namespace qt
{
using namespace sofa::helper;
using namespace QtCharts;

typedef sofa::helper::system::thread::ctime_t ctime_t;
typedef sofa::helper::system::thread::CTime CTime;


///////////////////////////////////////// ProfilerChartView ///////////////////////////////////

ProfilerChartView::ProfilerChartView(QChart *chart, QWidget *parent, int bufferSize)
    : QChartView(chart, parent)
    , m_bufferSize(bufferSize)
    , m_pointSelected(-1)
    , m_maxY(1000)
{

}

void ProfilerChartView::mousePressEvent(QMouseEvent *event)
{
    auto const valueInSeries = chart()->mapToValue(event->localPos());

    int width = valueInSeries.x();

    if (width >= 0 && width < m_bufferSize)
    {
        m_pointSelected = width;
        updateSelection(m_pointSelected);
        emit pointSelected(m_pointSelected);
    }
    else
        m_pointSelected = -1;
}

void ProfilerChartView::updateSelection(int x)
{
    m_pointSelected = x;
    m_lineSelect = chart()->mapToPosition(QPointF(x, m_maxY));
    m_lineOrigin = chart()->mapToPosition(QPointF(x, 0));

    this->scene()->update(this->sceneRect());
}


void ProfilerChartView::drawForeground(QPainter *painter, const QRectF &)
{    
    if (m_pointSelected == -1)
        return;

    painter->drawLine(m_lineOrigin, m_lineSelect);
}



// quick method to convert freq time into ms
SReal convertInMs(ctime_t t, int nbIter=1)
{
    static SReal timer_freqd = SReal(CTime::getTicksPerSec());
    return 1000.0 * SReal(t) / SReal (timer_freqd * nbIter);
}

///////////////////////////////////////// AnimationSubStepData ///////////////////////////////////

SofaWindowProfiler::AnimationSubStepData::AnimationSubStepData(int level, std::string name, ctime_t start)
    : m_level(level)
    , m_name(name)
    , m_nbrCall(1)
    , m_start(start)
{

}

SofaWindowProfiler::AnimationSubStepData::~AnimationSubStepData()
{
    for (unsigned int i=0; i<m_children.size(); ++i)
        delete m_children[i];

    m_children.clear();
}

void SofaWindowProfiler::AnimationSubStepData::computeTimeAndPercentage(SReal invTotalMs)
{
    if (!m_children.empty()) // compute from leaf to trunk
    {
        SReal totalChildrenMs = 0.0;
        for (unsigned int i=0; i<m_children.size(); i++)
        {
            m_children[i]->computeTimeAndPercentage(invTotalMs);
            totalChildrenMs += m_children[i]->m_totalMs;
        }

        // now that all children are update, compute ms and %
        m_totalMs = convertInMs(m_end - m_start);
        m_selfMs = m_totalMs - totalChildrenMs;

        m_selfPercent = m_selfMs * invTotalMs;
        m_totalPercent = m_totalMs * invTotalMs;
    }
    else // leaf
    {
        // compute ms:
        m_totalMs = convertInMs(m_end - m_start);
        if (m_nbrCall != 1)
            m_selfMs = SReal(m_totalMs / m_nbrCall);
        else
            m_selfMs = m_totalMs;

        // compute %
        m_selfPercent = m_selfMs * invTotalMs;
        m_totalPercent = m_totalMs * invTotalMs;
    }
}

SReal SofaWindowProfiler::AnimationSubStepData::getStepMs(const std::string& stepName, const std::string& parentName)
{
    SReal result = 0.0;
    if (parentName == m_name)
    {
        for (unsigned int i=0; i<m_children.size(); i++)
        {
            if (m_children[i]->m_name == stepName)
                return m_children[i]->m_totalMs;
        }
    }
    else
    {
        for (unsigned int i=0; i<m_children.size(); i++)
        {
            result = m_children[i]->getStepMs(stepName, parentName);
            if (result != 0.0)
                return result;
        }
    }

    return 0.0;
}



///////////////////////////////////////// AnimationStepData ///////////////////////////////////

SofaWindowProfiler::AnimationStepData::AnimationStepData(int step, const std::string& idString)
    : m_stepIteration(step)
    , m_totalMs(0.0)
{
    m_subSteps.clear();
    
    bool res = processData(idString);
    if (!res) // error clear data
    {
        for (unsigned int i=0; i<m_subSteps.size(); ++i)
        {
            delete m_subSteps[i];
            m_subSteps[i] = nullptr;
        }
        m_subSteps.clear();
    }
}


bool SofaWindowProfiler::AnimationStepData::processData(const std::string& idString)
{
    helper::vector<Record> _records = sofa::helper::AdvancedTimer::getRecords(idString);

    //AnimationSubStepData* currentSubStep = nullptr;
    std::stack<AnimationSubStepData*> processStack;
    int level = 0;
    ctime_t t0 = 0;
    ctime_t tEnd = CTime::getTime();
    ctime_t tCurr;
    for (unsigned int ri = 0; ri < _records.size(); ++ri)
    {
        const Record& rec = _records[ri];

        if (level == 0) // main step
        {
            t0 = rec.time;
            level++;
            continue;
        }

        tCurr = rec.time - t0;

        if (rec.type == Record::RBEGIN || rec.type == Record::RSTEP_BEGIN || rec.type == Record::RSTEP)
        {
//            for (int i=0; i<level; ++i)
//                std::cout << ".";
//            std::cout << level << " Begin: " << rec.label << " at " << rec.time << " obj: " << rec.obj << " val: " << rec.val << std::endl;

            AnimationSubStepData* currentSubStep = new AnimationSubStepData(level, rec.label, tCurr);
            if (rec.obj)
                currentSubStep->m_tag = std::string(AdvancedTimer::IdObj(rec.obj));

            if (level == 1) // Add top level step
                m_subSteps.push_back(currentSubStep);
            else
            {
                if (processStack.empty())
                {
                    msg_error("SofaWindowProfiler") << "No parent found to add step: " << currentSubStep->m_name;
                    delete currentSubStep;
                    return false;
                }
                else if (processStack.top()->m_level + 1 != currentSubStep->m_level)
                {
                    msg_warning("SofaWindowProfiler") << "Problem of level coherence between step: " << currentSubStep->m_name << " with level: " << currentSubStep->m_level
                                                      << " and parent step: " << processStack.top()->m_name << " with level: " << processStack.top()->m_level;
                }

                // add next step to the hierarchy
                processStack.top()->m_children.push_back(currentSubStep);
            }

            // add step into the stack for parent/child order
            processStack.push(currentSubStep);
            ++level;
        }

        if (rec.type == Record::REND || rec.type == Record::RSTEP_END)
        {
            --level;
//            for (int i=0; i<level; ++i)
//                std::cout << ".";
//            std::cout << level << " End: " << rec.label << " at " << rec.time << " obj: " << rec.obj << " val: " << rec.val << std::endl;

            if (processStack.empty())
            {
                msg_error("SofaWindowProfiler") << "End step with no step in the stack for: " << rec.label;
                return false;
            }
            else if (rec.label != processStack.top()->m_name)
            {
                msg_error("SofaWindowProfiler") << "Not the same name to end step between logs: " << rec.label << " and top stack: " << processStack.top()->m_name;
                return false;
            }

            AnimationSubStepData* currentSubStep = processStack.top();
            processStack.pop();

            currentSubStep->m_end = tCurr;
        }
    }
    // compute total MS step:
    m_totalMs = convertInMs(tEnd - t0);

    // update percentage
    SReal invTotalMs = 100 / m_totalMs;
    for (unsigned int i=0; i<m_subSteps.size(); i++)
    {
        m_subSteps[i]->computeTimeAndPercentage(invTotalMs);
    }

    return true;
}


SReal SofaWindowProfiler::AnimationStepData::getStepMs(const std::string& stepName, const std::string& parentName)
{
    SReal result = 0.0;
    if (parentName == "")
    {
        for (unsigned int i=0; i<m_subSteps.size(); i++)
        {
            if (m_subSteps[i]->m_name == stepName)
                return m_subSteps[i]->m_totalMs;
        }
    }
    else
    {
        for (unsigned int i=0; i<m_subSteps.size(); i++)
        {
            result = m_subSteps[i]->getStepMs(stepName, parentName);
            if (result != 0.0)
                return result;
        }
    }

    return 0.0;
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



///////////////////////////////////////// SofaWindowProfiler ///////////////////////////////////

SofaWindowProfiler::SofaWindowProfiler(QWidget *parent)
    : QDialog(parent)
    , m_step(0)
    , m_bufferSize(100)
    , m_maxFps(0)
    , m_fpsMaxAxis(0)
    , m_selectedStep("")
    , m_selectedParentStep("")
{
    setupUi(this);

    // fill buffer with empty data.
    m_profilingData.resize(m_bufferSize);
    for (unsigned int i=0; i<m_bufferSize; ++i)
        m_profilingData[i] = new AnimationStepData();

    // creating chart widget
    createChart();

    // create treeView
    createTreeView();

    // create and connect different widgets
    step_scroller->setRange(0, m_bufferSize-1);
    step_scroller->setMinimumWidth(200);
    step_scroller->setMaximumWidth(200);
    connect(step_scroller, SIGNAL(valueChanged(int)), this, SLOT(updateSummaryLabels(int)));
    connect(step_scroller, SIGNAL(valueChanged(int)), this, SLOT(updateTree(int)));

    connect(step_scroller, SIGNAL(valueChanged(int)), m_chartView, SLOT(updateSelection(int)));
    connect(m_chartView, SIGNAL(pointSelected(int)), this, SLOT(updateFromSelectedStep(int)));
}


void SofaWindowProfiler::activateATimer(bool activate)
{
    sofa::helper::AdvancedTimer::setEnabled("Animate", activate);
    sofa::helper::AdvancedTimer::setInterval("Animate", 1);
    sofa::helper::AdvancedTimer::setOutputType("Animate", "gui");
}


void SofaWindowProfiler::pushStepData()
{
    m_profilingData.pop_front();
    std::string idString = "Animate";
    m_profilingData.push_back(new AnimationStepData(m_step, idString));
    m_step++;

    updateChart();
}


void SofaWindowProfiler::resetGraph()
{
    if (m_step == 0)
        return;

    for(unsigned int i=0; i<m_bufferSize; i++)
    {
        m_series->replace(i, 0.0, 0.0);
        m_selectionSeries->replace(i, 0.0, 0.0);
        if (m_profilingData[i] && m_profilingData[i]->m_stepIteration != -1)
        {
            delete m_profilingData[i];
            m_profilingData[i] = new AnimationStepData();
        }
    }

    step_scroller->setValue(1); // for rest by changing 2 times value
    step_scroller->setValue(0);
    m_step = 0;
    m_selectedStep = "";
    m_selectedParentStep = "";
}


void SofaWindowProfiler::createTreeView()
{
    // set column names
    QStringList columnNames;
    columnNames << "Hierarchy Step Name" << "Total (%)" << "Self (%)" << "Time (ms)" << "Self (ms)";
    tree_steps->setHeaderLabels(columnNames);

    // set column properties
    tree_steps->header()->setStretchLastSection(false);
    tree_steps->header()->setSectionResizeMode(0, QHeaderView::Stretch);

    connect(tree_steps, SIGNAL(itemClicked(QTreeWidgetItem*,int)), this, SLOT(onStepSelected(QTreeWidgetItem*,int)));
}


void SofaWindowProfiler::createChart()
{
    m_series = new QLineSeries();
    QPen pen = m_series->pen();
    pen.setColor(Qt::red);
    pen.setWidth(2);
    m_series->setName("Full Animation Step");
    m_series->setPen(pen);

    m_selectionSeries = new QLineSeries();
    m_selectionSeries->setName("Selected SubStep");

    for(unsigned int i=0; i<m_bufferSize; i++)
    {
        m_series->append(i, 0.0f);
        m_selectionSeries->append(i, 0.0f);
    }

    m_chart = new QChart();
    m_chart->addSeries(m_series);
    m_chart->addSeries(m_selectionSeries);
    QValueAxis *axisY = new QValueAxis();
    m_chart->addAxis(axisY, Qt::AlignLeft);
    m_series->attachAxis(axisY);
    m_selectionSeries->attachAxis(axisY);

    m_chart->setTitle("Steps durations (in ms)");
    m_chart->axisY()->setRange(0, 1000);

    m_chartView = new ProfilerChartView(m_chart, this, m_bufferSize);
    m_chartView->setRenderHint(QPainter::Antialiasing);

    Layout_graph->addWidget(m_chartView);
}


void SofaWindowProfiler::updateChart()
{
    bool updateAxis = false;

    // Need to slide all the serie. Sure this could be optimised with deeper knowledge in QLineSeries/QChart
    int cpt = 0;
    for (auto* stepData : m_profilingData)
    {
        m_series->replace(cpt, cpt, stepData->m_totalMs);

        if (m_selectedStep != "")
        {
            SReal value = stepData->getStepMs(m_selectedStep, m_selectedParentStep);
            m_selectionSeries->replace(cpt, cpt, value);
        }

        if (m_fpsMaxAxis < stepData->m_totalMs){
            m_fpsMaxAxis = stepData->m_totalMs;
            updateAxis = true;
        }

        // keep max ms value
        if (m_maxFps < stepData->m_totalMs)
            m_maxFps = stepData->m_totalMs;

        cpt++;
    }

    // if needed enlarge the Y axis to cover new data
    if (updateAxis){
        m_chart->axisY()->setRange(0, m_fpsMaxAxis*1.1);
        m_chartView->updateYMax(m_fpsMaxAxis*1.1);
    }

    // every loop on buffer size check if Y axis can be reduced
    if ((m_step% m_bufferSize) == 0)
    {
        if (m_maxFps < m_fpsMaxAxis)
            m_fpsMaxAxis = m_maxFps;

        m_maxFps = 0;
        m_chart->axisY()->setRange(0, m_fpsMaxAxis*1.1);
        m_chartView->updateYMax(m_fpsMaxAxis*1.1);
    }

 //   m_chartView->update();

    // update all widgets from value sliced
    updateSummaryLabels(step_scroller->value());
    updateTree(step_scroller->value());
}


void SofaWindowProfiler::updateFromSelectedStep(int step)
{
    // only update scroller as all the connection are already made from the scroller value change.
    step_scroller->setValue(step);
}


void SofaWindowProfiler::updateSummaryLabels(int step)
{
    const AnimationStepData* stepData = m_profilingData.at(step);
    label_stepValue->setText(QString::number(stepData->m_stepIteration));
    label_timeValue->setText(QString::number(stepData->m_totalMs));
}

void SofaWindowProfiler::updateTree(int step)
{
    const AnimationStepData* stepData = m_profilingData.at(step);

    //clear old values
    tree_steps->clear();

    // add new step items
    for (unsigned int i=0; i<stepData->m_subSteps.size(); i++)
    {
        addTreeItem(stepData->m_subSteps[i], nullptr);
    }
}

void SofaWindowProfiler::addTreeItem(AnimationSubStepData* subStep, QTreeWidgetItem* parent)
{
    // add item to the tree
    QTreeWidgetItem* treeItem = nullptr;
    if (parent == nullptr) // top item
        treeItem = new QTreeWidgetItem(tree_steps);
    else
        treeItem = new QTreeWidgetItem(parent);

    treeItem->setText(0, QString::fromStdString(subStep->m_name));
    treeItem->setText(1, QString::number(subStep->m_totalPercent, 'g', 2));
    treeItem->setText(2, QString::number(subStep->m_selfPercent, 'g', 2));
    treeItem->setText(3, QString::number(subStep->m_totalMs));
    treeItem->setText(4, QString::number(subStep->m_selfMs));

    if (subStep->m_level <= 1)
    {        
        QFont font = QApplication::font();
        font.setBold(true);

        for (int i=0; i<treeItem->columnCount(); i++)
            treeItem->setFont(i, font);
    }
    if (subStep->m_level <= 3)
        treeItem->setExpanded(true);

    // process children
    for (unsigned int i=0; i<subStep->m_children.size(); i++)
        addTreeItem(subStep->m_children[i], treeItem);
}

void SofaWindowProfiler::onStepSelected(QTreeWidgetItem *item, int /*column*/)
{
    if (item->parent())
        m_selectedParentStep = item->parent()->text(0).toStdString();
    m_selectedStep = item->text(0).toStdString();

    int cpt = 0;
    for (auto* stepData : m_profilingData)
    {
        SReal value = stepData->getStepMs(m_selectedStep, m_selectedParentStep);
        m_selectionSeries->replace(cpt, cpt, value);
        cpt++;
    }
}


} // namespace qt

} // namespace gui

} // namespace sofa
