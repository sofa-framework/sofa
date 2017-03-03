/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "DataEngineMonitor.h"

#include <cassert>

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

using namespace core;
using namespace core::objectmodel;

namespace component
{

DataEngineMonitor::DataEngineMonitor()
    : l_engines(initLink("engines", "Engine(s) to monitor"))
    , m_isInitialized(false)
{ }

void DataEngineMonitor::bwdInit()
{
    m_trackers.clear();
    m_inputTrackerToEngine.clear();
    m_outputTrackerToEngine.clear();
    VecEngines const& engines = l_engines.getValue();
    for (size_t i=0; i<engines.size(); ++i) {
        DataEngine* engine = engines[i].ptr.get();
        // input tracker
        m_trackers.push_back( DataTrackerFunctorSPtr(new DataTrackerFunctor<DataEngineMonitor>(*this)) );
        m_inputTrackerToEngine[m_trackers.back().get()] = engine;
        const DDGNode::DDGLinkContainer& inputs = engine->DDGNode::getInputs();
        for (DDGNode::DDGLinkIterator it = inputs.begin(); it!= inputs.end(); ++it)
            m_trackers.back()->addInput((**it).getData());
        // output tracker
        m_trackers.push_back( DataTrackerFunctorSPtr(new DataTrackerFunctor<DataEngineMonitor>(*this)) );
        m_outputTrackerToEngine[m_trackers.back().get()] = engine;
        const DDGNode::DDGLinkContainer& outputs = engine->DDGNode::getOutputs();
        for (DDGNode::DDGLinkIterator it = outputs.begin(); it!= outputs.end(); ++it)
            m_trackers.back()->addInput((**it).getData());
        sout << getName() << " add monitor to " << " [" << engine->getClassName() << "] " << engine->getName()
             << " - nb inputs: " << inputs.size() << " - nb outputs: " << outputs.size()
             << sendl;
    }
    m_isInitialized = true;
}

void DataEngineMonitor::operator()(core::DataTrackerFunctor<DataEngineMonitor>* tracker)
{
    if (!m_isInitialized) return; // to ignore callback triggered by the DataTrackerFunctor::addInput
    auto itInput = m_inputTrackerToEngine.find(tracker);
    if (itInput != m_inputTrackerToEngine.end()) {
        DataEngine* engine = itInput->second;
        sout << getName() << " [" << engine->getClassName() << "] " << engine->getName() << " - ";
        if (engine->isDirty()) sout << "dirty";
        else sout << "clean";
        sout << " - dirty INPUT: ";
        const DDGNode::DDGLinkContainer& inputs = engine->DDGNode::getInputs();
        for (DDGNode::DDGLinkIterator it = inputs.begin(); it!= inputs.end(); ++it) {
            BaseData const* data = (**it).getData();
            if (data->isDirty()) sout << data->getName() << ", ";
        }
        sout << sendl;
    }
    auto itOutput = m_outputTrackerToEngine.find(tracker);
    if (itOutput != m_outputTrackerToEngine.end()) {
        DataEngine* engine = itOutput->second;
        sout << getName() << " [" << engine->getClassName() << "] " << engine->getName() << " - ";
        if (engine->isDirty()) sout << "dirty";
        else sout << "clean";
        sout << " - dirty OUTPUT: ";
        const DDGNode::DDGLinkContainer& outputs = engine->DDGNode::getOutputs();
        for (DDGNode::DDGLinkIterator it = outputs.begin(); it!= outputs.end(); ++it) {
            BaseData const* data = (**it).getData();
            if (data->isDirty()) sout << data->getName() << ", ";
        }
        sout << sendl;
    }

}

SOFA_DECL_CLASS(DataEngineMonitor)
int DataEngineMonitorClass = core::RegisterObject("Monitor engines").add< DataEngineMonitor >();

}
}
