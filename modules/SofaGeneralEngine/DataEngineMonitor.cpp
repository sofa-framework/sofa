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
{ }

void DataEngineMonitor::bwdInit()
{
    m_trackers.clear();
    m_trackerToEngine.clear();
    VecEngines const& engines = l_engines.getValue();
    for (size_t i=0; i<engines.size(); ++i) {
        m_trackers.push_back( DataTrackerFunctorSPtr(new DataTrackerFunctor<DataEngineMonitor>(*this)) );
        DataEngine* engine = engines[i].ptr.get();
        m_trackerToEngine[m_trackers[i].get()] = engine;
        const DDGNode::DDGLinkContainer& inputs = engine->getInputs();
        for (DDGNode::DDGLinkIterator it = inputs.begin(); it!= inputs.end(); ++it)
            m_trackers[i]->addInput((**it).getData());
    }
}

void DataEngineMonitor::operator()(core::DataTrackerFunctor<DataEngineMonitor>* tracker)
{
    DataEngine* engine = m_trackerToEngine[tracker];
    sout << getName() << " [" << engine->getClassName() << "] " << engine->getName() << " - ";
    if (engine->isDirty()) sout << "dirty";
    else sout << "clean";
    sout << " - dirty data: ";
    const DDGNode::DDGLinkContainer& inputs = engine->getInputs();
    for (DDGNode::DDGLinkIterator it = inputs.begin(); it!= inputs.end(); ++it) {
        BaseData const* data = (**it).getData();
        if (data->isDirty()) sout << data->getName() << ", ";
    }
    sout << sendl;
}

SOFA_DECL_CLASS(DataEngineMonitor)
int DataEngineMonitorClass = core::RegisterObject("Monitor engines").add< DataEngineMonitor >();

}
}
