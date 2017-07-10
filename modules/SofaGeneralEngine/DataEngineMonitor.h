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
#ifndef SOFA_CORE_DATAENGINEMONITOR_H
#define SOFA_CORE_DATAENGINEMONITOR_H

#include <memory>
#include <vector>
#include <map>

#include "config.h"

#include <sofa/core/objectmodel/BaseObject.h>

#include <sofa/core/DataEngine.h>
#include <sofa/core/DataTracker.h>

namespace sofa
{

namespace component
{

/** This component monitors several engines. Each time an engine input changes a message is printed with engine name and dirty data names.
 * This component should help debugging a chain of engines that does not behave as expected.
 */
class SOFA_GENERAL_ENGINE_API DataEngineMonitor : public virtual core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(DataEngineMonitor, core::objectmodel::BaseObject);

    DataEngineMonitor();
    virtual ~DataEngineMonitor() { }

    void bwdInit();

    void operator()(core::DataTrackerFunctor<DataEngineMonitor>* tracker);

protected:

    typedef MultiLink<DataEngineMonitor, core::DataEngine, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> LinkEngines;
    typedef LinkEngines::Container VecEngines;
    typedef std::unique_ptr<core::DataTrackerFunctor<DataEngineMonitor> > DataTrackerFunctorSPtr;

    LinkEngines l_engines;
    std::vector< DataTrackerFunctorSPtr > m_trackers;

    std::map< core::DataTrackerFunctor<DataEngineMonitor>*, core::DataEngine*> m_inputTrackerToEngine;
    std::map< core::DataTrackerFunctor<DataEngineMonitor>*, core::DataEngine*> m_outputTrackerToEngine;

    bool m_isInitialized;

};

} // namespace component

} // namespace sofa

#endif
