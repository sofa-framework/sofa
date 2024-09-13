/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#include <sofa/component/setting/StatsSetting.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::setting
{

void registerStatsSetting(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Stats settings.")
        .add< StatsSetting >());
}

StatsSetting::StatsSetting():
        d_dumpState(initData(&d_dumpState, false, "dumpState", "Dump state vectors at each time step of the simulation"))
    , d_logTime(initData(&d_logTime, false, "logTime", "Output in the console an average of the time spent during different stages of the simulation"))
    , d_exportState(initData(&d_exportState, false, "exportState", "Create GNUPLOT files with the positions, velocities and forces of all the simulated objects of the scene"))
#ifdef SOFA_DUMP_VISITOR_INFO
    , traceVisitors(initData(&traceVisitors, "traceVisitors", "Trace the time spent by each visitor, and allows to profile precisely one step of a simulation"))
#endif
{
    dumpState.setOriginalData(&d_dumpState);
    logTime.setOriginalData(&d_logTime);
    exportState.setOriginalData(&d_exportState);

}

} // namespace sofa::component::setting
