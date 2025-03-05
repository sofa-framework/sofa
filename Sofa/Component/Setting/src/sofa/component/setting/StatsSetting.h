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
#pragma once

#include <sofa/component/setting/config.h>

#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/type/Vec.h>

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::component::setting
{

///Class for the configuration of stats settings.
class SOFA_COMPONENT_SETTING_API StatsSetting: public core::objectmodel::ConfigurationSetting
{
public:
    SOFA_CLASS(StatsSetting,core::objectmodel::ConfigurationSetting);   ///< Sofa macro to define typedef.
protected:
    /**
     * @brief Default constructor.
     *
     * By default :
     *  - @ref d_dumpState is set to false.
     *  - @ref logTime is set to false.
     *  - @ref d_exportState is set to false.
     */
    StatsSetting();
public:

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SETTING()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> dumpState;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SETTING()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> logTime;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SETTING()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> exportState;


    Data<bool> d_dumpState; ///< Dump state vectors at each time step of the simulation
    Data<bool> d_logTime; ///< Output in the console an average of the time spent during different stages of the simulation
    Data<bool> d_exportState; ///< Create GNUPLOT files with the positions, velocities and forces of all the simulated objects of the scene
#ifdef SOFA_DUMP_VISITOR_INFO
    Data<bool> traceVisitors; ///< Trace the time spent by each visitor, and allows to profile precisely one step of a simulation
#endif

};

} // namespace sofa::component::setting
