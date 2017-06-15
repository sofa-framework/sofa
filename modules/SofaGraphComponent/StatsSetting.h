/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_CONFIGURATIONSETTING_STATS_H
#define SOFA_COMPONENT_CONFIGURATIONSETTING_STATS_H
#include "config.h"

#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace configurationsetting
{

///Class for the configuration of stats settings.
class SOFA_GRAPH_COMPONENT_API StatsSetting: public core::objectmodel::ConfigurationSetting
{
public:
    SOFA_CLASS(StatsSetting,core::objectmodel::ConfigurationSetting);   ///< Sofa macro to define typedef.
protected:
    /**
     * @brief Default constructor.
     *
     * By default :
     *  - @ref dumpState is set to false.
     *  - @ref logTime is set to false.
     *  - @ref exportState is set to false.
     */
    StatsSetting();
public:
    Data<bool> dumpState;       ///< If true, dump state vectors at each time step of the simulation.
    Data<bool> logTime;         ///< If true, output in the console an average of the time spent during different stages of the simulation.
    Data<bool> exportState;     ///< If true, create GNUPLOT files with the positions, velocities and forces of all the simulated objects of the scene.
#ifdef SOFA_DUMP_VISITOR_INFO
    Data<bool> traceVisitors;   ///< If true, trace the time spent by each visitor, and allows to profile precisely one step of a simulation.
#endif

};

}

}

}
#endif
