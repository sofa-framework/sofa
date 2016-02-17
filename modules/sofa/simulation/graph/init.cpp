/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "graph.h"

#include <sofa/simulation/common/init.h>
#include <sofa/helper/init.h>

namespace sofa
{

namespace simulation
{

namespace graph
{

static bool s_initialized = false;
static bool s_cleanedUp = false;

SOFA_SIMULATION_GRAPH_API void init()
{
    if (!s_initialized)
    {
        sofa::simulation::common::init();
        s_initialized = true;
    }
}

SOFA_SIMULATION_GRAPH_API bool isInitialized()
{
    return s_initialized;
}

SOFA_SIMULATION_GRAPH_API void cleanup()
{
    if (!s_cleanedUp)
    {
        sofa::simulation::common::cleanup();
        s_cleanedUp = true;
    }
}

SOFA_SIMULATION_GRAPH_API bool isCleanedUp()
{
    return s_cleanedUp;
}

// Detect missing cleanup() call.
static const struct CleanupCheck
{
    CleanupCheck() {}
    ~CleanupCheck()
    {
        if (simulation::graph::isInitialized() && !simulation::graph::isCleanedUp())
            helper::printLibraryNotCleanedUpWarning("SofaSimulationGraph", "sofa::simulation::graph::cleanup()");
    }
} check;

} // namespace graph

} // namespace simulation

} // namespace sofa
