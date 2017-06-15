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
#include "init.h"

#include <sofa/core/init.h>
#include <sofa/helper/init.h>

#include <sofa/helper/Factory.h>
#include <sofa/simulation/Node.inl>
#include <SofaSimulationCommon/xml/NodeElement.h>

namespace sofa
{

namespace simulation
{

namespace common
{

static bool s_initialized = false;
static bool s_cleanedUp = false;

//create method of Node called if the user wants the default node. The object created will depend on the simulation currently in use.
SOFA_SIMULATION_COMMON_API sofa::helper::Creator<xml::NodeElement::Factory, Node> NodeClass("default");

SOFA_SIMULATION_COMMON_API void init()
{
    if (!s_initialized)
    {
        sofa::core::init();
        s_initialized = true;
    }
}

SOFA_SIMULATION_COMMON_API bool isInitialized()
{
    return s_initialized;
}

SOFA_SIMULATION_COMMON_API void cleanup()
{
    if (!s_cleanedUp)
    {
        sofa::core::cleanup();
        s_cleanedUp = true;
    }
}

SOFA_SIMULATION_COMMON_API bool isCleanedUp()
{
    return s_cleanedUp;
}

// Detect missing cleanup() call.
static const struct CleanupCheck
{
    CleanupCheck() {}
    ~CleanupCheck()
    {
        if (simulation::common::isInitialized() && !simulation::common::isCleanedUp())
            helper::printLibraryNotCleanedUpWarning("SofaSimulationCommon", "sofa::simulation::common::cleanup()");
    }
} check;

} // namespace common

} // namespace simulation

} // namespace sofa
