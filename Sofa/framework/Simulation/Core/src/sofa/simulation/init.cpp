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
#include <sofa/simulation/init.h>

#include <sofa/core/init.h>
#include <sofa/helper/init.h>

#include <sofa/simulation/MainTaskSchedulerRegistry.h>

#include <sofa/core/ObjectFactory.h>

namespace sofa::simulation
{

extern void registerRequiredPlugin(sofa::core::ObjectFactory* factory);
extern void registerDefaultVisualManagerLoop(sofa::core::ObjectFactory* factory);
extern void registerDefaultAnimationLoop(sofa::core::ObjectFactory* factory);

namespace core
{

static bool s_initialized = false;
static bool s_cleanedUp = false;


SOFA_SIMULATION_CORE_API void init()
{
    if (!s_initialized)
    {
        sofa::core::init();
        s_initialized = true;

        auto* factory = sofa::core::ObjectFactory::getInstance();
        registerRequiredPlugin(factory);
        registerDefaultVisualManagerLoop(factory);
        registerDefaultAnimationLoop(factory);
    }
}

SOFA_SIMULATION_CORE_API bool isInitialized()
{
    return s_initialized;
}

SOFA_SIMULATION_CORE_API void cleanup()
{
    if (!s_cleanedUp)
    {
        sofa::simulation::MainTaskSchedulerRegistry::clear();
        sofa::core::cleanup();
        s_cleanedUp = true;
    }
}

SOFA_SIMULATION_CORE_API bool isCleanedUp()
{
    return s_cleanedUp;
}

// Detect missing cleanup() call.
static const struct CleanupCheck
{
    CleanupCheck() {}
    ~CleanupCheck()
    {
        if (simulation::core::isInitialized() && !simulation::core::isCleanedUp())
            helper::printLibraryNotCleanedUpWarning("SofaSimulationCore", "sofa::simulation::core::cleanup()");
    }
} check;

} // namespace core

} // namespace sofa::simulation




