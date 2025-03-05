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
#include <sofa/component/linearsolver/preconditioner/init.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>

namespace sofa::component::linearsolver::preconditioner
{

extern void registerBlockJacobiPreconditioner(sofa::core::ObjectFactory* factory);
extern void registerJacobiPreconditioner(sofa::core::ObjectFactory* factory);
extern void registerPrecomputedMatrixSystem(sofa::core::ObjectFactory* factory);
extern void registerPrecomputedWarpPreconditioner(sofa::core::ObjectFactory* factory);
extern void registerRotationMatrixSystem(sofa::core::ObjectFactory* factory);
extern void registerSSORPreconditioner(sofa::core::ObjectFactory* factory);
extern void registerWarpPreconditioner(sofa::core::ObjectFactory* factory);

extern "C" {
    SOFA_EXPORT_DYNAMIC_LIBRARY void initExternalModule();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleName();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleVersion();
    SOFA_EXPORT_DYNAMIC_LIBRARY void registerObjects(sofa::core::ObjectFactory* factory);
}

void initExternalModule()
{
    init();
}

const char* getModuleName()
{
    return MODULE_NAME;
}

const char* getModuleVersion()
{
    return MODULE_VERSION;
}

void registerObjects(sofa::core::ObjectFactory* factory)
{
    registerBlockJacobiPreconditioner(factory);
    registerJacobiPreconditioner(factory);
    registerPrecomputedMatrixSystem(factory);
    registerPrecomputedWarpPreconditioner(factory);
    registerRotationMatrixSystem(factory);
    registerSSORPreconditioner(factory);
    registerWarpPreconditioner(factory);
}

void init()
{
    static bool first = true;
    if (first)
    {
        // make sure that this plugin is registered into the PluginManager
        sofa::helper::system::PluginManager::getInstance().registerPlugin(MODULE_NAME);

        first = false;
    }
}

} // namespace sofa::component::linearsolver::preconditioner
