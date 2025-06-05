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
#include <SofaMatrix/config.h>

#include <SofaMatrix/MatrixImageExporter.h>

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory;
#include <sofa/helper/system/PluginManager.h>

namespace sofa::component::linearsystem
{
    extern void registerGlobalSystemMatrixExporter(sofa::core::ObjectFactory* factory);
}
namespace sofa::component::linearsolver
{
    extern void registerFillReducingOrdering(sofa::core::ObjectFactory* factory);
    extern void registerGlobalSystemMatrixImage(sofa::core::ObjectFactory* factory);
}
namespace sofa::component::constraintset
{
    extern void registerComplianceMatrixExporter(sofa::core::ObjectFactory* factory);
    extern void registerComplianceMatrixImage(sofa::core::ObjectFactory* factory);
}


namespace sofamatrix
{
    
extern "C" {
    SOFA_SOFAMATRIX_API void initExternalModule();
    SOFA_SOFAMATRIX_API const char* getModuleName();
    SOFA_SOFAMATRIX_API const char* getModuleVersion();
    SOFA_SOFAMATRIX_API const char* getModuleLicense();
    SOFA_SOFAMATRIX_API const char* getModuleDescription();
    SOFA_SOFAMATRIX_API void registerObjects(sofa::core::ObjectFactory* factory);
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        // make sure that this plugin is registered into the PluginManager
        sofa::helper::system::PluginManager::getInstance().registerPlugin(MODULE_NAME);
        
        first = false;

        sofa::component::initializeMatrixExporterComponents();
    }
}

const char* getModuleName()
{
    return MODULE_NAME;
}

const char* getModuleVersion()
{
    return MODULE_VERSION;
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "SOFA plugin gathering components related to linear system matrices.";
}

void registerObjects(sofa::core::ObjectFactory* factory)
{
    sofa::component::constraintset::registerComplianceMatrixExporter(factory);
    sofa::component::constraintset::registerComplianceMatrixImage(factory);
    sofa::component::linearsolver::registerFillReducingOrdering(factory);
    sofa::component::linearsolver::registerGlobalSystemMatrixImage(factory);
    sofa::component::linearsystem::registerGlobalSystemMatrixExporter(factory);
}

} // namespace sofamatrix
