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
#include <sofa/component/mechanicalload/init.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>

namespace sofa::component::mechanicalload
{

extern void registerConicalForceField(sofa::core::ObjectFactory* factory);
extern void registerConstantForceField(sofa::core::ObjectFactory* factory);
extern void registerDiagonalVelocityDampingForceField(sofa::core::ObjectFactory* factory);
extern void registerEdgePressureForceField(sofa::core::ObjectFactory* factory);
extern void registerEllipsoidForceField(sofa::core::ObjectFactory* factory);
extern void registerGravity(sofa::core::ObjectFactory* factory);
extern void registerInteractionEllipsoidForceField(sofa::core::ObjectFactory* factory);
extern void registerLinearForceField(sofa::core::ObjectFactory* factory);
extern void registerOscillatingTorsionPressureForceField(sofa::core::ObjectFactory* factory);
extern void registerPlaneForceField(sofa::core::ObjectFactory* factory);
extern void registerQuadPressureForceField(sofa::core::ObjectFactory* factory);
extern void registerSphereForceField(sofa::core::ObjectFactory* factory);
extern void registerSurfacePressureForceField(sofa::core::ObjectFactory* factory);
extern void registerTaitSurfacePressureForceField(sofa::core::ObjectFactory* factory);
extern void registerTorsionForceField(sofa::core::ObjectFactory* factory);
extern void registerTrianglePressureForceField(sofa::core::ObjectFactory* factory);
extern void registerUniformVelocityDampingForceField(sofa::core::ObjectFactory* factory);

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
    registerConicalForceField(factory);
    registerConstantForceField(factory);
    registerDiagonalVelocityDampingForceField(factory);
    registerEdgePressureForceField(factory);
    registerEllipsoidForceField(factory);
    registerGravity(factory);
    registerInteractionEllipsoidForceField(factory);
    registerLinearForceField(factory);
    registerOscillatingTorsionPressureForceField(factory);
    registerPlaneForceField(factory);
    registerQuadPressureForceField(factory);
    registerSphereForceField(factory);
    registerSurfacePressureForceField(factory);
    registerTaitSurfacePressureForceField(factory);
    registerTorsionForceField(factory);
    registerTrianglePressureForceField(factory);
    registerUniformVelocityDampingForceField(factory);
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

} // namespace sofa::component::mechanicalload
