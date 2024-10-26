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
#include <sofa/component/engine/transform/init.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>

namespace sofa::component::engine::transform
{
    
extern void registerDifferenceEngine(sofa::core::ObjectFactory* factory);
extern void registerDilateEngine(sofa::core::ObjectFactory* factory);
extern void registerDisplacementTransformEngine(sofa::core::ObjectFactory* factory);
extern void registerIndexValueMapper(sofa::core::ObjectFactory* factory);
extern void registerIndices2ValuesMapper(sofa::core::ObjectFactory* factory);
extern void registerMapIndices(sofa::core::ObjectFactory* factory);
extern void registerMathOp(sofa::core::ObjectFactory* factory);
extern void registerProjectiveTransformEngine(sofa::core::ObjectFactory* factory);
extern void registerQuatToRigidEngine(sofa::core::ObjectFactory* factory);
extern void registerRigidToQuatEngine(sofa::core::ObjectFactory* factory);
extern void registerROIValueMapper(sofa::core::ObjectFactory* factory);
extern void registerSmoothMeshEngine(sofa::core::ObjectFactory* factory);
extern void registerTransformEngine(sofa::core::ObjectFactory* factory);
extern void registerTranslateTransformMatrixEngine(sofa::core::ObjectFactory* factory);
extern void registerInvertTransformMatrixEngine(sofa::core::ObjectFactory* factory);
extern void registerScaleTransformMatrixEngine(sofa::core::ObjectFactory* factory);
extern void registerRotateTransformMatrixEngine(sofa::core::ObjectFactory* factory);
extern void registerTransformPosition(sofa::core::ObjectFactory* factory);
extern void registerVertex2Frame(sofa::core::ObjectFactory* factory);


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
    registerDifferenceEngine(factory);
    registerDilateEngine(factory);
    registerDisplacementTransformEngine(factory);
    registerIndexValueMapper(factory);
    registerIndices2ValuesMapper(factory);
    registerMapIndices(factory);
    registerMathOp(factory);
    registerProjectiveTransformEngine(factory);
    registerQuatToRigidEngine(factory);
    registerRigidToQuatEngine(factory);
    registerROIValueMapper(factory);
    registerSmoothMeshEngine(factory);
    registerTransformEngine(factory);
    registerTranslateTransformMatrixEngine(factory);
    registerInvertTransformMatrixEngine(factory);
    registerScaleTransformMatrixEngine(factory);
    registerRotateTransformMatrixEngine(factory);
    registerTransformPosition(factory);
    registerVertex2Frame(factory);
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

} // namespace sofa::component::engine::transform
