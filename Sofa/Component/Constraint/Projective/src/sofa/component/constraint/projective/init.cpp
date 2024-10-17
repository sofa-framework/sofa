/******************************************************************************
*                 SOFA, Simulatconstraintn Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundatconstraintn; either versconstraintn 2.1 of the License, or (at     *
* your optconstraintn) any later versconstraintn.                                             *
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
* Contact informatconstraintn: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/constraint/projective/init.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>

namespace sofa::component::constraint::projective
{

extern void registerAffineMovementProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerAttachProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerDirectionProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerFixedPlaneProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerFixedProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerFixedRotationProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerFixedTranslationProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerHermiteSplineProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerLinearMovementProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerLinearVelocityProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerLineProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerOscillatorProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerParabolicProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerPartialFixedProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerPartialLinearMovementProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerPatchTestMovementProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerPlaneProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerPointProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerPositionBasedDynamicsProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerSkeletalMotionProjectiveConstraint(sofa::core::ObjectFactory* factory);

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
    registerAffineMovementProjectiveConstraint(factory);
    registerAttachProjectiveConstraint(factory);
    registerDirectionProjectiveConstraint(factory);
    registerFixedPlaneProjectiveConstraint(factory);
    registerFixedProjectiveConstraint(factory);
    registerFixedRotationProjectiveConstraint(factory);
    registerFixedTranslationProjectiveConstraint(factory);
    registerHermiteSplineProjectiveConstraint(factory);
    registerLinearMovementProjectiveConstraint(factory);
    registerLinearVelocityProjectiveConstraint(factory);
    registerLineProjectiveConstraint(factory);
    registerOscillatorProjectiveConstraint(factory);
    registerParabolicProjectiveConstraint(factory);
    registerPartialFixedProjectiveConstraint(factory);
    registerPartialLinearMovementProjectiveConstraint(factory);
    registerPatchTestMovementProjectiveConstraint(factory);
    registerPlaneProjectiveConstraint(factory);
    registerPointProjectiveConstraint(factory);
    registerPositionBasedDynamicsProjectiveConstraint(factory);
    registerSkeletalMotionProjectiveConstraint(factory);
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

} // namespace sofa::component::constraint::projective
