/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <sofa/helper/system/config.h>
#include <PhysicsBasedInteractiveModeler/gui/qt/QMouseOperations.h>
#include <sofa/gui/OperationFactory.h>
#include <sofa/gui/qt/SofaMouseManager.h>

#ifndef WIN32
#define SOFA_EXPORT_DYNAMIC_LIBRARY
#define SOFA_IMPORT_DYNAMIC_LIBRARY
#define SOFA_PIM_API
#else
#ifdef SOFA_BUILD_PIM
#define SOFA_PIM_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_PIM_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif
#endif

namespace sofa
{

namespace component
{

using namespace sofa::gui;
using namespace sofa::gui::qt;
using namespace plugins::pim::gui::qt;

//Here are just several convenient functions to help user to know what contains the plugin

extern "C" {
    SOFA_PIM_API void initExternalModule();
    SOFA_PIM_API const char* getModuleName();
    SOFA_PIM_API const char* getModuleVersion();
    SOFA_PIM_API const char* getModuleLicense();
    SOFA_PIM_API const char* getModuleDescription();
    SOFA_PIM_API const char* getModuleComponentList();
}

void initExternalModule()
{
    RegisterOperation("Sculpt").add< QSculptOperation >();
    SofaMouseManager::getInstance()->LeftOperationCombo->insertItem(QString(OperationFactory::GetDescription("Sculpt").c_str()));
    SofaMouseManager::getInstance()->MiddleOperationCombo->insertItem(QString(OperationFactory::GetDescription("Sculpt").c_str()));
    SofaMouseManager::getInstance()->RightOperationCombo->insertItem(QString(OperationFactory::GetDescription("Sculpt").c_str()));
    SofaMouseManager::getInstance()->getMapIndexOperation().insert(std::make_pair(SofaMouseManager::getInstance()->getMapIndexOperation().size(), "Sculpt"));
}

const char* getModuleName()
{
    return "Plugin PIM";
}

const char* getModuleVersion()
{
    return "beta 1.0";
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "PIM port into SOFA Framework";
}

const char* getModuleComponentList()
{
    return "SculptBodyPerformer, ProgressiveScaling ";
}


SOFA_LINK_CLASS(ProgressiveScaling)
SOFA_LINK_CLASS(ComputeMeshIntersection)

}

}
