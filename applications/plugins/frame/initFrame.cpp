/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/helper/system/config.h>
#include "initFrame.h"

namespace sofa
{

namespace component
{

//Here are just several convenient functions to help user to know what contains the plugin

extern "C" {
    SOFA_FRAME_API void initExternalModule();
    SOFA_FRAME_API const char* getModuleName();
    SOFA_FRAME_API const char* getModuleVersion();
    SOFA_FRAME_API const char* getModuleLicense();
    SOFA_FRAME_API const char* getModuleDescription();
    SOFA_FRAME_API const char* getModuleComponentList();
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

const char* getModuleName()
{
    return "Frame Based Dynamic Plugin";
}

const char* getModuleVersion()
{
    return "0.1";
}

const char* getModuleLicense()
{
    return "LGPL";
}


const char* getModuleDescription()
{
    return "Use frame based dynamic technic in SOFA";
}

const char* getModuleComponentList()
{
    return "FrameDiagonalMass, FixedConstraint, FrameHookeForceField, MechanicalObject, FrameSpringForceField2";
}

} // namespace frame

} // namespace sofa

////////// BEGIN CLASS LIST //////////
SOFA_LINK_CLASS(FrameBlendingMapping)
//SOFA_LINK_CLASS(AffineSkinningMapping)
SOFA_LINK_CLASS(FrameDiagonalMass)
SOFA_LINK_CLASS(FrameConstantForceField)
//SOFA_LINK_CLASS(FrameDualQuatSkinningMapping)
SOFA_LINK_CLASS(FrameFixedConstraint)
//SOFA_LINK_CLASS(FrameHookeForceField)
SOFA_LINK_CLASS(FrameLoydAlgo)
SOFA_LINK_CLASS(FrameMechanicalObject)
//SOFA_LINK_CLASS(FrameSpringForceField2)
SOFA_LINK_CLASS(HookeMaterial3)
SOFA_LINK_CLASS(GridMaterial)
//SOFA_LINK_CLASS(PrimitiveSkinningMapping)
//SOFA_LINK_CLASS(QuadraticSkinningMapping)
//SOFA_LINK_CLASS(RigidSkinningMapping)
