/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/system/config.h>

#include "ContinuousContact.h"

namespace sofa
{

namespace component
{

//Here are just several convenient functions to help user to know what contains the plugin

extern "C" {
    SOFA_CONTINUOUSCONTACT_API void initExternalModule();
    SOFA_CONTINUOUSCONTACT_API const char* getModuleName();
    SOFA_CONTINUOUSCONTACT_API const char* getModuleVersion();
    SOFA_CONTINUOUSCONTACT_API const char* getModuleLicense();
    SOFA_CONTINUOUSCONTACT_API const char* getModuleDescription();
    SOFA_CONTINUOUSCONTACT_API const char* getModuleComponentList();
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
    return "Plugin Continuous Contact";
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
    return "Continuous contact management for friction contacts";
}

const char* getModuleComponentList()
{
    return "ContinuousContactBarycentricMapping, ContinuousFrictionContact, ContinuousContactRigidMapping, ContinuousUnilateralInteractionConstraint";
}

} // namespace component

} // namespace sofa


SOFA_LINK_CLASS(ContinuousContactBarycentricMapping)
SOFA_LINK_CLASS(ContinuousContactRigidMapping)
SOFA_LINK_CLASS(ContinuousFrictionContact)
SOFA_LINK_CLASS(ContinuousUnilateralInteractionConstraint)

