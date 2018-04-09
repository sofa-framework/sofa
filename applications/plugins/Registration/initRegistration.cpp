/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <Registration/config.h>

namespace sofa
{

namespace component
{

  //Here are just several convenient functions to help user to know what contains the plugin

  extern "C" {
    SOFA_REGISTRATION_API void initExternalModule();
    SOFA_REGISTRATION_API const char* getModuleName();
    SOFA_REGISTRATION_API const char* getModuleVersion();
    SOFA_REGISTRATION_API const char* getModuleLicense();
    SOFA_REGISTRATION_API const char* getModuleDescription();
    SOFA_REGISTRATION_API const char* getModuleComponentList();
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
    return "Registration Plugin";
  }

  const char* getModuleVersion()
  {
    return "0.1";
  }

  const char* getModuleLicense()
  {
    return "QPL";
  }


  const char* getModuleDescription()
  {
    return "Model/image registration force fields";
  }

  const char* getModuleComponentList()
  {
    return "ImageRegistrationLoader, IntensityProfileRegistrationForceField, RegistrationContact, RegistrationContactForceField, ClosestPointRegistrationForceField, RegistrationExporter";
  }

} // namespace registration

} // namespace sofa

////////// BEGIN CLASS LIST //////////
SOFA_LINK_CLASS(RegistrationContact)
SOFA_LINK_CLASS(RegistrationContactForceField)
SOFA_LINK_CLASS(ClosestPointRegistrationForceField)
SOFA_LINK_CLASS(RegistrationExporter)
SOFA_LINK_CLASS(GroupwiseRegistrationEngine)
#ifdef SOFA_HAVE_IMAGE
    SOFA_LINK_CLASS(IntensityProfileRegistrationForceField)
#endif
