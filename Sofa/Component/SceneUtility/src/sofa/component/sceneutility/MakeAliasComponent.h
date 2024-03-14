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
#pragma once

#include <sofa/component/sceneutility/config.h>

#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Data.h>

#include <string>

namespace sofa::component::sceneutility::makealiascomponent
{

/// I use a per-file namespace so that I can employ the 'using' keywords without
/// fearing it will leack names into the global namespace. When closing this namespace
/// selected object from this per-file namespace are then imported into their parent namespace.
/// for ease of use
/// 
/// A component to add alias to other components.
class SOFA_COMPONENT_SCENEUTILITY_API MakeAliasComponent : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(MakeAliasComponent, core::objectmodel::BaseObject);

    MakeAliasComponent() ;
    ~MakeAliasComponent() override{}

    /// Inherited from BaseObject.
    /// Parse the given description to assign values to this object's fields and
    /// potentially other parameters.
    void parse ( core::objectmodel::BaseObjectDescription* arg ) override;

    Data<std::string>   d_targetcomponent       ; ///< The component class for which to create an alias.
    Data<std::string>   d_alias                 ; ///< The new alias of the component.

    /// Returns the sofa class name. By default the name of the c++ class is exposed... but
    /// Here we want it to be MakeAlias so we need to customize it.
    /// More details on the name customization infrastructure is in NameDecoder.h
    static std::string GetCustomClassName()
    {
        return "MakeAlias" ;
    }
};

} // namespace sofa::component::sceneutility::makealiascomponent

namespace sofa::component::sceneutility
{
/// Import the component from the per-file namespace.
using makealiascomponent::MakeAliasComponent ;

} // namespace sofa::component::sceneutility
