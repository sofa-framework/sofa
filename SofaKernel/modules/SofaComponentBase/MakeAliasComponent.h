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
#ifndef SOFA_MAKEALIASCOMPONENT_H
#define SOFA_MAKEALIASCOMPONENT_H

#include "config.h"

#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Data.h>

#include <string>

namespace sofa
{
namespace component
{

/// I use a per-file namespace so that I can employ the 'using' keywords without
/// fearing it will leack names into the global namespace. When closing this namespace
/// selected object from this per-file namespace are then imported into their parent namespace.
/// for ease of use
namespace makealiascomponent
{

/// A component to add alias to other components.
class SOFA_COMPONENT_BASE_API MakeAliasComponent : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(MakeAliasComponent, core::objectmodel::BaseObject);

    MakeAliasComponent() ;
    virtual ~MakeAliasComponent(){}

    /// Inherited from BaseObject.
    /// Parse the given description to assign values to this object's fields and
    /// potentially other parameters.
    virtual void parse ( core::objectmodel::BaseObjectDescription* arg ) override;

    Data<std::string>   d_targetcomponent       ;
    Data<std::string>   d_alias                 ;


    static std::string className(const MakeAliasComponent* ptr)
    {
        SOFA_UNUSED(ptr);
        return "MakeAlias" ;
    }

    virtual std::string getClassName() const override
    {
        return "MakeAlias" ;
    }


};

}

/// Import the component from the per-file namespace.
using makealiascomponent::MakeAliasComponent ;

}
}
#endif // SOFA_AliasComponent_H
