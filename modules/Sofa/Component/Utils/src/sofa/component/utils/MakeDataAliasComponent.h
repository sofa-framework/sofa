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
#ifndef SOFA_MAKEDATAALIASCOMPONENT_H
#define SOFA_MAKEDATAALIASCOMPONENT_H

#include <Sofa.Component.Utils.h>

#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Data.h>

#include <string>

namespace sofa
{
namespace component
{
namespace utils
{

/// I use a per-file namespace so that I can employ the 'using' keywords without
/// fearing it will leack names into the global namespace. When closing this namespace
/// selected object from this per-file namespace are then imported into their parent namespace.
/// for ease of use
namespace makedataaliascomponent
{

/// A component to add alias to other components.
class SOFA_COMPONENT_UTILS_API MakeDataAliasComponent : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(MakeDataAliasComponent, core::objectmodel::BaseObject);

    MakeDataAliasComponent() ;
    virtual ~MakeDataAliasComponent(){}

    /// Inherited from BaseObject.
    /// Parse the given description to assign values to this object's fields and
    /// potentially other parameters.
    virtual void parse ( core::objectmodel::BaseObjectDescription* arg ) override;

    Data<std::string>   d_componentname       ; ///< The component class for which to create an alias.
    Data<std::string>   d_dataname            ; ///< The data field for which to create an alias.
    Data<std::string>   d_alias               ; ///< The alias of the data field.

    /// Inherited virtual function from Base
    static std::string className(const MakeDataAliasComponent* ptr)
    {
        SOFA_UNUSED(ptr);
        return "MakeDataAlias" ;
    }

    virtual std::string getClassName() const override
    {
        return "MakeDataAlias" ;
    }

};

}

/// Import the component from the per-file namespace.
using makedataaliascomponent::MakeDataAliasComponent ;

}
}
}
#endif // SOFA_MAKEDATAALIASCOMPONENT_H
