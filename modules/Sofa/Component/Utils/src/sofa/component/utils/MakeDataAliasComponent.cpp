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
#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject ;
using sofa::core::ObjectFactory ;

using sofa::core::objectmodel::ComponentState ;

#include <sofa/component/utils/MakeDataAliasComponent.h>

using std::string;

namespace sofa
{
namespace component
{
namespace utils
{
namespace makedataaliascomponent
{

MakeDataAliasComponent::MakeDataAliasComponent() :
   d_componentname(initData(&d_componentname, "componentname", "The component class for which to create an alias."))
  ,d_dataname(initData(&d_dataname, "dataname", "The data field for which to create an alias."))
  ,d_alias(initData(&d_alias, "alias", "The alias of the data field."))
{
    m_componentstate = ComponentState::Invalid ;
}

void MakeDataAliasComponent::parse ( core::objectmodel::BaseObjectDescription* arg )
{
    BaseObject::parse(arg) ;

    const char* component=arg->getAttribute("componentname") ;
    if(component==nullptr)
    {
        msg_error(this) << "The mandatory 'componentname' attribute is missing.  "
                           "The component is disabled.  "
                           "To remove this error message you need to add a targetcomponent attribute pointing to a valid component's ClassName.";
        return ;
    }
    string scomponent(component) ;


    const char* dataname=arg->getAttribute("dataname") ;
    if(dataname==nullptr)
    {
        msg_error(this) << "The mandatory 'dataname' attribute is missing.  "
                           "The component is disabled.  "
                           "To remove this error message you need to add a targetcomponent attribute pointing to a valid component's ClassName.";
        return ;
    }
    string sdataname(dataname) ;


    const char* alias=arg->getAttribute("alias") ;
    if(alias==nullptr)
    {
        msg_error(this) << "The mandatory 'alias' attribute is missing.  "
                           "The component is disabled.  "
                           "To remove this error message you need to add an alias attribute with a valid string component's ClassName.";
        return ;
    }
    string salias(alias);

    if(!ObjectFactory::getInstance()->hasCreator(scomponent)){
        msg_error(this) << "The value '"<< scomponent << "' for 'componentname' does not correspond to a valid name.  "
                           "The component is disabled.  "
                           "To remove this error message you need to add a targetcomponent attribute pointing to a valid component's ClassName.";
        return ;
    }

    ObjectFactory::ClassEntry& creatorentry = ObjectFactory::getInstance()->getEntry(scomponent);
    if(creatorentry.m_dataAlias.find(dataname) != creatorentry.m_dataAlias.end()){
        creatorentry.m_dataAlias[dataname] = std::vector<std::string>();
    }
    creatorentry.m_dataAlias[dataname].push_back(salias) ;

    m_componentstate = ComponentState::Valid ;
}

SOFA_DECL_CLASS(MakeDataAliasComponent)

int MakeDataAliasComponentClass = RegisterObject("This object create an alias to a data field. ")
        .add< MakeDataAliasComponent >()
        ;

}
}
}
}


