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
#include <sofa/component/sceneutility/config.h>
#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject ;
using sofa::core::ObjectFactory ;

using sofa::core::objectmodel::ComponentState ;

#include <sofa/component/sceneutility/MakeAliasComponent.h>

using std::string;

namespace sofa::component::sceneutility::makealiascomponent
{

MakeAliasComponent::MakeAliasComponent() :
   d_targetcomponent(initData(&d_targetcomponent, "targetcomponent", "The component class for which to create an alias."))
  ,d_alias(initData(&d_alias, "alias", "The new alias of the component."))
{
    d_componentState.setValue(ComponentState::Invalid) ;
}

void MakeAliasComponent::parse ( core::objectmodel::BaseObjectDescription* arg )
{
    BaseObject::parse(arg) ;

    const char* target=arg->getAttribute("targetcomponent") ;
    if(target==nullptr)
    {
        msg_error(this) << "The mandatory 'targetcomponent' attribute is missing.  "
                           "The component is disabled.  "
                           "To remove this error message you need to add a targetcomponent attribute pointing to a valid component's ClassName.";
        return ;
    }
    const string starget(target) ;

    const char* alias=arg->getAttribute("alias") ;
    if(alias==nullptr)
    {
        msg_error(this) << "The mandatory 'alias' attribute is missing.  "
                           "The component is disabled.  "
                           "To remove this error message you need to add an alias attribute with a valid string component's ClassName.";
        return ;
    }
    const string salias(alias);

    if(!ObjectFactory::getInstance()->hasCreator(starget))
    {
        msg_error(this) << "The provided attribute 'targetcomponent= "<< starget << "' does not correspond to a valid component ClassName  "
                           "The component is disabled.  "
                           "To remove this error message you need to fix your scene and provide a valid component ClassName in the 'targetcomponent' attribute. ";
        return ;
    }

    ObjectFactory::getInstance()->addAlias(salias, starget);

    d_componentState.setValue(ComponentState::Valid) ;
}

int MakeAliasComponentClass = RegisterObject("This object create an alias to a component name to make the scene more readable. ")
        .add< MakeAliasComponent >()
        ;

} // namespace sofa::component::sceneutility::makealiascomponent
