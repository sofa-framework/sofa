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
#include <sofa/component/setting/PrintLog.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/cast.h>


namespace sofa::component::setting
{
int PrintLogClass = core::RegisterObject("Set the printLog attribute to all components in the same node")
    .add< PrintLog >()
    ;

void PrintLog::linkTo(core::Base* base)
{
    if (!base->f_printLog.getParent() && base->f_printLog.validParent(&this->f_printLog)
        && !(base->f_printLog.isSet() && !base->f_printLog.getValue()))
    {
        base->f_printLog.setParent(&this->f_printLog);
    }
}

void PrintLog::NodeInsertionListener::onEndAddObject(simulation::Node* parent,
    core::objectmodel::BaseObject* object)
{
    printLogComponent.linkTo(object);
}

PrintLog::NodeInsertionListener::NodeInsertionListener(PrintLog& _printLogComponent): printLogComponent(_printLogComponent)
{}

void PrintLog::init()
{
    ConfigurationSetting::init();

    if (auto* context = this->getContext())
    {
        sofa::type::vector<Base*> allBase;
        context->getObjects(allBase, sofa::core::objectmodel::BaseContext::SearchDirection::SearchDown);

        for (auto* base : allBase)
        {
            if (base != this)
            {
                linkTo(base);
            }
        }
    }

    simulation::Node* root = down_cast<simulation::Node>(this->getContext()->getRootContext()->toBaseNode());
    if (root)
    {
        root->addListener(&m_nodeListener);
    }
}

PrintLog::PrintLog()
    : ConfigurationSetting()
    , m_nodeListener(*this)
{
    if (!this->f_printLog.isSet())
    {
        this->f_printLog.setValue(true);
    }
}

PrintLog::~PrintLog()
{
    simulation::Node* root = down_cast<simulation::Node>(this->getContext()->getRootContext()->toBaseNode());
    if (root)
    {
        root->removeListener(&m_nodeListener);
    }
}
}
