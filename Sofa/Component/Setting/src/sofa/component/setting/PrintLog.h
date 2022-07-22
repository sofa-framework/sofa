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

#include <sofa/component/setting/config.h>
#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/simulation/MutationListener.h>
#include <sofa/simulation/Node.h>

namespace sofa::component::setting
{

/**
 * A component to link the printLog attributes of all the objects which are contained in the
 * current context. It is a way to log all the components of graph with only one variable. This is
 * useful to debug a scene.
 *
 * The conditions to create the links are:
 * - the printLog attribute of the component must not be linked
 * - the printLog attribute of the component must not be set (isSet method) to false. It is useful
 * to exclude a component if it is too verbose.
 *
 * Example in XML:
 * \code{.xml}
 * <Node name="root">
 *     <PrintLog/>
 *     <ComponentA/>
 *     <Node name="node">
 *         <ComponentB/>
 *     </Node>
 * </Node>
 * \endcode
 *
 * In the example, the printLog attribute of ComponentA and ComponentB is linked to the one in the
 * PrintLog component.
 *
 * The link means:
 * - if the PrintLog component switches its printLog attribute, it also affects the linked components
 * - if the PrintLog component is removed from the graph, the links are broken
 *
 * The PrintLog component creates the links at the initialization stage. But if a new component is
 * added in the graph, the newly component will also get a link to the PrintLog component.
 */
class SOFA_COMPONENT_SETTING_API PrintLog: public core::objectmodel::ConfigurationSetting
{
public:
    SOFA_CLASS(PrintLog, core::objectmodel::ConfigurationSetting);

    void init() override;

    void linkTo(core::Base* base);

protected:

    class NodeInsertionListener final : public simulation::MutationListener
    {
    public:
        void onEndAddObject(simulation::Node *parent, core::objectmodel::BaseObject *object) override;
        NodeInsertionListener(PrintLog& _printLogComponent);
    private:
        PrintLog& printLogComponent;
    } m_nodeListener;

    PrintLog();
    ~PrintLog() override;
};

}
