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
#include <functional>
#include <vector>
#include <sofa/core/config.h>
#include <sofa/core/fwd.h>
#include <sofa/core/objectmodel/DDGNode.h>
namespace sofa::core::objectmodel
{

/// Private namespace declaration, this allows to
/// have in this namespace as much as alias we want and not having
/// them leacking into the public ones.
namespace _datacallback_
{

/// Import the long names into the private namespace so we
/// can write directly Base, BaseData and DDGNode
using sofa::core::objectmodel::Base;
using sofa::core::objectmodel::BaseData;
using sofa::core::objectmodel::DDGNode;

/// Associate to a set of data a set of callback
///
/// The callbacks are called when one of the input is changed.
///
/// Example of use:
///   Data<int> a;
///   Data<int> b;
///   DataCallback cb;
///
///   In constructor:
///   cb.addCallback([&a,&b](DataCallback*){
///                     std::cout << "sum is:" << a.getValue()+b.getValue() << std::endl;
///                   });
///
///   cb.addInputs({&a,&b});  or cb.addInput(&a);  cb.addInput(&b);
///
///   a.setValue(5);       /// should print: "sum is 5"
///   b.setValue(6);       /// should print: "sum is 11"
class SOFA_CORE_API DataCallback : public DDGNode
{
public:
    /// Create a DataCallback object associated with multiple Data.
    void addInputs(std::initializer_list<BaseData*> datas);

    /// Register a new callback function to this DataCallback
    void addCallback(std::function<void(void)>);

    /// The trick is here, this function is called as soon as the input data changes
    /// and can then trigger the callback
    void notifyEndEdit() override ;

    void update() override;

private:
    bool m_updating {false};
    std::vector<std::function<void()>> m_callbacks;
};

} /// namespace _datacallback_

/// Import DataCallback from its private namespace into the public one.
using _datacallback_::DataCallback;

} /// sofa::core::objectmodel


