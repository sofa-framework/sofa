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

#include <sofa/core/fwd.h>
#include <functional>
#include <vector>
#include <string>
#include <sofa/core/objectmodel/DDGNode.h>
#include <sofa/core/objectmodel/ComponentState.h>
#include <sofa/core/DataTracker.h>

namespace sofa::core
{
/// A DDGNode with trackable input Data (containing a DataTracker)
class SOFA_CORE_API DataTrackerDDGNode : public core::objectmodel::DDGNode
{
public:

    DataTrackerDDGNode() : core::objectmodel::DDGNode() {}

private:
    DataTrackerDDGNode(const DataTrackerDDGNode&);
    void operator=(const DataTrackerDDGNode&);

public:
    /// Create a DataCallback object associated with multiple Data fields.
    void addInputs(std::initializer_list<sofa::core::objectmodel::BaseData*> datas);
    void addOutputs(std::initializer_list<sofa::core::objectmodel::BaseData*> datas);

    /// Set dirty flag to false
    /// for the DDGNode and for all the tracked Data
    virtual void cleanDirty(const core::ExecParams* params = nullptr);


    /// utility function to ensure all inputs are up-to-date
    /// can be useful for particulary complex DDGNode
    /// with a lot input/output imbricated access
    void updateAllInputsIfDirty();

protected:

    /// @name Tracking Data mechanism
    /// each Data added to the DataTracker
    /// is tracked to be able to check if its value changed
    /// since their last clean, called by default
    /// in DataEngine::cleanDirty().
    /// @{

    DataTracker m_dataTracker;

    ///@}

};


///////////////////

/// a DDGNode that automatically triggers its update function
/// when asking for an output and any input changed.
/// Similar behavior than a DataEngine, but this is NOT a component
/// and can be used everywhere.
///
/// Note that it contains a DataTracker (m_dataTracker)
/// to be able to check precisly which input changed if needed.
///
///
///
///
/// **** Implementation good rules: (similar to DataEngine)
///
/// //init
///    addInput // indicate all inputs
///    addOutput // indicate all outputs
///    setDirtyValue(); // the engine must start dirty (of course, no output are up-to-date)
///
///  DataTrackerCallback is usually created using the "addUpdateCallback()" method from Base.
///  Thus the context is usually passed to the lambda making all public & private
///  attributes & methods of the component accessible within the callback function.
///  example:
///
///  addUpdateCallback("name", {&name}, [this](DataTracker& tracker){
///       // Increment the state counter but without changing the state.
///       return d_componentState.getValue();
///  }, {&d_componentState});
///
///  A member function with the same signature - core::objectmodel::ComponentState(DataTracker&) - can
///  also be used.
///
///  The update of the inputs is done for you before calling the callback,
///  and they are also cleaned for you after the call. Thus there's no need
///  to manually call updateAllInputsIfDirty() or cleanDirty() (see implementation of update()
///
class SOFA_CORE_API DataTrackerCallback : public DataTrackerDDGNode
{
public:
    /// set the update function to call
    /// when asking for an output and any input changed.
    void setCallback(std::function<sofa::core::objectmodel::ComponentState(const DataTracker&)> f);

    /// Calls the callback when one of the data has changed.
    void update() override;

    inline void setOwner(sofa::core::objectmodel::Base* owner) { m_owner = owner; }

protected:
    std::function<sofa::core::objectmodel::ComponentState(const DataTracker&)> m_callback;
    sofa::core::objectmodel::Base* m_owner {nullptr};
};


///////////////////////
} // namespace sofa::core

