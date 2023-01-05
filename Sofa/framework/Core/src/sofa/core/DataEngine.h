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

#include <sofa/core/config.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/DataTracker.h>

namespace sofa::core
{

/**
 *  \brief from a set of Data inputs computes a set of Data outputs
 *
 * Implementation good rules:
 *
 * void init()
 * {
 *    addInput // indicate all inputs
 *    addOutput // indicate all outputs
 * }
 *
 * // optional (called each time a data is modified in the gui)
 * // it is not always desired
 * void reinit()
 * {
 *    update();
 * }
 *
 * void doUpdate() override
 * {
 *    access your inputs, set your outputs...
 * }
 *
 */
class SOFA_CORE_API DataEngine : public core::DataTrackerDDGNode, public virtual core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(DataEngine, core::objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(DataEngine)
protected:
    /// Constructor
    DataEngine();

    /// Destructor. Do nothing
    ~DataEngine() override;

private:
	DataEngine(const DataEngine& n) ;
	DataEngine& operator=(const DataEngine& n) ;

    /// Called in update(), back-propagates the data update
    /// in the data dependency graph
    void updateAllInputs();

protected:
    /// Where you put your engine's impl
    virtual void doUpdate() = 0;

    /// Prevent engines to use the internalUpdate mechanism, so that only update/doUpdate is used
    virtual void doInternalUpdate() final {}

public:
    /// Updates your inputs and calls cleanDirty() for you.
    /// User implementation moved to doUpdate()
    void update() final;

    /// Add a new input to this engine
    /// Automatically adds the input fields to the datatracker
    void addInput(sofa::core::objectmodel::BaseData* data);

    /// Add a new output to this engine
    void addOutput(objectmodel::BaseData* n);
};
} // namespace sofa
