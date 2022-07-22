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
#include <sofa/core/fwd.h>
#include <vector>

namespace sofa::core::objectmodel
{

class DDGNode;

/**
 *  \brief A DDGNode is a vertex in the data dependencies graph.
 * The data dependency graph is used to update the data when
 * some of other changes and it is at the root of the implementation
 * of the data update mecanisme as well as DataEngines.
 */
class SOFA_CORE_API DDGNode
{
public:
    typedef std::vector<DDGNode*> DDGLinkContainer;
    typedef DDGLinkContainer::const_iterator DDGLinkIterator;

    /// Constructor
    DDGNode();

    /// Destructor. Automatically remove remaining links
    virtual ~DDGNode();

    /// Add a new input to this node
    void addInput(DDGNode* n);

    /// Remove an input from this node
    void delInput(DDGNode* n);

    /// Add a new output to this node
    void addOutput(DDGNode* n);

    /// Remove an output from this node
    void delOutput(DDGNode* n);

    /// Get the list of inputs for this DDGNode
    const DDGLinkContainer& getInputs();

    /// Get the list of outputs for this DDGNode
    const DDGLinkContainer& getOutputs();

    /// Update this value
    virtual void update() = 0;

    /// Returns true if the DDGNode needs to be updated
    bool isDirty() const { return dirtyFlags.dirtyValue; }

    /// Indicate the value needs to be updated
    virtual void setDirtyValue();

    /// Indicate the outputs needs to be updated. This method must be called after changing the value of this node.
    virtual void setDirtyOutputs();

    /// Set dirty flag to false
    void cleanDirty();

    /// Notify links that the DGNode has been modified
    virtual void notifyEndEdit();

    /// Utility method to call update if necessary. This method should be called before reading of writing the value of this node.
    void updateIfDirty() const;

protected:
    DDGLinkContainer inputs;
    DDGLinkContainer outputs;

    virtual void doAddInput(DDGNode* n);
    virtual void doDelInput(DDGNode* n);
    virtual void doAddOutput(DDGNode* n);
    virtual void doDelOutput(DDGNode* n);

    /// the dirtyOutputs flags of all the inputs will be set to false
    void cleanDirtyOutputsOfInputs();

private:

    struct DirtyFlags
    {
        bool dirtyValue {false};
        bool dirtyOutputs {false};
    };
    DirtyFlags dirtyFlags;
};

} // namespace sofa::core::objectmodel

