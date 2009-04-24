/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_VOIDMAPPING_H
#define SOFA_COMPONENT_MAPPING_VOIDMAPPING_H

#include <sofa/core/componentmodel/behavior/BaseMechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>
#include <sofa/core/BaseMapping.h>
#include <vector>

namespace sofa
{

namespace component
{

namespace mapping
{

class VoidMapping : public sofa::core::componentmodel::behavior::BaseMechanicalMapping, public sofa::core::BaseMapping
{
public:
    typedef sofa::core::componentmodel::behavior::BaseMechanicalMapping Inherit;
    typedef sofa::core::componentmodel::behavior::BaseMechanicalState In;
    typedef sofa::core::componentmodel::behavior::BaseMechanicalState Out;

    Data<bool> f_isMechanical;

    VoidMapping()
        : f_isMechanical( initData( &f_isMechanical, true, "isMechanical", "set to false if this mapping should only be used as a regular mapping instead of a mechanical mapping" ) )
        , fromModel(NULL), toModel(NULL)
    {
    }

    virtual ~VoidMapping()
    {
    }

    void init()
    {
        fromModel = dynamic_cast<In*>(this->getContext()->getMechanicalState());
        toModel = dynamic_cast<Out*>(this->getContext()->getMechanicalState());
    }

    /// Apply the transformation from the input model to the output model (like apply displacement from BehaviorModel to VisualModel)
    virtual void updateMapping()
    {
    }

    /// Accessor to the input model of this mapping
    virtual sofa::core::objectmodel::BaseObject* getFrom()
    {
        return fromModel;
    }

    /// Accessor to the output model of this mapping
    virtual sofa::core::objectmodel::BaseObject* getTo()
    {
        return toModel;
    }

    /// Disable the mapping to get the original coordinates of the mapped model.
    virtual void disable()
    {
    }

    /// Get the source (upper) model.
    virtual sofa::core::componentmodel::behavior::BaseMechanicalState* getMechFrom()
    {
        return fromModel;
    }

    /// Get the destination (lower, mapped) model.
    virtual sofa::core::componentmodel::behavior::BaseMechanicalState* getMechTo()
    {
        return toModel;
    }

    /// Return false if this mapping should only be used as a regular mapping instead of a mechanical mapping.
    bool isMechanical()
    {
        return this->f_isMechanical.getValue();
    }

    /// Determine if this mapping should only be used as a regular mapping instead of a mechanical mapping.
    void setMechanical(bool b)
    {
        this->f_isMechanical.setValue(b);
    }

    /// Propagate position from the source model to the destination model.
    ///
    /// If the MechanicalMapping can be represented as a matrix J, this method computes
    /// $ x_out = J x_in $
    virtual void propagateX()
    {
    }

    /// Propagate free-motion position from the source model to the destination model.
    virtual void propagateXfree()
    {
    }

    /// Propagate velocity from the source model to the destination model.
    virtual void propagateV()
    {
    }

protected:
    In* fromModel;
    Out* toModel;
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
