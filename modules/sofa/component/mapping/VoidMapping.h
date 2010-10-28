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

#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/helper/vector.h>

namespace sofa
{

namespace component
{

namespace mapping
{

class VoidMapping : public sofa::core::BaseMapping
{
public:
    SOFA_CLASS(VoidMapping, sofa::core::BaseMapping);

    typedef sofa::core::BaseMapping Inherit;
    typedef sofa::core::behavior::BaseMechanicalState In;
    typedef sofa::core::behavior::BaseMechanicalState Out;

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

    /// Accessor to the input model of this mapping
    virtual  helper::vector<core::BaseState*> getFrom()
    {
        helper::vector<core::BaseState*> vec(1,fromModel);
        return vec;
    }

    /// Accessor to the output model of this mapping
    virtual helper::vector<core::BaseState*> getTo()
    {
        helper::vector<core::BaseState*> vec(1,toModel);
        return vec;
    }

    /// Disable the mapping to get the original coordinates of the mapped model.
    virtual void disable()
    {
    }

    /// Get the source (upper) model.
    virtual helper::vector<sofa::core::behavior::BaseMechanicalState*> getMechFrom()
    {
        helper::vector<sofa::core::behavior::BaseMechanicalState*> vec(1, fromModel);
        return vec;
    }

    /// Get the destination (lower, mapped) model.
    virtual helper::vector<sofa::core::behavior::BaseMechanicalState*> getMechTo()
    {
        helper::vector<sofa::core::behavior::BaseMechanicalState*> vec(1, toModel);
        return vec;
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

    virtual void apply (core::MultiVecCoordId /* outPos */, core::ConstMultiVecCoordId /* inPos */, const core::MechanicalParams* /* mparams = core::MechanicalParams::defaultInstance() */)
    {
    }

    virtual void applyJ(core::MultiVecDerivId /* outVel */, core::ConstMultiVecDerivId /* inVel */, const core::MechanicalParams* /* mparams = core::MechanicalParams::defaultInstance() */)
    {
    }

    virtual void applyJT(core::MultiVecDerivId /* inForce */, core::ConstMultiVecDerivId /* outForce */, const core::MechanicalParams* /* mparams = core::MechanicalParams::defaultInstance() */)
    {
    }

    virtual void applyJT(core::MultiMatrixDerivId /* inConst */, core::ConstMultiMatrixDerivId /* outConst */, const core::ConstraintParams * /*cparams*/)
    {
    }

    virtual void computeAccFromMapping(core::MultiVecDerivId /* outAcc */, core::ConstMultiVecDerivId /* inVel */, core::ConstMultiVecDerivId /* inAcc */, const core::MechanicalParams* /*mparams = core::MechanicalParams::defaultInstance() */)
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
