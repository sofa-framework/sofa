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
#ifndef SOFA_COMPONENT_MAPPING_VOIDMAPPING_H
#define SOFA_COMPONENT_MAPPING_VOIDMAPPING_H
#include "config.h"

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

    void init() override
    {
        fromModel = dynamic_cast<In*>(this->getContext()->getMechanicalState());
        toModel = dynamic_cast<Out*>(this->getContext()->getMechanicalState());
    }

    /// Accessor to the input model of this mapping
    virtual  helper::vector<core::BaseState*> getFrom() override
    {
        helper::vector<core::BaseState*> vec(1,fromModel);
        return vec;
    }

    /// Accessor to the output model of this mapping
    virtual helper::vector<core::BaseState*> getTo() override
    {
        helper::vector<core::BaseState*> vec(1,toModel);
        return vec;
    }

    /// Disable the mapping to get the original coordinates of the mapped model.
    virtual void disable() override
    {
    }

    /// Get the source (upper) model.
    virtual helper::vector<sofa::core::behavior::BaseMechanicalState*> getMechFrom() override
    {
        helper::vector<sofa::core::behavior::BaseMechanicalState*> vec(1, fromModel);
        return vec;
    }

    /// Get the destination (lower, mapped) model.
    virtual helper::vector<sofa::core::behavior::BaseMechanicalState*> getMechTo() override
    {
        helper::vector<sofa::core::behavior::BaseMechanicalState*> vec(1, toModel);
        return vec;
    }

    virtual void apply (const core::MechanicalParams* /* mparams = core::MechanicalParams::defaultInstance() */, core::MultiVecCoordId /* outPos */, core::ConstMultiVecCoordId /* inPos */) override
    {
    }

    virtual void applyJ(const core::MechanicalParams* /* mparams = core::MechanicalParams::defaultInstance() */, core::MultiVecDerivId /* outVel */, core::ConstMultiVecDerivId /* inVel */) override
    {
    }

    virtual void applyJT(const core::MechanicalParams* /* mparams = core::MechanicalParams::defaultInstance() */, core::MultiVecDerivId /* inForce */, core::ConstMultiVecDerivId /* outForce */) override
    {
    }

    virtual void applyDJT(const core::MechanicalParams* /*mparams*/, core::MultiVecDerivId /*inForce*/, core::ConstMultiVecDerivId /*outForce*/) override {}

    virtual void applyJT(const core::ConstraintParams * /*cparams*/, core::MultiMatrixDerivId /* inConst */, core::ConstMultiMatrixDerivId /* outConst */) override
    {
    }

    virtual void computeAccFromMapping(const core::MechanicalParams* /*mparams = core::MechanicalParams::defaultInstance() */, core::MultiVecDerivId /* outAcc */, core::ConstMultiVecDerivId /* inVel */, core::ConstMultiVecDerivId /* inAcc */) override
    {
    }

protected:
    In* fromModel;
    Out* toModel;

    VoidMapping():Inherit(),fromModel(NULL),toModel(NULL)
    {
        this->f_mapForces.setValue(false);
        this->f_mapConstraints.setValue(false);
        this->f_mapMasses.setValue(false);
    }

    virtual ~VoidMapping()
    {
    }

    virtual void updateForceMask() override { fromModel->forceMask.assign(fromModel->getSize(),true); }
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
