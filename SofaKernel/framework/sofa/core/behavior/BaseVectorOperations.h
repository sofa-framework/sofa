/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_BEHAVIOR_BASEVECTOROPERATION_H
#define SOFA_CORE_BEHAVIOR_BASEVECTOROPERATION_H

#include <sofa/core/MultiVecId.h>
#include <sofa/core/ExecParams.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/BaseMechanicalState.h>

namespace sofa
{

namespace core
{

namespace behavior
{

class BaseVectorOperations
{
protected:
    const core::ExecParams* params;
    core::objectmodel::BaseContext* ctx;
    SReal result;

public:
    BaseVectorOperations(const core::ExecParams* params, core::objectmodel::BaseContext* ctx)
        : params(params),ctx(ctx)
    {}

    virtual ~BaseVectorOperations()
    {}

    /// Allocate a temporary vector
    virtual void v_alloc(sofa::core::MultiVecCoordId& id) = 0;
    virtual void v_alloc(sofa::core::MultiVecDerivId& id) = 0;
    /// Free a previously allocated temporary vector
    virtual void v_free(sofa::core::MultiVecCoordId& id, bool interactionForceField=false, bool propagate=false) = 0;
    virtual void v_free(sofa::core::MultiVecDerivId& id, bool interactionForceField=false, bool propagate=false) = 0;

    /// keep already allocated vectors and allocates others. If interactionForceField, also allocates mechanical states linked by an InteractionForceField
    virtual void v_realloc(sofa::core::MultiVecCoordId& id, bool interactionForceField=false, bool propagate=false) = 0;
    virtual void v_realloc(sofa::core::MultiVecDerivId& id, bool interactionForceField=false, bool propagate=false) = 0;

    virtual void v_clear(core::MultiVecId v) = 0; ///< v=0
    virtual void v_eq(core::MultiVecId v, core::ConstMultiVecId a) = 0; ///< v=a
    virtual void v_eq(core::MultiVecId v, core::ConstMultiVecId a, SReal f) = 0; ///< v=f*a
    virtual void v_peq(core::MultiVecId v, core::ConstMultiVecId a, SReal f=1.0) = 0; ///< v+=f*a
    virtual void v_teq(core::MultiVecId v, SReal f) = 0; ///< v*=f
    virtual void v_op(core::MultiVecId v, core::ConstMultiVecId a, core::ConstMultiVecId b, SReal f=1.0) = 0; ///< v=a+b*f
    virtual void v_multiop(const core::behavior::BaseMechanicalState::VMultiOp& o) = 0;
    virtual void v_dot(core::ConstMultiVecId a, core::ConstMultiVecId b) = 0; ///< a dot b ( get result using finish )
    virtual void v_norm(core::ConstMultiVecId a, unsigned l)=0; ///< Compute the norm of a vector ( get result using finish ). The type of norm is set by parameter l. Use 0 for the infinite norm. Note that the 2-norm is more efficiently computed using the square root of the dot product.
    virtual void v_threshold(core::MultiVecId a, SReal threshold) = 0; ///< nullify the values below the given threshold


    virtual SReal finish() = 0;

    virtual void print( core::ConstMultiVecId v, std::ostream& out, std::string prefix="", std::string suffix="" ) = 0;

    virtual size_t v_size(core::MultiVecId v) = 0;

};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif //SOFA_CORE_BEHAVIOR_BASEVECTOROPERATION_H

