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
#ifndef SOFA_SIMULATION_CORE_VECTOROPERATIONS_H
#define SOFA_SIMULATION_CORE_VECTOROPERATIONS_H

#include <sofa/core/behavior/BaseVectorOperations.h>
#include <sofa/simulation/config.h>
#include <sofa/simulation/VisitorExecuteFunc.h>

namespace sofa
{

namespace core
{
class ExecParams;
}

namespace simulation::common
{

class SOFA_SIMULATION_CORE_API VectorOperations : public sofa::core::behavior::BaseVectorOperations
{
public:

    VectorOperations(const sofa::core::ExecParams* params, sofa::core::objectmodel::BaseContext* ctx, bool precomputedTraversalOrder=false);

    /// Allocate a temporary vector
    void v_alloc(sofa::core::MultiVecCoordId& v, const core::VecIdProperties& properties = {}) override;
    void v_alloc(sofa::core::MultiVecDerivId& v, const core::VecIdProperties& properties = {}) override;
    /// Free a previously allocated temporary vector
    void v_free(sofa::core::MultiVecCoordId& id, bool interactionForceField=false, bool propagate=false) override;
    void v_free(sofa::core::MultiVecDerivId& id, bool interactionForceField=false, bool propagate=false) override;

    void v_realloc(sofa::core::MultiVecCoordId& id, bool interactionForceField=false, bool propagate=false, const core::VecIdProperties& properties = {}) override;
    void v_realloc(sofa::core::MultiVecDerivId& id, bool interactionForceField=false, bool propagate=false, const core::VecIdProperties& properties = {}) override;

    void v_clear(core::MultiVecId v) override; ///< v=0
    void v_eq(core::MultiVecId v, core::ConstMultiVecId a) override; ///< v=a
    void v_eq(core::MultiVecId v, core::ConstMultiVecId a, SReal f) override; ///< v=f*a
    void v_peq(core::MultiVecId v, core::ConstMultiVecId a, SReal f=1.0) override; ///< v+=f*a

    void v_teq(core::MultiVecId v, SReal f) override ; ///< v*=f
    void v_op(core::MultiVecId v, core::ConstMultiVecId a, core::ConstMultiVecId  b, SReal f=1.0) override ; ///< v=a+b*f
    void v_multiop(const core::behavior::BaseMechanicalState::VMultiOp& o) override;
    void v_dot(core::ConstMultiVecId a, core::ConstMultiVecId  b) override; ///< a dot b ( get result using finish )
    void v_norm(core::ConstMultiVecId a, unsigned l) override; ///< Compute the norm of a vector ( get result using finish ). The type of norm is set by parameter l. Use 0 for the infinite norm. Note that the 2-norm is more efficiently computed using the square root of the dot product.
    void v_threshold(core::MultiVecId a, SReal threshold) override; ///< nullify the values below the given threshold

    SReal finish() override;
    void print(sofa::core::ConstMultiVecId v, std::ostream& out, std::string prefix="", std::string suffix="" ) override;

    size_t v_size(core::MultiVecId v) override;

protected:
    VisitorExecuteFunc executeVisitor;
    /// Result of latest v_dot operation
    SReal result;

};

}
}


#endif //SOFA_SIMULATION_CORE_VECTOROPERATIONS_H
