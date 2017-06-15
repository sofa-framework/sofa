/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_SIMULATION_CORE_VECTOROPERATIONS_H
#define SOFA_SIMULATION_CORE_VECTOROPERATIONS_H

#include <sofa/core/behavior/BaseVectorOperations.h>
#include <sofa/simulation/simulationcore.h>
#include <sofa/simulation/VisitorExecuteFunc.h>

namespace sofa
{

namespace core
{
class ExecParams;
}

namespace simulation
{

namespace common
{

class SOFA_SIMULATION_CORE_API VectorOperations : public sofa::core::behavior::BaseVectorOperations
{
public:

    VectorOperations(const sofa::core::ExecParams* params, sofa::core::objectmodel::BaseContext* ctx, bool precomputedTraversalOrder=false);

    /// Allocate a temporary vector
    void v_alloc(sofa::core::MultiVecCoordId& v);
    void v_alloc(sofa::core::MultiVecDerivId& v);
    /// Free a previously allocated temporary vector
    void v_free(sofa::core::MultiVecCoordId& id, bool interactionForceField=false, bool propagate=false);
    void v_free(sofa::core::MultiVecDerivId& id, bool interactionForceField=false, bool propagate=false);

    void v_realloc(sofa::core::MultiVecCoordId& id, bool interactionForceField=false, bool propagate=false);
    void v_realloc(sofa::core::MultiVecDerivId& id, bool interactionForceField=false, bool propagate=false);

    void v_clear(core::MultiVecId v); ///< v=0
    void v_eq(core::MultiVecId v, core::ConstMultiVecId a); ///< v=a
    void v_eq(core::MultiVecId v, core::ConstMultiVecId a, SReal f); ///< v=f*a
    void v_peq(core::MultiVecId v, core::ConstMultiVecId a, SReal f=1.0); ///< v+=f*a
#ifdef SOFA_SMP
    void v_peq(core::MultiVecId v, core::ConstMultiVecId a, Shared<SReal> &fSh, SReal f=1.0) ; ///< v+=f*a
    void v_meq(core::MultiVecId v, core::ConstMultiVecId a, Shared<SReal> &fSh) ; ///< v+=f*a
#endif
    void v_teq(core::MultiVecId v, SReal f) ; ///< v*=f
    void v_op(core::MultiVecId v, core::ConstMultiVecId a, core::ConstMultiVecId  b, SReal f=1.0) ; ///< v=a+b*f
#ifdef SOFA_SMP
    void v_op(core::MultiVecId v, core::ConstMultiVecId a, core::ConstMultiVecId b, Shared<SReal> &f) ; ///< v=a+b*f
#endif
    void v_multiop(const core::behavior::BaseMechanicalState::VMultiOp& o);
    void v_dot(core::ConstMultiVecId a, core::ConstMultiVecId  b); ///< a dot b ( get result using finish )
    void v_norm(core::ConstMultiVecId a, unsigned l); ///< Compute the norm of a vector ( get result using finish ). The type of norm is set by parameter l. Use 0 for the infinite norm. Note that the 2-norm is more efficiently computed using the square root of the dot product.
#ifdef SOFA_SMP
    void v_dot(Shared<SReal> &result,core::ConstMultiVecId a, core::ConstMultiVecId b) ; ///< a dot b
#endif
    void v_threshold(core::MultiVecId a, SReal threshold); ///< nullify the values below the given threshold

    SReal finish();
    void print(sofa::core::ConstMultiVecId v, std::ostream& out, std::string prefix="", std::string suffix="" );

    virtual size_t v_size(core::MultiVecId v);

protected:
    VisitorExecuteFunc executeVisitor;
    /// Result of latest v_dot operation
    SReal result;

};

}
}
}


#endif //SOFA_SIMULATION_CORE_VECTOROPERATIONS_H
