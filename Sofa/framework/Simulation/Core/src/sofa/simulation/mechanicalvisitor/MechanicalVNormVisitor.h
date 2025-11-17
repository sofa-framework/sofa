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

#include <sofa/simulation/BaseMechanicalVisitor.h>

namespace sofa::simulation::mechanicalvisitor
{

/** Compute the norm of a vector.
 * The type of norm is set by parameter @a l. Use 0 for the infinite norm.
 * Note that the 2-norm is more efficiently computed using the square root of the dot product.
 * @author Francois Faure, 2013
 */
class SOFA_SIMULATION_CORE_API MechanicalVNormVisitor : public BaseMechanicalVisitor
{
    SReal accum; ///< accumulate value before computing its root
public:
    sofa::core::ConstMultiVecId a;
    unsigned l; ///< Type of norm:  for l>0, \f$ \|v\|_l = ( \sum_{i<dim(v)} \|v[i]\|^{l} )^{1/l} \f$, while we use l=0 for the infinite norm: \f$ \|v\|_\infinite = \max_{i<dim(v)} \|v[i]\| \f$
    MechanicalVNormVisitor(const sofa::core::ExecParams* eparams, sofa::core::ConstMultiVecId avecid, unsigned lnorm)
            : BaseMechanicalVisitor(eparams), accum(0), a(avecid), l(lnorm)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    SReal getResult() const;

    Result fwdMechanicalState(VisitorContext* ctx,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalVNormVisitor";}
    virtual std::string getInfos() const override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadVector(a);
    }
#endif
};
}
