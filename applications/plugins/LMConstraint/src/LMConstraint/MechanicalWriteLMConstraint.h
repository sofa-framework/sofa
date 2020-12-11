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

#include <sofa/simulation/MechanicalVisitor.h>
#include <LMConstraint/BaseLMConstraint.h>

namespace sofa
{

namespace simulation
{


class LMCONSTRAINT_API MechanicalWriteLMConstraint : public sofa::simulation::BaseMechanicalVisitor
{
public:
    MechanicalWriteLMConstraint(const sofa::core::ExecParams * params)
        : BaseMechanicalVisitor(params)
        , offset(0)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    Result fwdConstraintSet(simulation::Node* /*node*/, core::behavior::BaseConstraintSet* c) override;
    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/) override
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalWriteLMConstraint"; }
    virtual std::string getInfos() const override ;

    virtual void clear() {datasC.clear(); offset=0;}
    virtual const std::vector< core::behavior::BaseLMConstraint *> &getConstraints() const {return datasC;}
    virtual unsigned int numConstraint() {return static_cast<unsigned int>(datasC.size());}

    virtual void setMultiVecId(core::MultiVecId i) {id=i;}
    core::MultiVecId getMultiVecId() const { return id; }


    virtual void setOrder(core::ConstraintParams::ConstOrder i) {order=i;}
    core::ConstraintParams::ConstOrder getOrder() const { return order; }

    bool isThreadSafe() const override
    {
        return false;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
    }
#endif

protected:
    unsigned int offset;
    sofa::core::ConstraintParams::ConstOrder order;
    core::MultiVecId id;
    helper::vector< core::behavior::BaseLMConstraint *> datasC;

};


} // namespace simulation

} // namespace sofa

