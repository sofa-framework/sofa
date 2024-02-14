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

#include <sofa/simulation/Visitor.h>
#include <sofa/core/fwd.h>
#include <sofa/core/VecId.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::simulation
{

/** Base class for easily creating new actions for mechanical simulation.

During the first traversal (top-down), method processNodeTopDown(simulation::Node*) is applied to each simulation::Node.
Each component attached to this node is processed using the appropriate method, prefixed by fwd.
During the second traversal (bottom-up), method processNodeBottomUp(simulation::Node*) is applied to each simulation::Node.
Each component attached to this node is processed using the appropriate method, prefixed by bwd.
The default behavior of the fwd* and bwd* is to do nothing.
Derived actions typically overload these methods to implement the desired processing.

*/
class SOFA_SIMULATION_CORE_API BaseMechanicalVisitor : public Visitor
{

protected:
    simulation::Node* root; ///< root node from which the visitor was executed

    SOFA_ATTRIBUTE_DEPRECATED_NODEDATA()
    SReal* rootData { nullptr }; ///< data for root node

    virtual Result processNodeTopDown(simulation::Node* node, VisitorContext* ctx);
    virtual void processNodeBottomUp(simulation::Node* node, VisitorContext* ctx);

public:
    BaseMechanicalVisitor(const sofa::core::ExecParams* params);

    SOFA_ATTRIBUTE_DEPRECATED_NODEDATA()
    virtual bool readNodeData() const { return false; };

    SOFA_ATTRIBUTE_DEPRECATED_NODEDATA()
    virtual bool writeNodeData() const { return false; };

    SOFA_ATTRIBUTE_DEPRECATED_NODEDATA()
    virtual void setNodeData(simulation::Node* /*node*/, SReal* /*nodeData*/, const SReal* /*parentData*/) {};

    SOFA_ATTRIBUTE_DEPRECATED_NODEDATA()
    virtual void addNodeData(simulation::Node* /*node*/, SReal* /*parentData*/, const SReal* /*nodeData*/) {};

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override;

    /**@name Forward processing
    Methods called during the forward (top-down) traversal of the data structure.
    Method processNodeTopDown(simulation::Node*) calls the fwd* methods in the order given here. When there is a mapping, it is processed first, then method fwdMappedMechanicalState is applied to the BaseMechanicalState.
    When there is no mapping, the BaseMechanicalState is processed first using method fwdMechanicalState.
    Then, the other fwd* methods are applied in the given order.
    */
    ///@{

    /// This method calls the fwd* methods during the forward traversal. You typically do not overload it.
    Result processNodeTopDown(simulation::Node* node) override;

    /// Parallel version of processNodeTopDown.
    /// This method calls the fwd* methods during the forward traversal. You typically do not overload it.
    SOFA_ATTRIBUTE_DEPRECATED_LOCALSTORAGE()
    Result processNodeTopDown(simulation::Node * node, LocalStorage * stack) override;

    /// Process the OdeSolver
    virtual Result fwdOdeSolver(simulation::Node* /*node*/, sofa::core::behavior::OdeSolver* /*solver*/);

    /// Process the OdeSolver
    virtual Result fwdOdeSolver(VisitorContext* ctx, sofa::core::behavior::OdeSolver* solver);

    /// Process the ConstraintSolver
    virtual Result fwdConstraintSolver(simulation::Node* /*node*/, sofa::core::behavior::ConstraintSolver* /*solver*/);

    /// Process the ConstraintSolver
    virtual Result fwdConstraintSolver(VisitorContext* ctx, sofa::core::behavior::ConstraintSolver* solver);

    /// Process the BaseMechanicalMapping
    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/);

    /// Process the BaseMechanicalMapping
    virtual Result fwdMechanicalMapping(VisitorContext* ctx, sofa::core::BaseMapping* map);

    /// Process the BaseMechanicalState if it is mapped from the parent level
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, sofa::core::behavior::BaseMechanicalState* /*mm*/);

    /// Process the BaseMechanicalState if it is mapped from the parent level
    virtual Result fwdMappedMechanicalState(VisitorContext* ctx, sofa::core::behavior::BaseMechanicalState* mm);

    /// Process the BaseMechanicalState if it is not mapped from the parent level
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, sofa::core::behavior::BaseMechanicalState* /*mm*/);

    /// Process the BaseMechanicalState if it is not mapped from the parent level
    virtual Result fwdMechanicalState(VisitorContext* ctx, sofa::core::behavior::BaseMechanicalState* mm);

    /// Process the BaseMass
    virtual Result fwdMass(simulation::Node* /*node*/, sofa::core::behavior::BaseMass* /*mass*/);

    /// Process the BaseMass
    virtual Result fwdMass(VisitorContext* ctx, sofa::core::behavior::BaseMass* mass);

    /// Process all the BaseForceField
    virtual Result fwdForceField(simulation::Node* /*node*/, sofa::core::behavior::BaseForceField* /*ff*/);

    /// Process all the BaseForceField
    virtual Result fwdForceField(VisitorContext* ctx, sofa::core::behavior::BaseForceField* ff);

    /// Process all the InteractionForceField
    virtual Result fwdInteractionForceField(simulation::Node* node, sofa::core::behavior::BaseInteractionForceField* ff);

    /// Process all the InteractionForceField
    virtual Result fwdInteractionForceField(VisitorContext* ctx, sofa::core::behavior::BaseInteractionForceField* ff);

    /// Process all the BaseProjectiveConstraintSet
    virtual Result fwdProjectiveConstraintSet(simulation::Node* /*node*/, sofa::core::behavior::BaseProjectiveConstraintSet* /*c*/);

    /// Process all the BaseConstraintSet
    virtual Result fwdConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseConstraintSet* /*c*/);

    /// Process all the BaseProjectiveConstraintSet
    virtual Result fwdProjectiveConstraintSet(VisitorContext* ctx,sofa::core::behavior::BaseProjectiveConstraintSet* c);

    /// Process all the BaseConstraintSet
    virtual Result fwdConstraintSet(VisitorContext* ctx,sofa::core::behavior::BaseConstraintSet* c);

    /// Process all the InteractionConstraint
    virtual Result fwdInteractionProjectiveConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseInteractionProjectiveConstraintSet* /*c*/);

    /// Process all the InteractionConstraint
    virtual Result fwdInteractionConstraint(simulation::Node* /*node*/,sofa::core::behavior::BaseInteractionConstraint* /*c*/);

    /// Process all the InteractionConstraint
    virtual Result fwdInteractionProjectiveConstraintSet(VisitorContext* ctx,sofa::core::behavior::BaseInteractionProjectiveConstraintSet* c);

    /// Process all the InteractionConstraint
    virtual Result fwdInteractionConstraint(VisitorContext* ctx,sofa::core::behavior::BaseInteractionConstraint* c);

    ///@}

    /**@name Backward processing
    Methods called during the backward (bottom-up) traversal of the data structure.
    Method processNodeBottomUp(simulation::Node*) calls the bwd* methods.
    When there is a mapping, method bwdMappedMechanicalState is applied to the BaseMechanicalState.
    When there is no mapping, the BaseMechanicalState is processed using method bwdMechanicalState.
    Finally, the mapping (if any) is processed using method bwdMechanicalMapping.
    */
    ///@{

    /// This method calls the bwd* methods during the backward traversal. You typically do not overload it.
    void processNodeBottomUp(simulation::Node* node) override;

    /// Parallel version of processNodeBottomUp.
    /// This method calls the bwd* methods during the backward traversal. You typically do not overload it.
    SOFA_ATTRIBUTE_DEPRECATED_LOCALSTORAGE()
    void processNodeBottomUp(simulation::Node* /*node*/, LocalStorage * stack) override;

    /// Process the BaseMechanicalState when it is not mapped from parent level
    virtual void bwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* /*mm*/);

    /// Process the BaseMechanicalState when it is not mapped from parent level
    virtual void bwdMechanicalState(VisitorContext* ctx,sofa::core::behavior::BaseMechanicalState* mm);

    /// Process the BaseMechanicalState when it is mapped from parent level
    virtual void bwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* /*mm*/);

    /// Process the BaseMechanicalState when it is mapped from parent level
    virtual void bwdMappedMechanicalState(VisitorContext* ctx,sofa::core::behavior::BaseMechanicalState* mm);

    /// Process the BaseMechanicalMapping
    virtual void bwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/);

    /// Process the BaseMechanicalMapping
    virtual void bwdMechanicalMapping(VisitorContext* ctx, sofa::core::BaseMapping* map);

    /// Process the OdeSolver
    virtual void bwdOdeSolver(simulation::Node* /*node*/,sofa::core::behavior::OdeSolver* /*solver*/);

    /// Process the OdeSolver
    virtual void bwdOdeSolver(VisitorContext* ctx,sofa::core::behavior::OdeSolver* solver);

    /// Process the ConstraintSolver
    virtual void bwdConstraintSolver(simulation::Node* /*node*/,sofa::core::behavior::ConstraintSolver* /*solver*/);

    /// Process the ConstraintSolver
    virtual void bwdConstraintSolver(VisitorContext* ctx,sofa::core::behavior::ConstraintSolver* solver);

    /// Process all the BaseProjectiveConstraintSet
    virtual void bwdProjectiveConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseProjectiveConstraintSet* /*c*/);

    /// Process all the BaseConstraintSet
    virtual void bwdConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseConstraintSet* /*c*/);

    /// Process all the BaseProjectiveConstraintSet
    virtual void bwdProjectiveConstraintSet(VisitorContext* ctx,sofa::core::behavior::BaseProjectiveConstraintSet* c);

    /// Process all the BaseConstraintSet
    virtual void bwdConstraintSet(VisitorContext* ctx,sofa::core::behavior::BaseConstraintSet* c);

    ///@}

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override;

    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map);

#ifdef SOFA_DUMP_VISITOR_INFO
    ctime_t begin(simulation::Node* node, sofa::core::objectmodel::BaseObject* obj, const std::string &info=std::string("type")) override;
    void end(simulation::Node* node, sofa::core::objectmodel::BaseObject* obj, ctime_t t0) override;

    virtual void setReadWriteVectors() {}
    virtual void addReadVector(core::ConstMultiVecId id) {  readVector.push_back(id);  }
    virtual void addWriteVector(core::MultiVecId id) {  writeVector.push_back(id);  }
    virtual void addReadWriteVector(core::MultiVecId id) {  readVector.push_back(core::ConstMultiVecId(id)); writeVector.push_back(id);  }
    void printReadVectors(core::behavior::BaseMechanicalState* mm);
    void printReadVectors(simulation::Node* node, sofa::core::objectmodel::BaseObject* obj);
    void printWriteVectors(core::behavior::BaseMechanicalState* mm);
    void printWriteVectors(simulation::Node* node, sofa::core::objectmodel::BaseObject* obj);
protected:
    sofa::type::vector< sofa::core::ConstMultiVecId > readVector;
    sofa::type::vector< sofa::core::MultiVecId > writeVector;
#endif

private:
    static const std::string fwdVisitorType;
    static const std::string bwdVisitorType;
};

} // namespace sofa::simulation
