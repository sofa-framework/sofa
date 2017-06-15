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
#ifndef SOFA_SIMULATION_MECHANICALVISITOR_H
#define SOFA_SIMULATION_MECHANICALVISITOR_H
#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/simulation/Visitor.h>
#include <sofa/core/VecId.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/behavior/BaseInteractionConstraint.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/core/behavior/BaseInteractionProjectiveConstraintSet.h>
#include <sofa/core/behavior/BaseConstraintSet.h>

//TO REMOVE ONCE THE CONVERGENCE IS DONE
#include <sofa/core/behavior/BaseLMConstraint.h>

//#include <sofa/defaulttype/BaseMatrix.h>
//#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/map.h>
#include <iostream>
#include <functional>

namespace sofa
{

namespace simulation
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
    SReal* rootData; ///< data for root node

    virtual Result processNodeTopDown(simulation::Node* node, VisitorContext* ctx);
    virtual void processNodeBottomUp(simulation::Node* node, VisitorContext* ctx);

    struct forceMaskActivator : public std::binary_function<core::behavior::BaseMechanicalState*, bool , void >
    {
        void operator()( core::behavior::BaseMechanicalState* m, bool activate ) const
        {
            m->forceMask.activate(activate);
        }
    };

public:

    BaseMechanicalVisitor(const core::ExecParams* params)
        : Visitor(params)
		, root(NULL), rootData(NULL)
    {
		// mechanical visitors shouldn't be able to acess a sleeping node, only visual visitor should
		canAccessSleepingNode = false;
	}

    /// Return true if this visitor need to read the node-specific data if given
    virtual bool readNodeData() const
    { return false; }

    /// Return true if this visitor need to write to the node-specific data if given
    virtual bool writeNodeData() const
    { return false; }

    virtual void setNodeData(simulation::Node* /*node*/, SReal* nodeData, const SReal* parentData)
    {
        *nodeData = (parentData == NULL) ? 0.0 : *parentData;
    }

    virtual void addNodeData(simulation::Node* /*node*/, SReal* parentData, const SReal* nodeData)
    {
        if (parentData)
            *parentData += *nodeData;
    }

    static inline void ForceMaskActivate( const helper::vector<core::behavior::BaseMechanicalState*>& v )
    {
        std::for_each( v.begin(), v.end(), std::bind2nd( forceMaskActivator(), true ) );
    }

    static inline void ForceMaskDeactivate( const helper::vector<core::behavior::BaseMechanicalState*>& v)
    {
        std::for_each( v.begin(), v.end(), std::bind2nd( forceMaskActivator(), false ) );
    }


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalVisitor"; }

    /**@name Forward processing
    Methods called during the forward (top-down) traversal of the data structure.
    Method processNodeTopDown(simulation::Node*) calls the fwd* methods in the order given here. When there is a mapping, it is processed first, then method fwdMappedMechanicalState is applied to the BaseMechanicalState.
    When there is no mapping, the BaseMechanicalState is processed first using method fwdMechanicalState.
    Then, the other fwd* methods are applied in the given order.
    */
    ///@{

    /// This method calls the fwd* methods during the forward traversal. You typically do not overload it.
    virtual Result processNodeTopDown(simulation::Node* node);

    /// Parallel version of processNodeTopDown.
    /// This method calls the fwd* methods during the forward traversal. You typically do not overload it.
    virtual Result processNodeTopDown(simulation::Node* node, LocalStorage* stack);

    /// Process the OdeSolver
    virtual Result fwdOdeSolver(simulation::Node* /*node*/, core::behavior::OdeSolver* /*solver*/)
    {
        return RESULT_CONTINUE;
    }

    /// Process the OdeSolver
    virtual Result fwdOdeSolver(VisitorContext* ctx, core::behavior::OdeSolver* solver)
    {
        return fwdOdeSolver(ctx->node, solver);
    }

    /// Process the ConstraintSolver
    virtual Result fwdConstraintSolver(simulation::Node* /*node*/, core::behavior::ConstraintSolver* /*solver*/)
    {
        return RESULT_CONTINUE;
    }

    /// Process the ConstraintSolver
    virtual Result fwdConstraintSolver(VisitorContext* ctx, core::behavior::ConstraintSolver* solver)
    {
        return fwdConstraintSolver(ctx->node, solver);
    }

    /// Process the BaseMechanicalMapping
    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return RESULT_CONTINUE;
    }

    /// Process the BaseMechanicalMapping
    virtual Result fwdMechanicalMapping(VisitorContext* ctx, core::BaseMapping* map)
    {
        return fwdMechanicalMapping(ctx->node, map);
    }

    /// Process the BaseMechanicalState if it is mapped from the parent level
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
    {
        return RESULT_CONTINUE;
    }

    /// Process the BaseMechanicalState if it is mapped from the parent level
    virtual Result fwdMappedMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm)
    {
        return fwdMappedMechanicalState(ctx->node, mm);
    }

    /// Process the BaseMechanicalState if it is not mapped from the parent level
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
    {
        return RESULT_CONTINUE;
    }

    /// Process the BaseMechanicalState if it is not mapped from the parent level
    virtual Result fwdMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm)
    {
        return fwdMechanicalState(ctx->node, mm);
    }

    /// Process the BaseMass
    virtual Result fwdMass(simulation::Node* /*node*/, core::behavior::BaseMass* /*mass*/)
    {
        return RESULT_CONTINUE;
    }

    /// Process the BaseMass
    virtual Result fwdMass(VisitorContext* ctx, core::behavior::BaseMass* mass)
    {
        return fwdMass(ctx->node, mass);
    }

    /// Process all the BaseForceField
    virtual Result fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* /*ff*/)
    {
        return RESULT_CONTINUE;
    }


    /// Process all the BaseForceField
    virtual Result fwdForceField(VisitorContext* ctx, core::behavior::BaseForceField* ff)
    {
        return fwdForceField(ctx->node, ff);
    }

    /// Process all the InteractionForceField
    virtual Result fwdInteractionForceField(simulation::Node* node, core::behavior::BaseInteractionForceField* ff)
    {
        return fwdForceField(node, ff);
    }

    /// Process all the InteractionForceField
    virtual Result fwdInteractionForceField(VisitorContext* ctx, core::behavior::BaseInteractionForceField* ff)
    {
        return fwdInteractionForceField(ctx->node, ff);
    }

    /// Process all the BaseProjectiveConstraintSet
    virtual Result fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* /*c*/)
    {
        return RESULT_CONTINUE;
    }

    /// Process all the BaseConstraintSet
    virtual Result fwdConstraintSet(simulation::Node* /*node*/, core::behavior::BaseConstraintSet* /*c*/)
    {
        return RESULT_CONTINUE;
    }

    /// Process all the BaseProjectiveConstraintSet
    virtual Result fwdProjectiveConstraintSet(VisitorContext* ctx, core::behavior::BaseProjectiveConstraintSet* c)
    {
        return fwdProjectiveConstraintSet(ctx->node, c);
    }

    /// Process all the BaseConstraintSet
    virtual Result fwdConstraintSet(VisitorContext* ctx, core::behavior::BaseConstraintSet* c)
    {
        return fwdConstraintSet(ctx->node, c);
    }

    /// Process all the InteractionConstraint
    virtual Result fwdInteractionProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseInteractionProjectiveConstraintSet* /*c*/)
    {
        return RESULT_CONTINUE;
    }

    /// Process all the InteractionConstraint
    virtual Result fwdInteractionConstraint(simulation::Node* /*node*/, core::behavior::BaseInteractionConstraint* /*c*/)
    {
        return RESULT_CONTINUE;
    }

    /// Process all the InteractionConstraint
    virtual Result fwdInteractionProjectiveConstraintSet(VisitorContext* ctx, core::behavior::BaseInteractionProjectiveConstraintSet* c)
    {
        return fwdProjectiveConstraintSet(ctx->node, c);
    }

    /// Process all the InteractionConstraint
    virtual Result fwdInteractionConstraint(VisitorContext* ctx, core::behavior::BaseInteractionConstraint* c)
    {
        return fwdConstraintSet(ctx->node, c);
    }

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
    virtual void processNodeBottomUp(simulation::Node* node);

    /// Parallel version of processNodeBottomUp.
    /// This method calls the bwd* methods during the backward traversal. You typically do not overload it.
    virtual void processNodeBottomUp(simulation::Node* /*node*/, LocalStorage* stack);

    /// Process the BaseMechanicalState when it is not mapped from parent level
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
    {}

    /// Process the BaseMechanicalState when it is not mapped from parent level
    virtual void bwdMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm)
    { bwdMechanicalState(ctx->node, mm); }

    /// Process the BaseMechanicalState when it is mapped from parent level
    virtual void bwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
    {}

    /// Process the BaseMechanicalState when it is mapped from parent level
    virtual void bwdMappedMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm)
    { bwdMappedMechanicalState(ctx->node, mm); }

    /// Process the BaseMechanicalMapping
    virtual void bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {}

    /// Process the BaseMechanicalMapping
    virtual void bwdMechanicalMapping(VisitorContext* ctx, core::BaseMapping* map)
    { bwdMechanicalMapping(ctx->node, map); }

    /// Process the OdeSolver
    virtual void bwdOdeSolver(simulation::Node* /*node*/, core::behavior::OdeSolver* /*solver*/)
    {}

    /// Process the OdeSolver
    virtual void bwdOdeSolver(VisitorContext* ctx, core::behavior::OdeSolver* solver)
    { bwdOdeSolver(ctx->node, solver); }

    /// Process the ConstraintSolver
    virtual void bwdConstraintSolver(simulation::Node* /*node*/, core::behavior::ConstraintSolver* /*solver*/)
    {}

    /// Process the ConstraintSolver
    virtual void bwdConstraintSolver(VisitorContext* ctx, core::behavior::ConstraintSolver* solver)
    { bwdConstraintSolver(ctx->node, solver); }

    /// Process all the BaseProjectiveConstraintSet
    virtual void bwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* /*c*/)
    {}

    /// Process all the BaseConstraintSet
    virtual void bwdConstraintSet(simulation::Node* /*node*/, core::behavior::BaseConstraintSet* /*c*/)
    {}

    /// Process all the BaseProjectiveConstraintSet
    virtual void bwdProjectiveConstraintSet(VisitorContext* ctx, core::behavior::BaseProjectiveConstraintSet* c)
    { bwdProjectiveConstraintSet(ctx->node, c); }

    /// Process all the BaseConstraintSet
    virtual void bwdConstraintSet(VisitorContext* ctx, core::behavior::BaseConstraintSet* c)
    { bwdConstraintSet(ctx->node, c); }

    ///@}


    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const
    {
        return "animate";
    }

    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
    {
        return !map->areForcesMapped();
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    ctime_t begin(simulation::Node* node, core::objectmodel::BaseObject* obj, const std::string &info=std::string("type"));
    void end(simulation::Node* node, core::objectmodel::BaseObject* obj, ctime_t t0);
#endif

#ifdef SOFA_DUMP_VISITOR_INFO
    virtual void setReadWriteVectors() {}
    virtual void addReadVector(core::ConstMultiVecId id) {  readVector.push_back(id);  }
    virtual void addWriteVector(core::MultiVecId id) {  writeVector.push_back(id);  }
    virtual void addReadWriteVector(core::MultiVecId id) {  readVector.push_back(core::ConstMultiVecId(id)); writeVector.push_back(id);  }
    void printReadVectors(core::behavior::BaseMechanicalState* mm);
    void printReadVectors(simulation::Node* node, core::objectmodel::BaseObject* obj);
    void printWriteVectors(core::behavior::BaseMechanicalState* mm);
    void printWriteVectors(simulation::Node* node, core::objectmodel::BaseObject* obj);
protected:
    sofa::helper::vector< core::ConstMultiVecId > readVector;
    sofa::helper::vector< core::MultiVecId > writeVector;
#endif
};

class SOFA_SIMULATION_CORE_API MechanicalVisitor : public BaseMechanicalVisitor
{

protected:
    const core::MechanicalParams* mparams;

public:

    MechanicalVisitor(const core::MechanicalParams* m_mparams)
        : BaseMechanicalVisitor(m_mparams)
        , mparams(m_mparams)
    {}
};

/** Compute the total number of DOFs */
class SOFA_SIMULATION_CORE_API MechanicalGetDimensionVisitor : public MechanicalVisitor
{
public:
    MechanicalGetDimensionVisitor(const core::MechanicalParams* mparams, SReal* result)
        : MechanicalVisitor(mparams)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
        rootData = result;
    }

    virtual Result fwdMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalGetDimensionVisitor";}

    virtual bool writeNodeData() const
    {
        return true;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
};

/** Find the first available index for a VecId
*/
template <sofa::core::VecType vtype>
class SOFA_SIMULATION_CORE_API MechanicalVAvailVisitor : public BaseMechanicalVisitor
{
public:
    typedef sofa::core::TVecId<vtype,sofa::core::V_WRITE> MyVecId;
    typedef sofa::core::TMultiVecId<vtype,sofa::core::V_WRITE> MyMultiVecId;
    typedef std::set<sofa::core::BaseState*> StateSet;
    MyVecId& v;
    StateSet states;
    MechanicalVAvailVisitor( const core::ExecParams* params, MyVecId& v)
        : BaseMechanicalVisitor(params), v(v)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalVAvailVisitor"; }
    virtual std::string getInfos() const;
    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return false;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        MyMultiVecId mv(v);
        addReadWriteVector( mv );
    }
#endif
};



/**
 * Initialize unset MState destVecId vectors with srcVecId vectors value.
 *
 */
template< sofa::core::VecType vtype >
class SOFA_SIMULATION_CORE_API MechanicalVInitVisitor : public BaseMechanicalVisitor
{
public:
    typedef sofa::core::TMultiVecId<vtype,sofa::core::V_WRITE> DestMultiVecId;
    typedef sofa::core::TMultiVecId<vtype,sofa::core::V_READ> SrcMultiVecId;

    DestMultiVecId vDest;
    SrcMultiVecId vSrc;
    bool m_propagate;

    /// Default constructor
    /// \param _vDest output vector
    /// \param _vSrc input vector
    /// \param propagate sets to true propagates vector initialization to mapped mechanical states
    MechanicalVInitVisitor(const core::ExecParams* params, DestMultiVecId _vDest, SrcMultiVecId _vSrc = SrcMultiVecId::null(), bool propagate=false)
        : BaseMechanicalVisitor(params)
        , vDest(_vDest)
        , vSrc(_vSrc)
        , m_propagate(propagate)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false;
    }

    virtual Result fwdMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm);

    virtual Result fwdMappedMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const
    {
        return "MechanicalVInitVisitor";
    }

    virtual std::string getInfos() const;

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return false;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadVector(vSrc);
        addWriteVector(vDest);
    }
#endif
};




/** Reserve an auxiliary vector identified by a symbolic constant.
*/
template <sofa::core::VecType vtype>
class SOFA_SIMULATION_CORE_API MechanicalVAllocVisitor : public BaseMechanicalVisitor
{
public:
    typedef sofa::core::TMultiVecId<vtype, sofa::core::V_WRITE> MyMultiVecId;
    MyMultiVecId v;
    MechanicalVAllocVisitor( const core::ExecParams* params, MyMultiVecId v )
        : BaseMechanicalVisitor(params) , v(v)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalVAllocVisitor"; }
    virtual std::string getInfos() const;
    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadWriteVector(v);
    }
#endif
};


/**
 * Reserve an auxiliary vector identified by a symbolic constant.
 *
 */
template< sofa::core::VecType vtype >
class SOFA_SIMULATION_CORE_API MechanicalVReallocVisitor : public BaseMechanicalVisitor
{
public:
    typedef sofa::core::TMultiVecId<vtype,sofa::core::V_WRITE> DestMultiVecId;
    typedef sofa::core::TVecId<vtype,sofa::core::V_WRITE> MyVecId;


    DestMultiVecId *v;
    bool m_propagate;
    bool m_interactionForceField;

    /// Default constructor
    /// \param _vDest output vector
    /// \param propagate sets to true propagates vector initialization to mapped mechanical states
    /// \param interactionForceField sets to true also initializes external mechanical states linked by an interaction force field
    MechanicalVReallocVisitor(const core::ExecParams* params, DestMultiVecId *v, bool interactionForceField=false, bool propagate=false)
        : BaseMechanicalVisitor(params)
        , v(v)
        , m_propagate(propagate)
        , m_interactionForceField(interactionForceField)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false;
    }

    virtual Result fwdMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm);

    virtual Result fwdMappedMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm);

    virtual Result fwdInteractionForceField(simulation::Node* node, core::behavior::BaseInteractionForceField* ff);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const
    {
        return "MechanicalVReallocVisitor";
    }

    virtual std::string getInfos() const;

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addWriteVector(*v);
    }
#endif
protected:


    MyVecId getId( core::behavior::BaseMechanicalState* mm );
};



/** Free an auxiliary vector identified by a symbolic constant */
template< sofa::core::VecType vtype >
class SOFA_SIMULATION_CORE_API MechanicalVFreeVisitor : public BaseMechanicalVisitor
{
public:
    typedef sofa::core::TMultiVecId<vtype,sofa::core::V_WRITE> MyMultiVecId;
    MyMultiVecId v;
    bool interactionForceField;
    bool propagate;

    MechanicalVFreeVisitor( const sofa::core::ExecParams* params, MyMultiVecId v, bool interactionForceField=false, bool propagate=false)
        : BaseMechanicalVisitor(params) , v(v), interactionForceField(interactionForceField), propagate(propagate)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdInteractionForceField(simulation::Node* node, core::behavior::BaseInteractionForceField* ff);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalVFreeVisitor"; }
    virtual std::string getInfos() const;
    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
};

/** Perform a vector operation v=a+b*f
*/
class SOFA_SIMULATION_CORE_API MechanicalVOpVisitor : public BaseMechanicalVisitor
{
public:
    sofa::core::MultiVecId v;
    sofa::core::ConstMultiVecId a;
    sofa::core::ConstMultiVecId b;
    SReal f;
    bool mapped;
    bool only_mapped;
    MechanicalVOpVisitor(const sofa::core::ExecParams* params,
                         sofa::core::MultiVecId v,sofa::core::ConstMultiVecId a = sofa::core::ConstMultiVecId::null(), sofa::core::ConstMultiVecId b = sofa::core::ConstMultiVecId::null(),
                         SReal f=1.0 )
        : BaseMechanicalVisitor(params) , v(v), a(a), b(b), f(f), mapped(false), only_mapped(false)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    // If mapped or only_mapped is ste, this visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
    {
        if (mapped || only_mapped)
            return false;
        else
            return !map->areForcesMapped();
    }

    MechanicalVOpVisitor& setMapped(bool m = true) { mapped = m; return *this; }
    MechanicalVOpVisitor& setOnlyMapped(bool m = true) { only_mapped = m; return *this; }

    virtual Result fwdMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMappedMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm);

    virtual const char* getClassName() const { return "MechanicalVOpVisitor";}
    virtual std::string getInfos() const
    {
        std::string info="v=";
        std::string aLabel;
        std::string bLabel;
        std::string fLabel;

        std::ostringstream out;
        out << "f["<<f<<"]";
        fLabel+= out.str();

        if (!a.isNull())
        {
            info+="a";
            aLabel="a[" + a.getName() + "] ";
            if (!b.isNull())
            {
                info += "+b*f";
                bLabel += "b[" + b.getName() + "] ";
            }
        }
        else
        {
            if (!b.isNull())
            {
                info += "b*f";
                bLabel += "b[" + b.getName() + "] ";
            }
            else
            {
                info+="zero"; fLabel.clear();
            }
        }
        info += " : with v[" + v.getName() + "] " + aLabel + bLabel + fLabel;
        return info;
    }
    //virtual void processNodeBottomUp(simulation::Node* node);

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
    virtual bool readNodeData() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadVector(a);
        addReadVector(b);
        addWriteVector(v);
    }
#endif
};

/** Perform a sequence of linear vector accumulation operation $r_i = sum_j (v_j*f_{ij})
*
*  This is used to compute in on steps operations such as $v = v + a*dt, x = x + v*dt$.
*  Note that if the result vector appears inside the expression, it must be the first operand.
*/
class SOFA_SIMULATION_CORE_API MechanicalVMultiOpVisitor : public BaseMechanicalVisitor
{
public:
    typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
    bool mapped;
    //     MechanicalVMultiOpVisitor()
    //     {}
    MechanicalVMultiOpVisitor(const sofa::core::ExecParams* params, const VMultiOp& o)
        : BaseMechanicalVisitor(params), mapped(false), ops(o)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    MechanicalVMultiOpVisitor& setMapped(bool m = true) { mapped = m; return *this; }

    virtual Result fwdMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMappedMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm);

    //virtual void processNodeBottomUp(simulation::Node* node);

    virtual const char* getClassName() const { return "MechanicalVMultiOpVisitor"; }
    virtual std::string getInfos() const
    {
        std::ostringstream out;
        for(VMultiOp::const_iterator it = ops.begin(), itend = ops.end(); it != itend; ++it)
        {
            if (it != ops.begin())
                out << " ;   ";
            core::MultiVecId r = it->first;
            out << r.getName();
            const helper::vector< std::pair< core::ConstMultiVecId, SReal > >& operands = it->second;
            int nop = (int)operands.size();
            if (nop==0)
            {
                out << " = 0";
            }
            else if (nop==1)
            {
                if (operands[0].first.getName() == r.getName())
                    out << " *= " << operands[0].second;
                else
                {
                    out << " = " << operands[0].first.getName();
                    if (operands[0].second != 1.0)
                        out << "*"<<operands[0].second;
                }
            }
            else
            {
                int i;
                if (operands[0].first.getName() == r.getName() && operands[0].second == 1.0)
                {
                    out << " +=";
                    i = 1;
                }
                else
                {
                    out << " =";
                    i = 0;
                }
                for (; i<nop; ++i)
                {
                    out << " " << operands[i].first.getName();
                    if (operands[i].second != 1.0)
                        out << "*"<<operands[i].second;
                    if (i < nop-1)
                        out << " +";
                }
            }
        }
        return out.str();
    }
    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
    virtual bool readNodeData() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        for (unsigned int i=0; i<ops.size(); ++i)
        {
            addWriteVector(ops[i].first);
            for (unsigned int j=0; j<ops[i].second.size(); ++j)
            {
                addReadVector(ops[i].second[j].first);
            }
        }
    }
#endif
    void setVMultiOp(VMultiOp &o)
    {
        ops = o;
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
protected:
    VMultiOp ops;
};

/** Compute the dot product of two vectors */
class SOFA_SIMULATION_CORE_API MechanicalVDotVisitor : public BaseMechanicalVisitor
{
public:
    sofa::core::ConstMultiVecId a;
    sofa::core::ConstMultiVecId b;
    MechanicalVDotVisitor(const sofa::core::ExecParams* params, sofa::core::ConstMultiVecId a, sofa::core::ConstMultiVecId b, SReal* t)
        : BaseMechanicalVisitor(params) , a(a), b(b) //, total(t)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
        rootData = t;
    }

    virtual Result fwdMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalVDotVisitor";}
    virtual std::string getInfos() const
    {
        std::string name("v= a*b with a[");
        name += a.getName() + "] and b[" + b.getName() + "]";
        return name;
    }
    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
    virtual bool writeNodeData() const
    {
        return true;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadVector(a);
        addReadVector(b);
    }
#endif
};

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
    MechanicalVNormVisitor(const sofa::core::ExecParams* params, sofa::core::ConstMultiVecId a, unsigned l)
        : BaseMechanicalVisitor(params), accum(0), a(a), l(l)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    SReal getResult() const;

    virtual Result fwdMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalVNormVisitor";}
    virtual std::string getInfos() const
    {
        std::string name("v= norm(a) with a[");
        name += a.getName() + "]";
        return name;
    }
    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
    virtual bool writeNodeData() const
    {
        return true;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadVector(a);
    }
#endif
};

/** Apply a hypothetical displacement.
This action does not modify the state (i.e. positions and velocities) of the objects.
It is typically applied before a MechanicalComputeDfVisitor, in order to compute the df corresponding to a given dx (i.e. apply stiffness).
Dx is propagated to all the layers through the mappings.
*/
class SOFA_SIMULATION_CORE_API MechanicalPropagateDxVisitor : public MechanicalVisitor
{
public:
    sofa::core::MultiVecDerivId dx;

    bool ignoreMask;
    bool ignoreFlag;
    MechanicalPropagateDxVisitor( const sofa::core::MechanicalParams* mparams,
                                  sofa::core::MultiVecDerivId dx, bool m, bool f = false )
        : MechanicalVisitor(mparams) , dx(dx), ignoreMask(m), ignoreFlag(f)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
    {
        if (ignoreFlag)
            return false;
        else
            return !map->areForcesMapped();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalPropagateDxVisitor"; }
    virtual std::string getInfos() const
    {
        std::string name="["+dx.getName()+"]"; return name;
    }
    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadWriteVector(dx);
    }
#endif
};

/** Same as MechanicalPropagateDxVisitor followed by MechanicalResetForceVisitor
*/
class SOFA_SIMULATION_CORE_API MechanicalPropagateDxAndResetForceVisitor : public MechanicalVisitor
{
public:
    sofa::core::MultiVecDerivId dx,f;
    bool ignoreMask;

    MechanicalPropagateDxAndResetForceVisitor(const sofa::core::MechanicalParams* mparams,
                                              sofa::core::MultiVecDerivId dx, sofa::core::MultiVecDerivId f, bool m)
        : MechanicalVisitor(mparams) , dx(dx), f(f), ignoreMask(m)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalPropagateDxAndResetForceVisitor";}
    virtual std::string getInfos() const { std::string name= "dx["+dx.getName()+"] f["+f.getName()+"]"; return name;}

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadWriteVector(dx);
        addWriteVector(f);
    }
#endif
};

/** Same as MechanicalPropagatePositionVisitor followed by MechanicalResetForceVisitor
*/
class SOFA_SIMULATION_CORE_API MechanicalPropagatePositionAndResetForceVisitor : public MechanicalVisitor
{
public:
    sofa::core::MultiVecCoordId x;
    sofa::core::MultiVecDerivId f;
    bool ignoreMask;
    bool applyProjections;

    MechanicalPropagatePositionAndResetForceVisitor(const sofa::core::MechanicalParams* mparams,
                                                    sofa::core::MultiVecCoordId x, sofa::core::MultiVecDerivId f, bool m)
        : MechanicalVisitor(mparams) , x(x), f(f), ignoreMask(m), applyProjections(true)
    {
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual Result fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c);
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalPropagatePositionAndResetForceVisitor"; }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadWriteVector(x);
        addWriteVector(f);
    }
#endif
};


/** Accumulate the product of the mass matrix by a given vector.
Typically used in implicit integration solved by a Conjugate Gradient algorithm.
Note that if a dx vector is given, it is used and propagated by the mappings, Otherwise the current value is used.
*/
class SOFA_SIMULATION_CORE_API MechanicalAddMDxVisitor : public MechanicalVisitor
{
public:
    sofa::core::MultiVecDerivId res;
    sofa::core::MultiVecDerivId dx;
    SReal factor;
    MechanicalAddMDxVisitor(const sofa::core::MechanicalParams* mparams,
                            sofa::core::MultiVecDerivId res, sofa::core::MultiVecDerivId dx, SReal factor)
        : MechanicalVisitor(mparams), res(res), dx(dx), factor(factor)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMass(simulation::Node* /*node*/, core::behavior::BaseMass* mass);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalAddMDxVisitor"; }
    virtual std::string getInfos() const { std::string name="dx["+dx.getName()+"] in res[" + res.getName()+"]"; return name; }

#ifdef SOFA_SUPPORT_MAPPED_MASS
    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual void bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
#else
    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/);
#endif

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadVector(res);

#ifdef SOFA_SUPPORT_MAPPED_MASS
        if (!dx.isNull()) addReadWriteVector(dx);
        else addReadVector(dx);
#else
        addReadVector(dx);
#endif
    }
#endif
};

/** Compute accelerations generated by given forces
*/
class SOFA_SIMULATION_CORE_API MechanicalAccFromFVisitor : public MechanicalVisitor
{
public:
    sofa::core::MultiVecDerivId a;
    //ConstMultiVecDerivId f; // in MechanicalParams
    MechanicalAccFromFVisitor(const sofa::core::MechanicalParams* mparams, sofa::core::MultiVecDerivId a)
        : MechanicalVisitor(mparams), a(a) //, f(f)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMass(simulation::Node* /*node*/, core::behavior::BaseMass* mass);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalAccFromFVisitor"; }
    virtual std::string getInfos() const { std::string name="a["+a.getName()+"] f["+mparams->f().getName()+"]"; return name; }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addWriteVector(a);
        addReadVector(mparams->f());
    }
#endif
};

class SOFA_SIMULATION_CORE_API MechanicalProjectJacobianMatrixVisitor : public MechanicalVisitor
{
public:
    sofa::core::MultiMatrixDerivId cId;
    SReal t;
    MechanicalProjectJacobianMatrixVisitor(const sofa::core::MechanicalParams* mparams,
                                           sofa::core::MultiMatrixDerivId c = sofa::core::MatrixDerivId::holonomicC(), SReal time = 0.0)
        : MechanicalVisitor(mparams), cId(c), t(time)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual Result fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c);


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalProjectJacobianMatrixVisitor"; }
    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
};

class SOFA_SIMULATION_CORE_API MechanicalProjectVelocityVisitor : public MechanicalVisitor
{
public:
    SReal t;
    sofa::core::MultiVecDerivId vel;
    MechanicalProjectVelocityVisitor(const sofa::core::MechanicalParams* mparams , SReal time=0,
                                     sofa::core::MultiVecDerivId v = sofa::core::VecDerivId::velocity())
        : MechanicalVisitor(mparams) , t(time),vel(v)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual Result fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c);


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalProjectVelocityVisitor"; }
    virtual std::string getInfos() const
    {
        std::string name="["+vel.getName()+"]"; return name;
    }
    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadWriteVector(vel);
    }
#endif
};

class SOFA_SIMULATION_CORE_API MechanicalProjectPositionVisitor : public MechanicalVisitor
{
public:
    SReal t;
    sofa::core::MultiVecCoordId pos;
    MechanicalProjectPositionVisitor(const sofa::core::MechanicalParams* mparams , SReal time=0,
                                     sofa::core::MultiVecCoordId x = sofa::core::VecCoordId::position())
        : MechanicalVisitor(mparams) , t(time), pos(x)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual Result fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c);


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalProjectPositionVisitor"; }
    virtual std::string getInfos() const
    {
        std::string name="["+pos.getName()+"]"; return name;
    }
    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadWriteVector(pos);
    }
#endif
};

/** Propagate positions  to all the levels of the hierarchy.
At each level, the mappings form the parent to the child is applied.
*/
class SOFA_SIMULATION_CORE_API MechanicalPropagatePositionVisitor : public MechanicalVisitor
{
public:
    SReal t;
    sofa::core::MultiVecCoordId x;
    bool ignoreMask;
    bool applyProjections;

    MechanicalPropagatePositionVisitor( const sofa::core::MechanicalParams* mparams, SReal time=0,
                                        sofa::core::MultiVecCoordId x = sofa::core::VecCoordId::position(), bool m=true);

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual Result fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c);
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalPropagatePositionVisitor";}
    virtual std::string getInfos() const
    {
        std::string name="x["+x.getName()+"]";
        if (ignoreMask) name += " Mask DISABLED";
        else            name += " Mask ENABLED";
        return name;
    }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadWriteVector(x);
    }
#endif
};
/** Propagate positions and velocities to all the levels of the hierarchy.
At each level, the mappings form the parent to the child is applied.
After the execution of this action, all the (mapped) degrees of freedom are consistent with the independent degrees of freedom.
This action is typically applied after time integration of the independent degrees of freedom.
*/
class SOFA_SIMULATION_CORE_API MechanicalPropagatePositionAndVelocityVisitor : public MechanicalVisitor
{
public:
    SReal currentTime;
    sofa::core::MultiVecCoordId x;
    sofa::core::MultiVecDerivId v;
    bool ignoreMask;
    bool applyProjections;

#ifdef SOFA_SUPPORT_MAPPED_MASS
    // compute the acceleration created by the input velocity and the derivative of the mapping
    MultiVecDerivId a;
    MechanicalPropagatePositionAndVelocityVisitor(
        const sofa::core::MechanicalParams* mparams, SReal time=0,
        sofa::core::MultiVecCoordId x = sofa::core::VecCoordId::position(), sofa::core::MultiVecDerivId v = sofa::core::VecDerivId::velocity(),
        sofa::core::MultiVecDerivId a = sofa::core::VecDerivId::dx() , bool m=true); //
#else
    MechanicalPropagatePositionAndVelocityVisitor(const sofa::core::MechanicalParams* mparams, SReal time=0,
                                                  sofa::core::MultiVecCoordId x = sofa::core::VecId::position(), sofa::core::MultiVecDerivId v = sofa::core::VecId::velocity(),
            bool m=true );
#endif
  

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual Result fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c);
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalPropagatePositionAndVelocityVisitor";}
    virtual std::string getInfos() const { std::string name="x["+x.getName()+"] v["+v.getName()+"]"; return name; }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadWriteVector(x);
        addReadWriteVector(v);
    }
#endif
};
/** Propagate velocities to all the levels of the hierarchy.
At each level, the mappings form the parent to the child is applied.
After the execution of this action, all the (mapped) degrees of freedom are consistent with the independent degrees of freedom.
*/
class SOFA_SIMULATION_CORE_API MechanicalPropagateVelocityVisitor : public MechanicalVisitor
{
public:
    SReal currentTime;
    sofa::core::MultiVecDerivId v;
    bool ignoreMask;    
    bool applyProjections;
    
#ifdef SOFA_SUPPORT_MAPPED_MASS
    // compute the acceleration created by the input velocity and the derivative of the mapping
    sofa::core::MultiVecDerivId a;
    MechanicalPropagateVelocityVisitor(
        const sofa::core::MechanicalParams* mparams, SReal time=0,
        sofa::core::MultiVecDerivId v = sofa::core::VecDerivId::velocity(),
        sofa::core::MultiVecDerivId a = sofa::core::VecDerivId::dx() , bool m=true);
#else
    MechanicalPropagateVelocityVisitor(const sofa::core::MechanicalParams* mparams, SReal time=0,
                                       sofa::core::MultiVecDerivId v = sofa::core::VecId::velocity(),
            bool m=true);
#endif

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual Result fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c);
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalPropagateVelocityVisitor";}
    virtual std::string getInfos() const { std::string name="v["+v.getName()+"]"; return name; }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadWriteVector(v);
    }
#endif
};

/**
* @brief Visitor class used to set positions and velocities of the top level MechanicalStates of the hierarchy.
*/
class SOFA_SIMULATION_CORE_API MechanicalSetPositionAndVelocityVisitor : public MechanicalVisitor
{
public:
    SReal t;
    sofa::core::MultiVecCoordId x;
    sofa::core::MultiVecDerivId v;

#ifdef SOFA_SUPPORT_MAPPED_MASS
    // compute the acceleration created by the input velocity and the derivative of the mapping
    sofa::core::MultiVecDerivId a;
    MechanicalSetPositionAndVelocityVisitor(const sofa::core::MechanicalParams* mparams ,
            SReal time=0, sofa::core::MultiVecCoordId x = sofa::core::VecCoordId::position() ,
            sofa::core::MultiVecDerivId v = sofa::core::VecDerivId::velocity() ,
            sofa::core::MultiVecDerivId a = sofa::core::VecDerivId::dx()); //
#else
    MechanicalSetPositionAndVelocityVisitor(const sofa::core::MechanicalParams* mparams ,SReal time=0,
                                            sofa::core::MultiVecCoordId x = sofa::core::VecCoordId::position(),
                                            sofa::core::MultiVecDerivId v = sofa::core::VecDerivId::velocity());
#endif

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalPropagatePositionAndVelocityVisitor";}
    virtual std::string getInfos() const { std::string name="x["+x.getName()+"] v["+v.getName()+"]"; return name; }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadVector(x);
        addReadVector(v);
    }
#endif
};



/** Reset the force in all the MechanicalModel
This action is typically applied before accumulating all the forces.
*/
class SOFA_SIMULATION_CORE_API MechanicalResetForceVisitor : public BaseMechanicalVisitor
{
public:
    sofa::core::MultiVecDerivId res;
    bool onlyMapped;

    MechanicalResetForceVisitor(const sofa::core::ExecParams* mparams,
                                sofa::core::MultiVecDerivId res, bool onlyMapped = false )
        : BaseMechanicalVisitor(mparams) , res(res), onlyMapped(onlyMapped)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const {  return "MechanicalResetForceVisitor";}
    virtual std::string getInfos() const
    {
        std::string name="["+res.getName()+"]";
        if (onlyMapped) name+= " Only Mapped";
        return name;
    }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addWriteVector(res);
    }
#endif
};

/** Accumulate the forces (internal and interactions).
This action is typically called after a MechanicalResetForceVisitor.
*/
class SOFA_SIMULATION_CORE_API MechanicalComputeForceVisitor : public MechanicalVisitor
{
public:
    sofa::core::MultiVecDerivId res;
    bool accumulate; ///< Accumulate everything back to the DOFs through the mappings
    bool neglectingCompliance; /// neglect Compliance?

    MechanicalComputeForceVisitor(const sofa::core::MechanicalParams* mparams,
                                  sofa::core::MultiVecDerivId res, bool accumulate = true,  bool neglectingCompliance = true )
        : MechanicalVisitor(mparams) , res(res), accumulate(accumulate), neglectingCompliance(neglectingCompliance)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* ff);
    virtual void bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const {return "MechanicalComputeForceVisitor";}
    virtual std::string getInfos() const
    {
        std::string name=std::string("[")+res.getName()+std::string("]");
        if (accumulate) name+= " Accumulating";
        else            name+= " Not Accumulating";
        return name;
    }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addWriteVector(res);
    }
#endif
};


/** Compute the variation of force corresponding to a variation of position.
This action is typically called after a MechanicalPropagateDxVisitor.
*/
class SOFA_SIMULATION_CORE_API MechanicalComputeDfVisitor : public MechanicalVisitor
{
public:
    sofa::core::MultiVecDerivId res;
    bool accumulate; ///< Accumulate everything back to the DOFs through the mappings
    MechanicalComputeDfVisitor(const sofa::core::MechanicalParams* mparams, sofa::core::MultiVecDerivId res)
        : MechanicalVisitor(mparams) , res(res), accumulate(true)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    MechanicalComputeDfVisitor(const sofa::core::MechanicalParams* mparams, sofa::core::MultiVecDerivId res, bool accumulate)
        : MechanicalVisitor(mparams) , res(res), accumulate(accumulate)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* ff);
    virtual void bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const {return "MechanicalComputeDfVisitor";}
    virtual std::string getInfos() const
    {
        std::string name="["+res.getName()+"]";
        if (accumulate) name+= " Accumulating";
        else            name+= " Not Accumulating";
        return name;
    }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addWriteVector(res);
    }
#endif
};




/** Compute the mapping geometric stiffness matrices.
This action must be call before BaseMapping::getK()
*/
class SOFA_SIMULATION_CORE_API MechanicalComputeGeometricStiffness : public MechanicalVisitor
{
public:
    sofa::core::ConstMultiVecDerivId childForce;
    MechanicalComputeGeometricStiffness(const sofa::core::MechanicalParams* mparams, sofa::core::ConstMultiVecDerivId childForce)
        : MechanicalVisitor(mparams) , childForce(childForce)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const {return "MechanicalComputeGeometricStiffness";}
    virtual std::string getInfos() const
    {
        std::string name="["+childForce.getName()+"]";
        return name;
    }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
//        addWriteVector(res);
    }
#endif
};


/** Accumulate the product of the system matrix by a given vector.
Typically used in implicit integration solved by a Conjugate Gradient algorithm.
The current value of the dx vector is used.
This action is typically called after a MechanicalPropagateDxAndResetForceVisitor.
*/
class SOFA_SIMULATION_CORE_API MechanicalAddMBKdxVisitor : public MechanicalVisitor
{
    sofa::core::MechanicalParams mparamsWithoutStiffness;
public:
    sofa::core::MultiVecDerivId res;
    bool accumulate; ///< Accumulate everything back to the DOFs through the mappings

    MechanicalAddMBKdxVisitor(const sofa::core::MechanicalParams* mparams,
                              sofa::core::MultiVecDerivId res, bool accumulate=true)
        : MechanicalVisitor(mparams) , res(res), accumulate(accumulate)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
        mparamsWithoutStiffness = *mparams;
        mparamsWithoutStiffness.setKFactor(0);
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* ff);
    virtual void bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalAddMBKdxVisitor"; }
    virtual std::string getInfos() const { std::string name= "["+res.getName()+"]"; return name; }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadWriteVector(res);
    }
#endif
};



class SOFA_SIMULATION_CORE_API MechanicalResetConstraintVisitor : public BaseMechanicalVisitor
{
public:
    //VecId res;
    MechanicalResetConstraintVisitor(const sofa::core::ExecParams* params)
        : BaseMechanicalVisitor(params)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdConstraintSet(simulation::Node* /*node*/, core::behavior::BaseConstraintSet* mm);

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalResetConstraintVisitor"; }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
};


class SOFA_SIMULATION_CORE_API MechanicalWriteLMConstraint : public BaseMechanicalVisitor
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

    virtual Result fwdConstraintSet(simulation::Node* /*node*/, core::behavior::BaseConstraintSet* c);
    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalWriteLMConstraint"; }
    virtual std::string getInfos() const
    {
        std::string name;
        if      (order == core::ConstraintParams::ACC)
            name= "["+sofa::core::VecId::dx().getName()+"]";
        else if (order == core::ConstraintParams::VEL)
            name= "["+sofa::core::VecId::velocity().getName()+"]";
        else if (order == core::ConstraintParams::POS)
            name= "["+sofa::core::VecId::position().getName()+"]";
        return name;
    }


    virtual void clear() {datasC.clear(); offset=0;}
    virtual const std::vector< core::behavior::BaseLMConstraint *> &getConstraints() const {return datasC;}
    virtual unsigned int numConstraint() {return (unsigned int)datasC.size();}

    virtual void setMultiVecId(core::MultiVecId i) {id=i;}
    core::MultiVecId getMultiVecId() const { return id; }


    virtual void setOrder(core::ConstraintParams::ConstOrder i) {order=i;}
    core::ConstraintParams::ConstOrder getOrder() const { return order; }

    virtual bool isThreadSafe() const
    {
        return false;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif

protected:
    unsigned int offset;
    sofa::core::ConstraintParams::ConstOrder order;
    core::MultiVecId id;
    helper::vector< core::behavior::BaseLMConstraint *> datasC;

};


class SOFA_SIMULATION_CORE_API MechanicalAccumulateConstraint : public BaseMechanicalVisitor
{
public:
    MechanicalAccumulateConstraint(const sofa::core::ConstraintParams* _cparams,
                                   sofa::core::MultiMatrixDerivId _res, unsigned int &_contactId)
        : BaseMechanicalVisitor(_cparams)
        , res(_res)
        , contactId(_contactId)
        , cparams(_cparams)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    const core::ConstraintParams* constraintParams() const { return cparams; }

    virtual Result fwdConstraintSet(simulation::Node* /*node*/, core::behavior::BaseConstraintSet* c);

    virtual void bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);

    /// This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalAccumulateConstraint"; }

    virtual bool isThreadSafe() const
    {
        return false;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif

protected:
    sofa::core::MultiMatrixDerivId res;
    unsigned int &contactId;
    const sofa::core::ConstraintParams *cparams;
};

class SOFA_SIMULATION_CORE_API MechanicalRenumberConstraint : public MechanicalVisitor
{
public:
    MechanicalRenumberConstraint(const sofa::core::MechanicalParams* mparams , const sofa::helper::vector<unsigned> &renumbering)
        : MechanicalVisitor(mparams) , renumbering(renumbering)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        mm->renumberConstraintId(renumbering);
        return RESULT_PRUNE;
    }
    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalRenumberConstraint"; }

    virtual bool isThreadSafe() const
    {
        return false;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif

protected:
    const sofa::helper::vector<unsigned> &renumbering;
};

/** Apply the constraints as filters to the given vector.
This works for simple independent constraints, like maintaining a fixed point.
*/
class SOFA_SIMULATION_CORE_API MechanicalApplyConstraintsVisitor : public MechanicalVisitor
{
public:
    sofa::core::MultiVecDerivId res;
    double **W;
    MechanicalApplyConstraintsVisitor(const sofa::core::MechanicalParams* mparams,
                                      sofa::core::MultiVecDerivId res, double **W = NULL)
        : MechanicalVisitor(mparams) , res(res), W(W)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/);
    virtual void bwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c);
    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalApplyConstraintsVisitor"; }
    virtual std::string getInfos() const { std::string name= "["+res.getName()+"]"; return name; }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadWriteVector(res);
    }
#endif
};

/** Visitor used to prepare a time integration step. Typically, does nothing.
*/
class SOFA_SIMULATION_CORE_API MechanicalBeginIntegrationVisitor : public BaseMechanicalVisitor
{
public:
    SReal dt;
    MechanicalBeginIntegrationVisitor (const sofa::core::ExecParams* _params, SReal _dt)
        : BaseMechanicalVisitor(_params) , dt(_dt)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalBeginIntegrationVisitor"; }

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
};

/** Visitor applied after a time step has been applied. Does typically nothing.
*/
class SOFA_SIMULATION_CORE_API MechanicalEndIntegrationVisitor : public BaseMechanicalVisitor
{
public:
    SReal dt;
    MechanicalEndIntegrationVisitor (const sofa::core::ExecParams* _params, SReal _dt)
        : BaseMechanicalVisitor(_params) , dt(_dt)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalEndIntegrationVisitor"; }

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
};

/** Visitor used to do a time integration step using OdeSolvers
*/
class SOFA_SIMULATION_CORE_API MechanicalIntegrationVisitor : public BaseMechanicalVisitor
{
public:
    SReal dt;
    MechanicalIntegrationVisitor (const sofa::core::ExecParams* m_params, SReal _dt)
        : BaseMechanicalVisitor(m_params) , dt(_dt)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdOdeSolver(simulation::Node* node, core::behavior::OdeSolver* obj);
    virtual Result fwdInteractionForceField(simulation::Node*, core::behavior::BaseInteractionForceField* obj);
    virtual void bwdOdeSolver(simulation::Node* /*node*/, core::behavior::OdeSolver* /*obj*/)
    {
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalIntegrationVisitor"; }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
};



/** Accumulate only the contact forces computed in applyContactForce.
This action is typically called after a MechanicalResetForceVisitor.
*/
class SOFA_SIMULATION_CORE_API MechanicalComputeContactForceVisitor : public MechanicalVisitor
{
public:
    sofa::core::MultiVecDerivId res;
    MechanicalComputeContactForceVisitor(const sofa::core::MechanicalParams* mparams,
                                         sofa::core::MultiVecDerivId res )
        : MechanicalVisitor(mparams) , res(res)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual void bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { std::string name= "MechanicalComputeContactForceVisitor["+res.getName()+"]"; return name.c_str(); }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadWriteVector(res);
    }
#endif
};

/** Add dt*mass*Gravity to the velocity
This is called if the mass wants to be added separately to the mm from the other forces
*/
class SOFA_SIMULATION_CORE_API MechanicalAddSeparateGravityVisitor : public MechanicalVisitor
{
public:
    sofa::core::MultiVecDerivId res;
    MechanicalAddSeparateGravityVisitor(const sofa::core::MechanicalParams* m_mparams,
                                        sofa::core::MultiVecDerivId res )
        : MechanicalVisitor(m_mparams) , res(res)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    /// Process the BaseMass
    virtual Result fwdMass(simulation::Node* /*node*/, core::behavior::BaseMass* mass);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalAddSeparateGravityVisitor"; }
    virtual std::string getInfos() const { std::string name= "["+res.getName()+"]"; return name; }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadWriteVector(res);
    }
#endif
};

/** Find mechanical particles hit by the given ray.
*
*  A mechanical particle is defined as a 2D or 3D, position or rigid DOF
*  which is linked to the free mechanical DOFs by mechanical mappings
*/
class SOFA_SIMULATION_CORE_API MechanicalPickParticlesVisitor : public BaseMechanicalVisitor
{
public:
    defaulttype::Vec3d rayOrigin, rayDirection;
    double radius0, dRadius;
    sofa::core::objectmodel::Tag tagNoPicking;
    typedef std::multimap< double, std::pair<sofa::core::behavior::BaseMechanicalState*, int> > Particles;
    Particles particles;
    MechanicalPickParticlesVisitor(const sofa::core::ExecParams* mparams, const defaulttype::Vec3d& origin, const defaulttype::Vec3d& direction, double r0=0.001, double dr=0.0, sofa::core::objectmodel::Tag tag = sofa::core::objectmodel::Tag("NoPicking") )
        : BaseMechanicalVisitor(mparams) , rayOrigin(origin), rayDirection(direction), radius0(r0), dRadius(dr), tagNoPicking(tag)
    {
    }

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalPickParticles"; }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif

    /// get the closest pickable particle
    void getClosestParticle( core::behavior::BaseMechanicalState*& mstate, unsigned int& indexCollisionElement, defaulttype::Vector3& point, SReal& rayLength );


};

/** Find mechanical particles hit by the given ray on dof containing one tag or all provided by a tag list
*
*  A mechanical particle is defined as a 2D or 3D, position or rigid DOF
*  which is linked to the free mechanical DOFs by mechanical mappings
*/
class SOFA_SIMULATION_CORE_API MechanicalPickParticlesWithTagsVisitor : public BaseMechanicalVisitor
{
public:
	defaulttype::Vec3d rayOrigin, rayDirection;
	double radius0, dRadius;
	std::list<sofa::core::objectmodel::Tag> tags;
	bool mustContainAllTags;
	typedef std::multimap< double, std::pair<sofa::core::behavior::BaseMechanicalState*, int> > Particles;
	Particles particles;
	MechanicalPickParticlesWithTagsVisitor(const sofa::core::ExecParams* mparams, const defaulttype::Vec3d& origin, const defaulttype::Vec3d& direction, double r0=0.001, double dr=0.0, std::list<sofa::core::objectmodel::Tag> _tags = std::list<sofa::core::objectmodel::Tag>(), bool _mustContainAllTags = false)
		: BaseMechanicalVisitor(mparams) , rayOrigin(origin), rayDirection(direction), radius0(r0), dRadius(dr), tags(_tags), mustContainAllTags(_mustContainAllTags)
	{
	}

	virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
	virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
	virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

	/// Return a class name for this visitor
	/// Only used for debugging / profiling purposes
	virtual const char* getClassName() const { return "MechanicalPickParticlesWithTags"; }

#ifdef SOFA_DUMP_VISITOR_INFO
	void setReadWriteVectors()
	{
	}
#endif

	/// get the closest pickable particle
	void getClosestParticle( core::behavior::BaseMechanicalState*& mstate, unsigned int& indexCollisionElement, defaulttype::Vector3& point, SReal& rayLength );

private:

    // this function checks if the component must be included in the pick process according to its tags
    bool isComponentTagIncluded(const core::behavior::BaseMechanicalState* mm);

};



/** Get vector size */
class SOFA_SIMULATION_CORE_API MechanicalVSizeVisitor : public BaseMechanicalVisitor
{
public:
    sofa::core::ConstMultiVecId v;
    size_t* result;

    MechanicalVSizeVisitor(const sofa::core::ExecParams* params, size_t* result, sofa::core::ConstMultiVecId v)
        : BaseMechanicalVisitor(params), v(v), result(result)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual Result fwdMechanicalState(simulation::Node*, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMappedMechanicalState(simulation::Node*, core::behavior::BaseMechanicalState* mm);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalVSizeVisitor";}
    virtual std::string getInfos() const
    {
        std::string name = "[" + v.getName() + "]";
        return name;
    }
    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return false;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addReadVector(v);
    }
#endif
};



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_SIMULATION_MECHANICALVISITOR_CPP)
extern template class MechanicalVAvailVisitor<sofa::core::V_COORD>;
extern template class MechanicalVAvailVisitor<sofa::core::V_DERIV>;
extern template class MechanicalVAllocVisitor<sofa::core::V_COORD>;
extern template class MechanicalVAllocVisitor<sofa::core::V_DERIV>;
extern template class MechanicalVReallocVisitor<sofa::core::V_COORD>;
extern template class MechanicalVReallocVisitor<sofa::core::V_DERIV>;
extern template class MechanicalVFreeVisitor<sofa::core::V_COORD>;
extern template class MechanicalVFreeVisitor<sofa::core::V_DERIV>;
extern template class MechanicalVInitVisitor<sofa::core::V_COORD>;
extern template class MechanicalVInitVisitor<sofa::core::V_DERIV>;
#endif

} // namespace simulation

} // namespace sofa

#endif
