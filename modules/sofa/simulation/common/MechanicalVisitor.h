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
#ifndef SOFA_SIMULATION_MECHANICALVISITOR_H
#define SOFA_SIMULATION_MECHANICALVISITOR_H
#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/simulation/common/Visitor.h>
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
#ifdef SOFA_HAVE_EIGEN2
//TO REMOVE ONCE THE CONVERGENCE IS DONE
#include <sofa/core/behavior/BaseLMConstraint.h>
#endif
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

using std::cerr;
using std::endl;

using namespace sofa::core;
using namespace sofa::defaulttype;

typedef std::map<core::objectmodel::BaseContext*, double> MultiNodeDataMap;

/** Base class for easily creating new actions for mechanical simulation.

During the first traversal (top-down), method processNodeTopDown(simulation::Node*) is applied to each simulation::Node.
Each component attached to this node is processed using the appropriate method, prefixed by fwd.
During the second traversal (bottom-up), method processNodeBottomUp(simulation::Node*) is applied to each simulation::Node.
Each component attached to this node is processed using the appropriate method, prefixed by bwd.
The default behavior of the fwd* and bwd* is to do nothing.
Derived actions typically overload these methods to implement the desired processing.

*/
class SOFA_SIMULATION_COMMON_API BaseMechanicalVisitor : public Visitor
{

protected:
    bool prefetching;
    simulation::Node* root; ///< root node from which the visitor was executed
    double* rootData; ///< data for root node
    MultiNodeDataMap* nodeMap;

    /// Temporary node -> double* map to sequential traversals requiring node-specific data
    std::map<simulation::Node*, double*> tmpNodeDataMap;

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
        , prefetching(false), root(NULL), rootData(NULL), nodeMap(NULL)
    {}

    BaseMechanicalVisitor& setNodeMap(MultiNodeDataMap* m) { nodeMap = m; return *this; }

    MultiNodeDataMap* getNodeMap() { return nodeMap; }

    /// Return true if this visitor need to read the node-specific data if given
    virtual bool readNodeData() const
    { return false; }

    /// Return true if this visitor need to write to the node-specific data if given
    virtual bool writeNodeData() const
    { return false; }

    virtual void setNodeData(simulation::Node* /*node*/, double* nodeData, const double* parentData)
    {
        *nodeData = (parentData == NULL) ? 0.0 : *parentData;
    }

    virtual void addNodeData(simulation::Node* /*node*/, double* parentData, const double* nodeData)
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

    //virtual void execute(core::objectmodel::BaseContext* node, bool doPrefetch) { Visitor::execute(node, doPrefetch); }
    //virtual void execute(core::objectmodel::BaseContext* node) { Visitor::execute(node, true); }


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
    virtual Result fwdInteractionProjectiveConstraintSet(simulation::Node* node, core::behavior::BaseInteractionProjectiveConstraintSet* c)
    {
        return fwdInteractionProjectiveConstraintSet(node, c);
    }

    /// Process all the InteractionConstraint
    virtual Result fwdInteractionConstraint(simulation::Node* node, core::behavior::BaseInteractionConstraint* c)
    {
        return fwdInteractionConstraint(node, c);
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
    virtual void addReadVector(ConstMultiVecId id) {  readVector.push_back(id);  }
    virtual void addWriteVector(MultiVecId id) {  writeVector.push_back(id);  }
    virtual void addReadWriteVector(MultiVecId id) {  readVector.push_back(ConstMultiVecId(id)); writeVector.push_back(id);  }
    void printReadVectors(core::behavior::BaseMechanicalState* mm);
    void printReadVectors(simulation::Node* node, core::objectmodel::BaseObject* obj);
    void printWriteVectors(core::behavior::BaseMechanicalState* mm);
    void printWriteVectors(simulation::Node* node, core::objectmodel::BaseObject* obj);
protected:
    sofa::helper::vector< ConstMultiVecId > readVector;
    sofa::helper::vector< MultiVecId > writeVector;
#endif
};

class SOFA_SIMULATION_COMMON_API MechanicalVisitor : public BaseMechanicalVisitor
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
class SOFA_SIMULATION_COMMON_API MechanicalGetDimensionVisitor : public MechanicalVisitor
{
public:
    MechanicalGetDimensionVisitor(double* result, const core::MechanicalParams* mparams)
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
template <VecType vtype>
class SOFA_SIMULATION_COMMON_API MechanicalVAvailVisitor : public BaseMechanicalVisitor
{
public:
    typedef sofa::core::TVecId<vtype,V_WRITE> MyVecId;
    typedef sofa::core::TMultiVecId<vtype,V_WRITE> MyMultiVecId;
    typedef std::set<sofa::core::BaseState*> StateSet;
    MyVecId& v;
    StateSet states;
    MechanicalVAvailVisitor( MyVecId& v, const core::ExecParams* params)
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
    virtual std::string getInfos() const { std::string name="[" + v.getName() + "]"; return name;  }
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

/** Reserve an auxiliary vector identified by a symbolic constant.
*/
template <VecType vtype>
class SOFA_SIMULATION_COMMON_API MechanicalVAllocVisitor : public BaseMechanicalVisitor
{
public:
    typedef sofa::core::TMultiVecId<vtype, V_WRITE> MyMultiVecId;
    MyMultiVecId v;
    MechanicalVAllocVisitor( MyMultiVecId v, const core::ExecParams* params )
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
    virtual std::string getInfos() const {std::string name="[" + v.getName() + "]"; return name;}
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

/** Free an auxiliary vector identified by a symbolic constant */
template< VecType vtype >
class SOFA_SIMULATION_COMMON_API MechanicalVFreeVisitor : public BaseMechanicalVisitor
{
public:
    typedef sofa::core::TMultiVecId<vtype,V_WRITE> MyMultiVecId;
    MyMultiVecId v;
    MechanicalVFreeVisitor( MyMultiVecId v, const sofa::core::ExecParams* params)
        : BaseMechanicalVisitor(params) , v(v)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalVFreeVisitor"; }
    virtual std::string getInfos() const {std::string name="[" + v.getName() + "]"; return name;}
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
class SOFA_SIMULATION_COMMON_API MechanicalVOpVisitor : public BaseMechanicalVisitor
{
public:
    MultiVecId v;
    ConstMultiVecId a;
    ConstMultiVecId b;
    double f;
    bool mapped;
    MechanicalVOpVisitor(MultiVecId v, ConstMultiVecId a = ConstMultiVecId::null(), ConstMultiVecId b = ConstMultiVecId::null(), double f=1.0, const sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance() )
        : BaseMechanicalVisitor(params) , v(v), a(a), b(b), f(f), mapped(false)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    MechanicalVOpVisitor& setMapped(bool m = true) { mapped = m; return *this; }

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
class SOFA_SIMULATION_COMMON_API MechanicalVMultiOpVisitor : public BaseMechanicalVisitor
{
public:
    typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
    bool mapped;
    //     MechanicalVMultiOpVisitor()
    //     {}
    MechanicalVMultiOpVisitor(const VMultiOp& o, const sofa::core::ExecParams* params)
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
            const helper::vector< std::pair< core::ConstMultiVecId, double > >& operands = it->second;
            int nop = operands.size();
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
class SOFA_SIMULATION_COMMON_API MechanicalVDotVisitor : public BaseMechanicalVisitor
{
public:
    ConstMultiVecId a;
    ConstMultiVecId b;
    MechanicalVDotVisitor(ConstMultiVecId a, ConstMultiVecId b, double* t, const sofa::core::ExecParams* params)
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

/** Apply a hypothetical displacement.
This action does not modify the state (i.e. positions and velocities) of the objects.
It is typically applied before a MechanicalComputeDfVisitor, in order to compute the df corresponding to a given dx (i.e. apply stiffness).
Dx is propagated to all the layers through the mappings.
*/
class SOFA_SIMULATION_COMMON_API MechanicalPropagateDxVisitor : public MechanicalVisitor
{
public:
    MultiVecDerivId dx;

    bool ignoreMask;
    bool ignoreFlag;
    MechanicalPropagateDxVisitor( MultiVecDerivId dx, bool m, bool f = false, const sofa::core::MechanicalParams* mparams = sofa::core::MechanicalParams::defaultInstance() )
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

/** V is propagated to all the layers through the mappings.
*/
class SOFA_SIMULATION_COMMON_API MechanicalPropagateVVisitor : public MechanicalVisitor
{
public:
    MultiVecDerivId v;
    bool ignoreMask;

    MechanicalPropagateVVisitor(const sofa::core::MechanicalParams* mparams , MultiVecDerivId _v, bool m)
        : MechanicalVisitor(mparams) , v(_v), ignoreMask(m)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalPropagateVVisitor"; }
    virtual std::string getInfos() const
    {
        std::string name="["+v.getName()+"]"; return name;
    }
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


/** Same as MechanicalPropagateDxVisitor followed by MechanicalResetForceVisitor
*/
class SOFA_SIMULATION_COMMON_API MechanicalPropagateDxAndResetForceVisitor : public MechanicalVisitor
{
public:
    MultiVecDerivId dx,f;
    bool ignoreMask;

    MechanicalPropagateDxAndResetForceVisitor(MultiVecDerivId dx, MultiVecDerivId f, bool m, const sofa::core::MechanicalParams* mparams = sofa::core::MechanicalParams::defaultInstance())
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


class SOFA_SIMULATION_COMMON_API MechanicalPropagateXVisitor : public MechanicalVisitor
{
public:
    MultiVecCoordId x;
    bool ignoreMask;

    MechanicalPropagateXVisitor( MultiVecCoordId x, bool m, const sofa::core::MechanicalParams* mparams = sofa::core::MechanicalParams::defaultInstance() )
        : MechanicalVisitor(mparams) , x(x), ignoreMask(m)
    {}
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalPropagateXVisitor"; }
    virtual std::string getInfos() const { std::string name= "X["+x.getName()+"]"; return name;}


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


/** Same as MechanicalPropagateXVisitor followed by MechanicalResetForceVisitor
*/
class SOFA_SIMULATION_COMMON_API MechanicalPropagateXAndResetForceVisitor : public MechanicalVisitor
{
public:
    MultiVecCoordId x;
    MultiVecDerivId f;
    bool ignoreMask;

    MechanicalPropagateXAndResetForceVisitor(MultiVecCoordId x, MultiVecDerivId f, bool m, const sofa::core::MechanicalParams* mparams = sofa::core::MechanicalParams::defaultInstance())
        : MechanicalVisitor(mparams) , x(x), f(f), ignoreMask(m)
    {}
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalPropagateXAndResetForceVisitor"; }

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



class SOFA_SIMULATION_COMMON_API MechanicalPropagateAndAddDxVisitor : public MechanicalVisitor
{
public:
    MultiVecDerivId dx, v;
    bool ignoreMask;

    MechanicalPropagateAndAddDxVisitor(const sofa::core::MechanicalParams* mparams , MultiVecDerivId dx = VecDerivId::dx(), MultiVecDerivId v =VecDerivId::velocity(), bool m=true)
        : MechanicalVisitor(mparams) , dx(dx) , v(v),ignoreMask(m)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalPropagateAndAddDxVisitor"; }
    virtual std::string getInfos() const { std::string name= "["+dx.getName()+"]"; return name; }


    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

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
        addReadWriteVector(v);
        addReadWriteVector(dx);
    }
#endif
};


/** Accumulate the product of the mass matrix by a given vector.
Typically used in implicit integration solved by a Conjugate Gradient algorithm.
Note that if a dx vector is given, it is used and propagated by the mappings, Otherwise the current value is used.
*/
class SOFA_SIMULATION_COMMON_API MechanicalAddMDxVisitor : public MechanicalVisitor
{
public:
    MultiVecDerivId res;
    MultiVecDerivId dx;
    double factor;
    MechanicalAddMDxVisitor(MultiVecDerivId res, MultiVecDerivId dx, double factor, const sofa::core::MechanicalParams* mparams)
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
#else
    virtual Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/);
#endif
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm);

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
class SOFA_SIMULATION_COMMON_API MechanicalAccFromFVisitor : public MechanicalVisitor
{
public:
    MultiVecDerivId a;
    //ConstMultiVecDerivId f; // in MechanicalParams
    MechanicalAccFromFVisitor(MultiVecDerivId a, const sofa::core::MechanicalParams* mparams)
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

class SOFA_SIMULATION_COMMON_API MechanicalProjectJacobianMatrixVisitor : public MechanicalVisitor
{
public:
    MultiMatrixDerivId cId;
    double t;
    MechanicalProjectJacobianMatrixVisitor(const sofa::core::MechanicalParams* mparams, MultiMatrixDerivId c = MatrixDerivId::holonomicC(), double time = 0.0)
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

class SOFA_SIMULATION_COMMON_API MechanicalProjectVelocityVisitor : public MechanicalVisitor
{
public:
    double t;
    MultiVecDerivId vel;
    MechanicalProjectVelocityVisitor(const sofa::core::MechanicalParams* mparams , double time=0, MultiVecDerivId v = VecDerivId::velocity())
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

class SOFA_SIMULATION_COMMON_API MechanicalProjectPositionVisitor : public MechanicalVisitor
{
public:
    double t;
    MultiVecCoordId pos;
    MechanicalProjectPositionVisitor(const sofa::core::MechanicalParams* mparams , double time=0, MultiVecCoordId x = VecCoordId::position())
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
class SOFA_SIMULATION_COMMON_API MechanicalPropagatePositionVisitor : public MechanicalVisitor
{
public:
    double t;
    MultiVecCoordId x;
    bool ignoreMask;

    MechanicalPropagatePositionVisitor( double time=0, MultiVecCoordId x = VecCoordId::position(), bool m=true, const sofa::core::MechanicalParams* mparams = sofa::core::MechanicalParams::defaultInstance());

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
class SOFA_SIMULATION_COMMON_API MechanicalPropagatePositionAndVelocityVisitor : public MechanicalVisitor
{
public:
    double t;
    MultiVecCoordId x;
    MultiVecDerivId v;

    MechanicalPropagatePositionAndVelocityVisitor(const sofa::core::MechanicalParams* mparams);
#ifdef SOFA_SUPPORT_MAPPED_MASS
    // compute the acceleration created by the input velocity and the derivative of the mapping
    MultiVecDerivId a;
    MechanicalPropagatePositionAndVelocityVisitor(double time=0,
            MultiVecCoordId x = VecCoordId::position(), MultiVecDerivId v = VecDerivId::velocity(),
            MultiVecDerivId a = VecDerivId::dx() , bool m=true,
            const sofa::core::MechanicalParams* mparams = sofa::core::MechanicalParams::defaultInstance()); //
#else
    MechanicalPropagatePositionAndVelocityVisitor(double time=0, VecCoordId x = VecId::position(), VecDerivId v = VecId::velocity(),
            bool m=true, const sofa::core::MechanicalParams* mparams = sofa::core::MechanicalParams::defaultInstance() );
#endif
    bool ignoreMask;

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


/**
* @brief Visitor class used to set positions and velocities of the top level MechanicalStates of the hierarchy.
*/
class SOFA_SIMULATION_COMMON_API MechanicalSetPositionAndVelocityVisitor : public MechanicalVisitor
{
public:
    double t;
    MultiVecCoordId x;
    MultiVecDerivId v;

#ifdef SOFA_SUPPORT_MAPPED_MASS
    // compute the acceleration created by the input velocity and the derivative of the mapping
    MultiVecDerivId a;
    MechanicalSetPositionAndVelocityVisitor(const sofa::core::MechanicalParams* mparams ,
            double time=0, MultiVecCoordId x = VecCoordId::position() ,
            MultiVecDerivId v = VecDerivId::velocity() ,
            MultiVecDerivId a = VecDerivId::dx()); //
#else
    MechanicalSetPositionAndVelocityVisitor(const sofa::core::MechanicalParams* mparams ,
            double time=0, MultiVecCoordId x = VecCoordId::position(), MultiVecDerivId v = VecDerivId::velocity());
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
class SOFA_SIMULATION_COMMON_API MechanicalResetForceVisitor : public BaseMechanicalVisitor
{
public:
    MultiVecDerivId res;
    bool onlyMapped;

    MechanicalResetForceVisitor(MultiVecDerivId res, bool onlyMapped = false, const sofa::core::ExecParams* mparams = sofa::core::ExecParams::defaultInstance() )
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
class SOFA_SIMULATION_COMMON_API MechanicalComputeForceVisitor : public MechanicalVisitor
{
public:
    MultiVecDerivId res;
    bool accumulate; ///< Accumulate everything back to the DOFs through the mappings

    MechanicalComputeForceVisitor(MultiVecDerivId res, bool accumulate = true, const sofa::core::MechanicalParams* mparams = sofa::core::MechanicalParams::defaultInstance() )
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
class SOFA_SIMULATION_COMMON_API MechanicalComputeDfVisitor : public MechanicalVisitor
{
public:
    MultiVecDerivId res;
    bool accumulate; ///< Accumulate everything back to the DOFs through the mappings
    MechanicalComputeDfVisitor(MultiVecDerivId res, const sofa::core::MechanicalParams* mparams)
        : MechanicalVisitor(mparams) , res(res), accumulate(true)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    MechanicalComputeDfVisitor(MultiVecDerivId res, bool accumulate, const sofa::core::MechanicalParams* mparams)
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


/** Accumulate the product of the system matrix by a given vector.
Typically used in implicit integration solved by a Conjugate Gradient algorithm.
The current value of the dx vector is used.
This action is typically called after a MechanicalPropagateDxAndResetForceVisitor.
*/
class SOFA_SIMULATION_COMMON_API MechanicalAddMBKdxVisitor : public MechanicalVisitor
{
public:
    MultiVecDerivId res;
    bool accumulate; ///< Accumulate everything back to the DOFs through the mappings

    MechanicalAddMBKdxVisitor(MultiVecDerivId res, const sofa::core::MechanicalParams* mparams)
        : MechanicalVisitor(mparams) , res(res), accumulate(true)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    MechanicalAddMBKdxVisitor(MultiVecDerivId res, bool accumulate, const sofa::core::MechanicalParams* mparams)
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

class SOFA_SIMULATION_COMMON_API MechanicalResetConstraintVisitor : public BaseMechanicalVisitor
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

#ifdef SOFA_HAVE_EIGEN2

//class SOFA_SIMULATION_COMMON_API MechanicalExpressJacobianVisitor: public MechanicalVisitor
//{
//public:
//    MechanicalExpressJacobianVisitor(simulation::Node* n);
//    virtual void bwdMechanicalMapping(simulation::Node* node, core::BaseMapping* map);
//    virtual Result fwdLMConstraint(simulation::Node* node, core::behavior::BaseLMConstraint* c);

//    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
//    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
//    {
//        return false; // !map->isMechanical();
//    }
//    /// Return a class name for this visitor
//    /// Only used for debugging / profiling purposes
//    virtual const char* getClassName() const { return "MechanicalExpressJacobianVisitor"; }
//    virtual bool isThreadSafe() const{ return false;}
//#ifdef SOFA_DUMP_VISITOR_INFO
//    void setReadWriteVectors()
//    {
//    }
//#endif
//  protected:
//    unsigned int constraintId;
//};



//class SOFA_SIMULATION_COMMON_API MechanicalSolveLMConstraintVisitor: public MechanicalVisitor
//{
// public:
// MechanicalSolveLMConstraintVisitor(VecId v,bool priorStatePropagation,bool updateVelocity=true):state(v), propagateState(priorStatePropagation),isPositionChangeUpdateVelocity(updateVelocity){
//#ifdef SOFA_DUMP_VISITOR_INFO
//    setReadWriteVectors();
//#endif
//        };


//  virtual Result fwdConstraintSolver(simulation::Node* /*node*/, core::behavior::ConstraintSolver* s);
//    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
//  virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
//    {
//        return false; // !map->isMechanical();
//    }

//  /// Return a class name for this visitor
//  /// Only used for debugging / profiling purposes
//  virtual const char* getClassName() const { return "MechanicalSolveLMConstraintVisitor"; }
//  virtual std::string getInfos() const { std::string name= "["+state.getName()+"]"; return name; }

//  virtual bool isThreadSafe() const
//  {
//    return false;
//  }
//#ifdef SOFA_DUMP_VISITOR_INFO
//    void setReadWriteVectors()
//    {
//        addReadWriteVector(state);
//    }
//#endif
//    VecId state;
//    bool propagateState;
//    bool isPositionChangeUpdateVelocity;
//};

class SOFA_SIMULATION_COMMON_API MechanicalWriteLMConstraint : public BaseMechanicalVisitor
{
public:
    MechanicalWriteLMConstraint(const sofa::core::ExecParams* params)
        : BaseMechanicalVisitor(params)
        , offset(0)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    };

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
            name= "["+VecId::dx().getName()+"]";
        else if (order == core::ConstraintParams::VEL)
            name= "["+VecId::velocity().getName()+"]";
        else if (order == core::ConstraintParams::POS)
            name= "["+VecId::position().getName()+"]";
        return name;
    }


    virtual void clear() {datasC.clear(); offset=0;}
    virtual const std::vector< core::behavior::BaseLMConstraint *> &getConstraints() const {return datasC;}
    virtual unsigned int numConstraint() {return datasC.size();}

    virtual void setOrder(core::ConstraintParams::ConstOrder i) {order=i;}
    core::ConstraintParams::ConstOrder getOrder() const { return order; }

    virtual void setVecId(core::VecId i) {id=i;}
    core::VecId getVecId() const { return id; }

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
    core::ConstraintParams::ConstOrder order;
    core::VecId id;
    helper::vector< core::behavior::BaseLMConstraint *> datasC;

};

#endif

class SOFA_SIMULATION_COMMON_API MechanicalAccumulateConstraint : public BaseMechanicalVisitor
{
public:
    MechanicalAccumulateConstraint(MultiMatrixDerivId _res, unsigned int &_contactId, const sofa::core::ConstraintParams* _cparams = sofa::core::ConstraintParams::defaultInstance())
        : BaseMechanicalVisitor(cparams)
        , res(_res)
        , contactId(_contactId)
        , cparams(_cparams)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

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
    MultiMatrixDerivId res;
    unsigned int &contactId;
    const sofa::core::ConstraintParams *cparams;
};

class SOFA_SIMULATION_COMMON_API MechanicalRenumberConstraint : public MechanicalVisitor
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
class SOFA_SIMULATION_COMMON_API MechanicalApplyConstraintsVisitor : public MechanicalVisitor
{
public:
    MultiVecDerivId res;
    double **W;
    MechanicalApplyConstraintsVisitor(MultiVecDerivId res, double **W = NULL, const sofa::core::MechanicalParams* mparams = sofa::core::MechanicalParams::defaultInstance())
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
class SOFA_SIMULATION_COMMON_API MechanicalBeginIntegrationVisitor : public BaseMechanicalVisitor
{
public:
    double dt;
    MechanicalBeginIntegrationVisitor (double _dt, const sofa::core::ExecParams* _params)
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
class SOFA_SIMULATION_COMMON_API MechanicalEndIntegrationVisitor : public BaseMechanicalVisitor
{
public:
    double dt;
    MechanicalEndIntegrationVisitor (double _dt, const sofa::core::ExecParams* _params)
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
class SOFA_SIMULATION_COMMON_API MechanicalIntegrationVisitor : public BaseMechanicalVisitor
{
public:
    double dt;
    MechanicalIntegrationVisitor (double _dt, const sofa::core::ExecParams* m_params)
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



// ACTION : Compute Compliance on mechanical models
class SOFA_SIMULATION_COMMON_API MechanicalComputeComplianceVisitor : public MechanicalVisitor
{
public:
    MechanicalComputeComplianceVisitor(const sofa::core::MechanicalParams* m_mparams , double **W)
        : MechanicalVisitor(m_mparams) , _W(W)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms);
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms);
    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalComputeComplianceVisitor"; }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
private:
    double **_W;
};


/** Accumulate only the contact forces computed in applyContactForce.
This action is typically called after a MechanicalResetForceVisitor.
*/
class SOFA_SIMULATION_COMMON_API MechanicalComputeContactForceVisitor : public MechanicalVisitor
{
public:
    MultiVecDerivId res;
    MechanicalComputeContactForceVisitor(MultiVecDerivId res, const sofa::core::MechanicalParams* mparams = sofa::core::MechanicalParams::defaultInstance() )
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
class SOFA_SIMULATION_COMMON_API MechanicalAddSeparateGravityVisitor : public MechanicalVisitor
{
public:
    MultiVecDerivId res;
    MechanicalAddSeparateGravityVisitor(MultiVecDerivId res, const sofa::core::MechanicalParams* m_mparams = sofa::core::MechanicalParams::defaultInstance() )
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
class SOFA_SIMULATION_COMMON_API MechanicalPickParticlesVisitor : public BaseMechanicalVisitor
{
public:
    defaulttype::Vec3d rayOrigin, rayDirection;
    double radius0, dRadius;
    std::multimap< double, std::pair<sofa::core::behavior::BaseMechanicalState*, int> > particles;
    MechanicalPickParticlesVisitor(const defaulttype::Vec3d& origin, const defaulttype::Vec3d& direction, double r0=0.001, double dr=0.0, const sofa::core::ExecParams* mparams = sofa::core::ExecParams::defaultInstance() )
        : BaseMechanicalVisitor(mparams) , rayOrigin(origin), rayDirection(direction), radius0(r0), dRadius(dr)
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
};
#if defined(WIN32) && !defined(SOFA_BUILD_SIMULATION_COMMON)
extern template class MechanicalVAvailVisitor<V_COORD>;
extern template class MechanicalVAvailVisitor<V_DERIV>;
extern template class MechanicalVAllocVisitor<V_COORD>;
extern template class MechanicalVAllocVisitor<V_DERIV>;
extern template class MechanicalVFreeVisitor<V_COORD>;
extern template class MechanicalVFreeVisitor<V_DERIV>;
#endif

} // namespace simulation

} // namespace sofa

#endif
