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
#ifndef SOFA_SIMULATION_MECHANICALVISITOR_H
#define SOFA_SIMULATION_MECHANICALVISITOR_H

#include <sofa/simulation/Visitor.h>
#include <sofa/core/fwd.h>
#include <sofa/core/VecId.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/simulation/BaseMechanicalVisitor.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MechanicalParams.h>

namespace sofa
{

namespace simulation
{

class SOFA_SIMULATION_CORE_API MechanicalVisitor : public BaseMechanicalVisitor
{

protected:
    const sofa::core::MechanicalParams* mparams;

public:

    MechanicalVisitor(const sofa::core::MechanicalParams* m_mparams)
        : BaseMechanicalVisitor(sofa::core::mechanicalparams::castToExecParams(m_mparams))
        , mparams(m_mparams)
    {}
};

/** Compute the total number of DOFs */
class SOFA_SIMULATION_CORE_API MechanicalGetDimensionVisitor : public MechanicalVisitor
{
public:
    MechanicalGetDimensionVisitor(const sofa::core::MechanicalParams* mparams, SReal* result)
        : MechanicalVisitor(mparams)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
        rootData = result;
    }

    Result fwdMechanicalState(VisitorContext* ctx, sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalGetDimensionVisitor";}

    bool writeNodeData() const override
    {
        return true;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    MechanicalVAvailVisitor( const sofa::core::ExecParams* params, MyVecId& v)
        : BaseMechanicalVisitor(params), v(v)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalVAvailVisitor"; }
    virtual std::string getInfos() const override;
    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return false;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    MechanicalVInitVisitor(const sofa::core::ExecParams* params, DestMultiVecId _vDest, SrcMultiVecId _vSrc = SrcMultiVecId::null(), bool propagate=false)
        : BaseMechanicalVisitor(params)
        , vDest(_vDest)
        , vSrc(_vSrc)
        , m_propagate(propagate)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override
    {
        return false;
    }

    Result fwdMechanicalState(simulation::Node* node,sofa::core::behavior::BaseMechanicalState* mm) override;

    Result fwdMappedMechanicalState(simulation::Node* node,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override
    {
        return "MechanicalVInitVisitor";
    }

    virtual std::string getInfos() const override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return false;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    MechanicalVAllocVisitor( const sofa::core::ExecParams* params, MyMultiVecId v )
        : BaseMechanicalVisitor(params) , v(v)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalVAllocVisitor"; }
    virtual std::string getInfos() const override;
    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    MechanicalVReallocVisitor(const sofa::core::ExecParams* params, DestMultiVecId *v, bool interactionForceField=false, bool propagate=false)
        : BaseMechanicalVisitor(params)
        , v(v)
        , m_propagate(propagate)
        , m_interactionForceField(interactionForceField)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override
    {
        return false;
    }

    Result fwdMechanicalState(simulation::Node* node,sofa::core::behavior::BaseMechanicalState* mm) override;

    Result fwdMappedMechanicalState(simulation::Node* node,sofa::core::behavior::BaseMechanicalState* mm) override;

    Result fwdInteractionForceField(simulation::Node* node,sofa::core::behavior::BaseInteractionForceField* ff) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override
    {
        return "MechanicalVReallocVisitor";
    }

    virtual std::string getInfos() const override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addWriteVector(*v);
    }
#endif
protected:


    MyVecId getId(sofa::core::behavior::BaseMechanicalState* mm );
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
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdInteractionForceField(simulation::Node* node,sofa::core::behavior::BaseInteractionForceField* ff) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalVFreeVisitor"; }
    virtual std::string getInfos() const override;
    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override
    {
        if (mapped || only_mapped)
            return false;
        else
            return !map->areForcesMapped();
    }

    MechanicalVOpVisitor& setMapped(bool m = true) { mapped = m; return *this; }
    MechanicalVOpVisitor& setOnlyMapped(bool m = true) { only_mapped = m; return *this; }

    Result fwdMechanicalState(VisitorContext* ctx,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(VisitorContext* ctx,sofa::core::behavior::BaseMechanicalState* mm) override;

    const char* getClassName() const override { return "MechanicalVOpVisitor";}
    virtual std::string getInfos() const override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
    bool readNodeData() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    typedef sofa::core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
    bool mapped;
    MechanicalVMultiOpVisitor(const sofa::core::ExecParams* params, const VMultiOp& o)
        : BaseMechanicalVisitor(params), mapped(false), ops(o)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    MechanicalVMultiOpVisitor& setMapped(bool m = true) { mapped = m; return *this; }

    Result fwdMechanicalState(VisitorContext* ctx,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(VisitorContext* ctx,sofa::core::behavior::BaseMechanicalState* mm) override;

    const char* getClassName() const override { return "MechanicalVMultiOpVisitor"; }
    virtual std::string getInfos() const override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
    bool readNodeData() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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

    Result fwdMechanicalState(VisitorContext* ctx,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalVDotVisitor";}
    virtual std::string getInfos() const override
    {
        std::string name("v= a*b with a[");
        name += a.getName() + "] and b[" + b.getName() + "]";
        return name;
    }
    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
    bool writeNodeData() const override
    {
        return true;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    bool writeNodeData() const override
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
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    void bwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override
    {
        if (ignoreFlag)
            return false;
        else
            return !map->areForcesMapped();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalPropagateDxVisitor"; }
    virtual std::string getInfos() const override
    {
        std::string name="["+dx.getName()+"]";
        return name;
    }
    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    void bwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalPropagateDxAndResetForceVisitor";}
    virtual std::string getInfos() const override { std::string name= "dx["+dx.getName()+"] f["+f.getName()+"]"; return name;}

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadWriteVector(dx);
        addWriteVector(f);
    }
#endif
};

/** Same as MechanicalPropagateOnlyPositionVisitor followed by MechanicalResetForceVisitor

Note that this visitor only propagate through the mappings, and does
not apply projective constraints as was previously done by
MechanicalPropagatePositionAndResetForceVisitor.
Use MechanicalProjectPositionVisitor before this visitor if projection
is needed.
*/
class SOFA_SIMULATION_CORE_API MechanicalPropagateOnlyPositionAndResetForceVisitor : public MechanicalVisitor
{
public:
    sofa::core::MultiVecCoordId x;
    sofa::core::MultiVecDerivId f;
    bool ignoreMask;

    MechanicalPropagateOnlyPositionAndResetForceVisitor(const sofa::core::MechanicalParams* mparams,
                                                    sofa::core::MultiVecCoordId x, sofa::core::MultiVecDerivId f, bool m)
        : MechanicalVisitor(mparams) , x(x), f(f), ignoreMask(m)
    {
    }
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    void bwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalPropagateOnlyPositionAndResetForceVisitor"; }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMass(simulation::Node* /*node*/,sofa::core::behavior::BaseMass* mass) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalAddMDxVisitor"; }
    virtual std::string getInfos() const override { std::string name="dx["+dx.getName()+"] in res[" + res.getName()+"]"; return name; }

    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* /*mm*/) override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadVector(res);
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
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMass(simulation::Node* /*node*/,sofa::core::behavior::BaseMass* mass) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalAccFromFVisitor"; }
    virtual std::string getInfos() const override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
                                           sofa::core::MultiMatrixDerivId c = sofa::core::MatrixDerivId::constraintJacobian(), SReal time = 0.0)
        : MechanicalVisitor(mparams), cId(c), t(time)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    Result fwdProjectiveConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseProjectiveConstraintSet* c) override;


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalProjectJacobianMatrixVisitor"; }
    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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

    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    Result fwdProjectiveConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseProjectiveConstraintSet* c) override;


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalProjectVelocityVisitor"; }
    virtual std::string getInfos() const override
    {
        std::string name="["+vel.getName()+"]"; return name;
    }
    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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

    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    Result fwdProjectiveConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseProjectiveConstraintSet* c) override;


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalProjectPositionVisitor"; }
    virtual std::string getInfos() const override
    {
        std::string name="["+pos.getName()+"]"; return name;
    }
    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadWriteVector(pos);
    }
#endif
};

class SOFA_SIMULATION_CORE_API MechanicalProjectPositionAndVelocityVisitor : public MechanicalVisitor
{
public:
    double t;
    sofa::core::MultiVecCoordId pos;
    sofa::core::MultiVecDerivId vel;
    MechanicalProjectPositionAndVelocityVisitor(const sofa::core::MechanicalParams* mparams , double time=0,
                                                sofa::core::MultiVecCoordId x = sofa::core::VecCoordId::position(),
                                                sofa::core::MultiVecDerivId v = sofa::core::VecDerivId::velocity())
        : MechanicalVisitor(mparams) , t(time), pos(x), vel(v)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    Result fwdProjectiveConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseProjectiveConstraintSet* c) override;


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalProjectPositionAndVelocityVisitor"; }
    virtual std::string getInfos() const override
    {
        std::string name="x["+pos.getName()+"] v["+vel.getName()+"]";
        return name;
    }
    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadWriteVector(pos);
        addReadWriteVector(vel);
    }
#endif
};

/** Propagate positions  to all the levels of the hierarchy.
At each level, the mappings form the parent to the child is applied.

Note that this visitor only propagate through the mappings, and does
not apply projective constraints as was previously done by
MechanicalPropagatePositionVisitor.
Use MechanicalProjectPositionVisitor before this visitor if projection
is needed.
*/
class SOFA_SIMULATION_CORE_API MechanicalPropagateOnlyPositionVisitor : public MechanicalVisitor
{
public:
    SReal t;
    sofa::core::MultiVecCoordId x;
    bool ignoreMask;

    MechanicalPropagateOnlyPositionVisitor( const sofa::core::MechanicalParams* mparams, SReal time=0,
                                        sofa::core::MultiVecCoordId x = sofa::core::VecCoordId::position(), bool m=true);

    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    void bwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalPropagateOnlyPositionVisitor";}
    virtual std::string getInfos() const override
    {
        std::string name="x["+x.getName()+"]";
        if (ignoreMask) name += " Mask DISABLED";
        else            name += " Mask ENABLED";
        return name;
    }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadWriteVector(x);
    }
#endif
};
/** Propagate positions and velocities to all the levels of the hierarchy.
At each level, the mappings form the parent to the child is applied.
After the execution of this action, all the (mapped) degrees of freedom are consistent with the independent degrees of freedom.
This action is typically applied after time integration of the independent degrees of freedom.

Note that this visitor only propagate through the mappings, and does
not apply projective constraints as was previously done by
MechanicalPropagatePositionAndVelocityVisitor.
Use MechanicalProjectPositionAndVelocityVisitor before this visitor if
projection is needed.
*/
class SOFA_SIMULATION_CORE_API MechanicalPropagateOnlyPositionAndVelocityVisitor : public MechanicalVisitor
{
public:
    SReal currentTime;
    sofa::core::MultiVecCoordId x;
    sofa::core::MultiVecDerivId v;
    bool ignoreMask;

    MechanicalPropagateOnlyPositionAndVelocityVisitor(const sofa::core::MechanicalParams* mparams, SReal time=0,
                                                  sofa::core::MultiVecCoordId x = sofa::core::VecId::position(), sofa::core::MultiVecDerivId v = sofa::core::VecId::velocity(),
            bool m=true );  

    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    void bwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override
    {
        return false;
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalPropagateOnlyPositionAndVelocityVisitor";}
    virtual std::string getInfos() const override { std::string name="x["+x.getName()+"] v["+v.getName()+"]"; return name; }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadWriteVector(x);
        addReadWriteVector(v);
    }
#endif
};

/** Propagate velocities to all the levels of the hierarchy.
At each level, the mappings form the parent to the child is applied.
After the execution of this action, all the (mapped) degrees of freedom are consistent with the independent degrees of freedom.

Note that this visitor only propagate through the mappings, and does
not apply projective constraints as was previously done by
MechanicalPropagateVelocityVisitor.
Use MechanicalProjectVelocityVisitor before this visitor if projection
is needed.
*/
class SOFA_SIMULATION_CORE_API MechanicalPropagateOnlyVelocityVisitor : public MechanicalVisitor
{
public:
    SReal currentTime;
    sofa::core::MultiVecDerivId v;
    bool ignoreMask;    
    
    MechanicalPropagateOnlyVelocityVisitor(const sofa::core::MechanicalParams* mparams, SReal time=0,
                                       sofa::core::MultiVecDerivId v = sofa::core::VecId::velocity(),
            bool m=true);

    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    void bwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalPropagateOnlyVelocityVisitor";}
    virtual std::string getInfos() const override { std::string name="v["+v.getName()+"]"; return name; }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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

    MechanicalSetPositionAndVelocityVisitor(const sofa::core::MechanicalParams* mparams ,SReal time=0,
                                            sofa::core::MultiVecCoordId x = sofa::core::VecCoordId::position(),
                                            sofa::core::MultiVecDerivId v = sofa::core::VecDerivId::velocity());

    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalSetPositionAndVelocityVisitor";}
    virtual std::string getInfos() const override { std::string name="x["+x.getName()+"] v["+v.getName()+"]"; return name; }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override {  return "MechanicalResetForceVisitor";}
    virtual std::string getInfos() const override
    {
        std::string name="["+res.getName()+"]";
        if (onlyMapped) name+= " Only Mapped";
        return name;
    }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdForceField(simulation::Node* /*node*/,sofa::core::behavior::BaseForceField* ff) override;
    void bwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    void bwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override {return "MechanicalComputeForceVisitor";}
    virtual std::string getInfos() const override
    {
        std::string name=std::string("[")+res.getName()+std::string("]");
        if (accumulate) name+= " Accumulating";
        else            name+= " Not Accumulating";
        return name;
    }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdForceField(simulation::Node* /*node*/,sofa::core::behavior::BaseForceField* ff) override;
    void bwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    void bwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override {return "MechanicalComputeDfVisitor";}
    virtual std::string getInfos() const override
    {
        std::string name="["+res.getName()+"]";
        if (accumulate) name+= " Accumulating";
        else            name+= " Not Accumulating";
        return name;
    }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override {return "MechanicalComputeGeometricStiffness";}
    virtual std::string getInfos() const override
    {
        std::string name="["+childForce.getName()+"]";
        return name;
    }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdForceField(simulation::Node* /*node*/,sofa::core::behavior::BaseForceField* ff) override;
    void bwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    void bwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalAddMBKdxVisitor"; }
    virtual std::string getInfos() const override { std::string name= "["+res.getName()+"]"; return name; }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadWriteVector(res);
    }
#endif
};



class SOFA_SIMULATION_CORE_API MechanicalResetConstraintVisitor : public BaseMechanicalVisitor
{
public:
    //VecId res;
    MechanicalResetConstraintVisitor(const sofa::core::ConstraintParams* cparams)
        : BaseMechanicalVisitor(cparams)
        , m_cparams(cparams)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseConstraintSet* mm) override;

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalResetConstraintVisitor"; }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
    }
#endif

private:
    const sofa::core::ConstraintParams* m_cparams;
};


/// Call each BaseConstraintSet to build the Jacobian matrices and accumulate it through the mappings up to the independant DOFs
/// @deprecated use MechanicalBuildConstraintMatrix followed by MechanicalAccumulateMatrixDeriv
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

    const sofa::core::ConstraintParams* constraintParams() const { return cparams; }

    Result fwdConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseConstraintSet* c) override;

    void bwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;

    /// This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalAccumulateConstraint"; }

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
    sofa::core::MultiMatrixDerivId res;
    unsigned int &contactId;
    const sofa::core::ConstraintParams *cparams;
};

/// Call each BaseConstraintSet to build the Jacobian matrices
class SOFA_SIMULATION_CORE_API MechanicalBuildConstraintMatrix : public BaseMechanicalVisitor
{
public:
    MechanicalBuildConstraintMatrix(const sofa::core::ConstraintParams* _cparams,
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

    const sofa::core::ConstraintParams* constraintParams() const { return cparams; }

    Result fwdConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseConstraintSet* c) override;

    /// This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalBuildConstraintMatrix"; }

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
    sofa::core::MultiMatrixDerivId res;
    unsigned int &contactId;
    const sofa::core::ConstraintParams *cparams;
};

/// Accumulate Jacobian matrices through the mappings up to the independant DOFs
class SOFA_SIMULATION_CORE_API MechanicalAccumulateMatrixDeriv : public BaseMechanicalVisitor
{
public:
    MechanicalAccumulateMatrixDeriv(const sofa::core::ConstraintParams* _cparams,
                                   sofa::core::MultiMatrixDerivId _res, bool _reverseOrder = false)
        : BaseMechanicalVisitor(_cparams)
        , res(_res)
        , cparams(_cparams)
        , reverseOrder(_reverseOrder)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    const sofa::core::ConstraintParams* constraintParams() const { return cparams; }

    void bwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;

    /// Return true to reverse the order of traversal of child nodes
    bool childOrderReversed(simulation::Node* /*node*/) override { return reverseOrder; }

    /// This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalAccumulateMatrixDeriv"; }

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
    sofa::core::MultiMatrixDerivId res;
    const sofa::core::ConstraintParams *cparams;
    bool reverseOrder;
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
                                      sofa::core::MultiVecDerivId res, double **W = nullptr)
        : MechanicalVisitor(mparams) , res(res), W(W)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* /*mm*/) override;
    void bwdProjectiveConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseProjectiveConstraintSet* c) override;
    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override
    {
        return false;
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalApplyConstraintsVisitor"; }
    virtual std::string getInfos() const override { std::string name= "["+res.getName()+"]"; return name; }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalBeginIntegrationVisitor"; }

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override
    {
        return false;
    }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalEndIntegrationVisitor"; }

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override
    {
        return false;
    }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    Result fwdOdeSolver(simulation::Node* node,sofa::core::behavior::OdeSolver* obj) override;
    Result fwdInteractionForceField(simulation::Node*,sofa::core::behavior::BaseInteractionForceField* obj) override;
    void bwdOdeSolver(simulation::Node* /*node*/,sofa::core::behavior::OdeSolver* /*obj*/) override
    {
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalIntegrationVisitor"; }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    void bwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { static std::string name= "MechanicalComputeContactForceVisitor["+res.getName()+"]"; return name.c_str(); }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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
    Result fwdMass(simulation::Node* /*node*/,sofa::core::behavior::BaseMass* mass) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalAddSeparateGravityVisitor"; }
    virtual std::string getInfos() const override { std::string name= "["+res.getName()+"]"; return name; }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
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

    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalPickParticles"; }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
    }
#endif

    /// get the closest pickable particle
    void getClosestParticle(sofa::core::behavior::BaseMechanicalState*& mstate, sofa::Index& indexCollisionElement, defaulttype::Vector3& point, SReal& rayLength );


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

	Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;
	Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
	Result fwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

	/// Return a class name for this visitor
	/// Only used for debugging / profiling purposes
	const char* getClassName() const override { return "MechanicalPickParticlesWithTags"; }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
	{
	}
#endif

	/// get the closest pickable particle
	void getClosestParticle(sofa::core::behavior::BaseMechanicalState*& mstate, unsigned int& indexCollisionElement, defaulttype::Vector3& point, SReal& rayLength );

private:

    // this function checks if the component must be included in the pick process according to its tags
    bool isComponentTagIncluded(const sofa::core::behavior::BaseMechanicalState* mm);

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

    Result fwdMechanicalState(simulation::Node*,sofa::core::behavior::BaseMechanicalState* mm) override;
    Result fwdMappedMechanicalState(simulation::Node*,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalVSizeVisitor";}
    virtual std::string getInfos() const override
    {
        std::string name = "[" + v.getName() + "]";
        return name;
    }
    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return false;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadVector(v);
    }
#endif
};

#if !defined(SOFA_SIMULATION_MECHANICALVISITOR_CPP)
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
