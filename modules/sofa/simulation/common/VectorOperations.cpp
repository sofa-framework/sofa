#include <sofa/simulation/common/VectorOperations.h>
#include <sofa/core/MultiVecId.h>

#ifndef SOFA_SMP
#include <sofa/simulation/common/MechanicalVisitor.h>
#else
#include <sofa/simulation/common/ParallelMechanicalVisitor.h>
#endif

#include <sofa/simulation/common/VelocityThresholdVisitor.h>
#include <sofa/simulation/common/MechanicalVPrintVisitor.h>

namespace sofa
{

namespace simulation
{

namespace common
{

VectorOperations::VectorOperations(const sofa::core::ExecParams* params, sofa::core::objectmodel::BaseContext *ctx):
    sofa::core::behavior::BaseVectorOperations(params,ctx),
    executeVisitor(*ctx)
{
}

void VectorOperations::v_alloc(sofa::core::MultiVecCoordId& v)
{
    /* template < VecType vtype > MechanicalVAvailVisitor;  */
    /* this can be probably merged in a single operation with the MultiVecId design */
    VecCoordId id(VecCoordId::V_FIRST_DYNAMIC_INDEX);
    //executeVisitor( MechanicalVAvailVisitor<V_COORD>( id, params) );
    //v.assign(id);
    MechanicalVAvailVisitor<V_COORD> avail(id, params);
    executeVisitor( &avail );
    //v.assign(id);
    v.setId(avail.states, id);
    executeVisitor( MechanicalVAllocVisitor<V_COORD>(v, params) );
}

void VectorOperations::v_alloc(sofa::core::MultiVecDerivId& v)
{
    VecDerivId id(VecDerivId::V_FIRST_DYNAMIC_INDEX);
    MechanicalVAvailVisitor<V_DERIV> avail(id, params);
    executeVisitor( &avail );
    //v.assign(id);
    v.setId(avail.states, id);
    executeVisitor(  MechanicalVAllocVisitor<V_DERIV>(v, params) );
}

void VectorOperations::v_free(sofa::core::MultiVecCoordId& id)
{
    executeVisitor( MechanicalVFreeVisitor<V_COORD>( id, params) );
}

void VectorOperations::v_free(sofa::core::MultiVecDerivId& id)
{
    executeVisitor( MechanicalVFreeVisitor<V_DERIV>(id, params) );
}

void VectorOperations::v_clear(sofa::core::MultiVecId v) //v=0
{
    executeVisitor( MechanicalVOpVisitor(v, ConstMultiVecId::null(), ConstMultiVecId::null(), 1.0, params) );
}

void VectorOperations::v_eq(sofa::core::MultiVecId v, sofa::core::MultiVecId a) // v=a
{
    executeVisitor( MechanicalVOpVisitor(v, a, ConstMultiVecId::null(), 1.0, params) );
}
#ifndef SOFA_SMP
void VectorOperations::v_peq(sofa::core::MultiVecId v, sofa::core::MultiVecId a, double f)
{
    executeVisitor( MechanicalVOpVisitor(v, v, a, f, params) );
}
#else
void VectorOperations::v_peq(sofa::core::MultiVecId v, sofa::core::MultiVecId a, Shared<double> &fSh,double f)
{
    ParallelMechanicalVOpVisitor(v, v, a, f, &fSh, params).execute( ctx );
}

void VectorOperations::v_peq(sofa::core::MultiVecId v, sofa::core::MultiVecId a, double f)
{
    // ParallelMechanicalVOpVisitor(v, v, a, f, params).execute( ctx );
}

void VectorOperations::v_meq(sofa::core::MultiVecId v, sofa::core::MultiVecId a, Shared<double> &fSh)
{
    ParallelMechanicalVOpMecVisitor(v, a, &fSh, params).execute( ctx );
}
#endif

void VectorOperations::v_teq(sofa::core::MultiVecId v, double f)
{
    executeVisitor( MechanicalVOpVisitor(v, core::MultiVecId::null(), v, f, params) );
}

void VectorOperations::v_op(core::MultiVecId v, sofa::core::ConstMultiVecId a, sofa::core::ConstMultiVecId b, double f )
{
    executeVisitor( MechanicalVOpVisitor(v, a, b, f, params) );
}

void VectorOperations::v_multiop(const core::behavior::BaseMechanicalState::VMultiOp& o)
{
    executeVisitor( MechanicalVMultiOpVisitor(o, params) );
}

#ifdef SOFA_SMP
void VectorOperations::v_op(sofa::core::MultiVecId v, sofa::core::MultiVecId a, sofa::core::MultiVecId b, Shared<double> &f) ///< v=a+b*f
{
    ParallelMechanicalVOpVisitor(v, a, b, 1.0, &f, params).execute( ctx );
}
#endif // SOFA_SMP

void VectorOperations::v_dot( sofa::core::ConstMultiVecId a, sofa::core::ConstMultiVecId b)
{
    result = 0;
    MechanicalVDotVisitor(a,b,&result, params).setTags(ctx->getTags()).execute( ctx );
}

#ifdef SOFA_SMP
void VectorOperations::v_dot( Shared<double> &result, core::MultiVecId a, core::MultiVecId b)
{
    ParallelMechanicalVDotVisitor(&result, a, b, params).execute( ctx );
}
#endif

void VectorOperations::v_threshold(sofa::core::MultiVecId a, double threshold)
{
    executeVisitor( VelocityThresholdVisitor(a,threshold,params) );
}

void VectorOperations::print(sofa::core::MultiVecId v, std::ostream &out)
{
    executeVisitor( MechanicalVPrintVisitor( v, out, params ) );
}

double VectorOperations::finish()
{
    return result;

}

}
}
}
