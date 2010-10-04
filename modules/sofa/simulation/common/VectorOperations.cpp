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
    executeVisitor( MechanicalVAvailVisitor<V_COORD>( id, params) );
    v.assign(id);
    executeVisitor( MechanicalVAllocVisitor<V_COORD>(v, params) );
}

void VectorOperations::v_alloc(sofa::core::MultiVecDerivId& v)
{
    VecDerivId id(VecDerivId::V_FIRST_DYNAMIC_INDEX);
    executeVisitor( MechanicalVAvailVisitor<V_DERIV>(id, params) );
    v.assign(id);
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
    executeVisitor( MechanicalVOpVisitor(params, v) );
}

void VectorOperations::v_eq(sofa::core::MultiVecId v, sofa::core::MultiVecId a) // v=a
{
    executeVisitor( MechanicalVOpVisitor(params, v,a) );
}
#ifndef SOFA_SMP
void VectorOperations::v_peq(sofa::core::MultiVecId v, sofa::core::MultiVecId a, double f)
{
    executeVisitor( MechanicalVOpVisitor(params, v,v,a,f), true ); // enable prefetching
}
#else
void VectorOperations::v_peq(sofa::core::MultiVecId v, sofa::core::MultiVecId a, Shared<double> &fSh,double f)
{
    ParallelMechanicalVOpVisitor(params, v,v,a,f,&fSh).execute( ctx );
}

void VectorOperations::v_peq(sofa::core::MultiVecId v, sofa::core::MultiVecId a, double f)
{
    ParallelMechanicalVOpVisitor(params, v,v,a,f).execute( ctx );
}

void VectorOperations::v_meq(sofa::core::MultiVecId v, sofa::core::MultiVecId a, Shared<double> &fSh)
{
    ParallelMechanicalVOpMecVisitor(params, v,a,&fSh).execute( ctx );
}
#endif

void VectorOperations::v_teq(sofa::core::MultiVecId v, double f)
{
    executeVisitor( MechanicalVOpVisitor(params, v, core::MultiVecId::null(),v,f) );
}

void VectorOperations::v_op(core::MultiVecId v, sofa::core::ConstMultiVecId a, sofa::core::ConstMultiVecId b, double f )
{
    executeVisitor( MechanicalVOpVisitor(params, v,a,b,f), true ); // enable prefetching
}

void VectorOperations::v_multiop(const core::behavior::BaseMechanicalState::VMultiOp& o)
{
    executeVisitor( MechanicalVMultiOpVisitor(o, params), true ); // enable prefetching
}

#ifdef SOFA_SMP
void VectorOperations::v_op(sofa::core::MultiVecId v, sofa::core::MultiVecId a, sofa::core::MultiVecId b, Shared<double> &f) ///< v=a+b*f
{
    ParallelMechanicalVOpVisitor(params, v,a,b,1.0,&f).execute( ctx );
}
#endif // SOFA_SMP

void VectorOperations::v_dot( sofa::core::ConstMultiVecId a, sofa::core::ConstMultiVecId b)
{
    result = 0;
    MechanicalVDotVisitor(a,b,&result, params).setTags(ctx->getTags()).execute( ctx, true ); // enable prefetching
}

#ifdef SOFA_SMP
void VectorOperations::v_dot( Shared<double> &result, core::MultiVecId a, core::MultiVecId b)
{
    ParallelMechanicalVDotVisitor(params, &result, a, b).execute( ctx );
}
#endif

void VectorOperations::v_threshold(sofa::core::MultiVecId a, double threshold)
{
    executeVisitor( VelocityThresholdVisitor(a,threshold,params) );
}

void VectorOperations::print(sofa::core::MultiVecId v, std::ostream &out)
{
    executeVisitor( MechanicalVPrintVisitor( v,params,out ) );
}

double VectorOperations::finish()
{
    return result;

}

}
}
}
