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
    //executeVisitor( MechanicalVAvailVisitor<V_COORD>( params /* PARAMS FIRST */, id) );
    //v.assign(id);
    MechanicalVAvailVisitor<V_COORD> avail(params /* PARAMS FIRST */, id);
    executeVisitor( &avail );
    //v.assign(id);
    v.setId(avail.states, id);
    executeVisitor( MechanicalVAllocVisitor<V_COORD>(params /* PARAMS FIRST */, v) );
}

void VectorOperations::v_alloc(sofa::core::MultiVecDerivId& v)
{
    VecDerivId id(VecDerivId::V_FIRST_DYNAMIC_INDEX);
    MechanicalVAvailVisitor<V_DERIV> avail(params /* PARAMS FIRST */, id);
    executeVisitor( &avail );
    //v.assign(id);
    v.setId(avail.states, id);
    executeVisitor(  MechanicalVAllocVisitor<V_DERIV>(params /* PARAMS FIRST */, v) );
}

void VectorOperations::v_free(sofa::core::MultiVecCoordId& id)
{
    executeVisitor( MechanicalVFreeVisitor<V_COORD>( params /* PARAMS FIRST */, id) );
}

void VectorOperations::v_free(sofa::core::MultiVecDerivId& id)
{
    executeVisitor( MechanicalVFreeVisitor<V_DERIV>(params /* PARAMS FIRST */, id) );
}

void VectorOperations::v_clear(sofa::core::MultiVecId v) //v=0
{
    executeVisitor( MechanicalVOpVisitor(params /* PARAMS FIRST */, v, ConstMultiVecId::null(), ConstMultiVecId::null(), 1.0) );
}

void VectorOperations::v_eq(sofa::core::MultiVecId v, sofa::core::MultiVecId a) // v=a
{
    executeVisitor( MechanicalVOpVisitor(params /* PARAMS FIRST */, v, a, ConstMultiVecId::null(), 1.0) );
}
#ifndef SOFA_SMP
void VectorOperations::v_peq(sofa::core::MultiVecId v, sofa::core::MultiVecId a, double f)
{
    executeVisitor( MechanicalVOpVisitor(params /* PARAMS FIRST */, v, v, a, f) );
}
#else
void VectorOperations::v_peq(sofa::core::MultiVecId v, sofa::core::MultiVecId a, Shared<double> &fSh,double f)
{
    ParallelMechanicalVOpVisitor(params /* PARAMS FIRST */, v, v, a, f, &fSh).execute( ctx );
}

void VectorOperations::v_peq(sofa::core::MultiVecId v, sofa::core::MultiVecId a, double f)
{
    // ParallelMechanicalVOpVisitor(params /* PARAMS FIRST */, v, v, a, f).execute( ctx );
}

void VectorOperations::v_meq(sofa::core::MultiVecId v, sofa::core::MultiVecId a, Shared<double> &fSh)
{
    ParallelMechanicalVOpMecVisitor(params /* PARAMS FIRST */, v, a, &fSh).execute( ctx );
}
#endif

void VectorOperations::v_teq(sofa::core::MultiVecId v, double f)
{
    executeVisitor( MechanicalVOpVisitor(params /* PARAMS FIRST */, v, core::MultiVecId::null(), v, f) );
}

void VectorOperations::v_op(core::MultiVecId v, sofa::core::ConstMultiVecId a, sofa::core::ConstMultiVecId b, double f )
{
    executeVisitor( MechanicalVOpVisitor(params /* PARAMS FIRST */, v, a, b, f) );
}

void VectorOperations::v_multiop(const core::behavior::BaseMechanicalState::VMultiOp& o)
{
    executeVisitor( MechanicalVMultiOpVisitor(params /* PARAMS FIRST */, o) );
}

#ifdef SOFA_SMP
void VectorOperations::v_op(sofa::core::MultiVecId v, sofa::core::MultiVecId a, sofa::core::MultiVecId b, Shared<double> &f) ///< v=a+b*f
{
    ParallelMechanicalVOpVisitor(params /* PARAMS FIRST */, v, a, b, 1.0, &f).execute( ctx );
}
#endif // SOFA_SMP

void VectorOperations::v_dot( sofa::core::ConstMultiVecId a, sofa::core::ConstMultiVecId b)
{
    result = 0;
    MechanicalVDotVisitor(params /* PARAMS FIRST */, a,b,&result).setTags(ctx->getTags()).execute( ctx );
}

#ifdef SOFA_SMP
void VectorOperations::v_dot( Shared<double> &result, core::MultiVecId a, core::MultiVecId b)
{
    ParallelMechanicalVDotVisitor(params /* PARAMS FIRST */, &result, a, b).execute( ctx );
}
#endif

void VectorOperations::v_threshold(sofa::core::MultiVecId a, double threshold)
{
    executeVisitor( VelocityThresholdVisitor(params /* PARAMS FIRST */, a,threshold) );
}

void VectorOperations::print(sofa::core::MultiVecId v, std::ostream &out)
{
    executeVisitor( MechanicalVPrintVisitor( params /* PARAMS FIRST */, v, out ) );
}

double VectorOperations::finish()
{
    return result;

}

}
}
}
