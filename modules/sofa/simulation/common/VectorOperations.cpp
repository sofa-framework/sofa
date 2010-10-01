#include <sofa/simulation/common/VectorOperations.h>
#include <sofa/core/MultiVecId.h>

#ifndef SOFA_SMP
#include <sofa/simulation/common/MechanicalVisitor.h>
#else
#include <sofa/simulation/common/ParallelMechanicalVisitor.h>
#endif


namespace sofa
{
namespace simulation
{
namespace common
{

VectorOperations::VectorOperations(const sofa::core::ExecParams* params, sofa::core::objectmodel::Context *ctx):
    sofa::core::behavior::BaseVectorOperations(params,ctx)
{
}

void VectorOperations::prepareVisitor(sofa::simulation::Visitor* v )
{
    v->setTags( ctx->getTags() );
}

// do not know what this is supposed to do
void VectorOperations::prepareVisitor(sofa::simulation::MechanicalVisitor* v)
{
    /*     if (v->writeNodeData())
            v->setNodeMap(this->getWriteNodeMap());
        else
            v->setNodeMap(this->getNodeMap());
            */
    prepareVisitor((Visitor*)v);
}
void VectorOperations::v_alloc(sofa::core::MultiVecCoordId& v)
{
    /* template < VecType vtype > MechanicalVAvailVisitor;
    /* this can be probably merge in a single operation with the MultiVecId design */
    VecCoordId id(VecCoordId::V_FIRST_DYNAMIC_INDEX);
    executeVisitor( MechanicalVAvailVisitor<V_COORD>(params, id) );
    v.assign(id);
    executeVisitor( MechanicalVAllocVisitor<V_COORD>(params, v) );
}

void VectorOperations::v_alloc(sofa::core::MultiVecDerivId& v)
{
    VecDerivId id(VecDerivId::V_FIRST_DYNAMIC_INDEX);
    executeVisitor( MechanicalVAvailVisitor<V_DERIV>(params, id) );
    v.assign(id);
    executeVisitor(  MechanicalVAllocVisitor<V_DERIV>(params, v) );
}

void VectorOperations::v_free(sofa::core::MultiVecCoordId& id)
{
    executeVisitor( MechanicalVFreeVisitor<V_COORD>(params, id) );
}

void VectorOperations::v_free(sofa::core::MultiVecDerivId& id)
{
    executeVisitor( MechanicalVFreeVisitor<V_DERIV>(params, id) );
}

void VectorOperations::v_clear(sofa::core::MultiVecId v) //v=0
{
    executeVisitor( MechanicalVOpVisitor(params, v) );
}

void VectorOperations::v_eq(sofa::core::MultiVecId v, sofa::core::MultiVecId a) // v=a
{
    executeVisitor( MechanicalVOpVisitor(params,v,a) );
}
#ifndef SOFA_SMP
void VectorOperations::v_peq(sofa::core::MultiVecId v, sofa::core::MultiVecId a, double f)
{
    executeVisitor( MechanicalVOpVisitor(params,v,v,a,f), true ); // enable prefetching
}
#else
void VectorOperations::v_peq(VecId v, VecId a, Shared<double> &fSh,double f)
{
    ParallelMechanicalVOpVisitor(v,v,a,f,&fSh).execute( ctx );
}

void VectorOperations::v_peq(VecId v, VecId a, double f)
{
    ParallelMechanicalVOpVisitor(v,v,a,f).execute( ctx );

}

void VectorOperations::v_meq(VecId v, VecId a, Shared<double> &fSh)
{
    ParallelMechanicalVOpMecVisitor(v,a,&fSh).execute( ctx );
}
#endif

void VectorOperations::v_teq(sofa::core::MultiVecId v, double f)
{
    using namespace sofa::core;
    MultiVecId null( VecId::null() );
    executeVisitor( MechanicalVOpVisitor(params, v, null,v,f) );
}

void VectorOperations::v_op(core::MultiVecId v, sofa::core::MultiVecId a, sofa::core::MultiVecId b, double f )
{
    executeVisitor( MechanicalVOpVisitor(params,v,a,b,f), true ); // enable prefetching
}

#ifdef SOFA_SMP
void VectorOperations::v_op(sofa::core::MultiVecId v, sofa::core::MultiVecId a, sofa::core::MultiVecId b, Shared<double> &f) ///< v=a+b*f
{
    ParallelMechanicalVOpVisitor(v,a,b,1.0,&f).execute( getContext() );
}
#endif // SOFA_SMP

void VectorOperations::v_dot( sofa::core::MultiVecId a, sofa::core::MultiVecId b)
{
    result = 0;
    MechanicalVDotVisitor(params,a,b,&result).setTags(ctx->getTags()).execute( ctx, true ); // enable prefetching

}


}
}
}
