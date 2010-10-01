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
    executeVisitor( MechanicalVOpVisitor(v, params) );
}

void VectorOperations::v_eq(sofa::core::MultiVecId v, sofa::core::MultiVecId a) // v=a
{
    executeVisitor( MechanicalVOpVisitor(v,a, params) );
}
#ifndef SOFA_SMP
void VectorOperations::v_peq(sofa::core::MultiVecId v, sofa::core::MultiVecId a, double f)
{
    executeVisitor( MechanicalVOpVisitor(v,v,a,f,params), true ); // enable prefetching
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
    executeVisitor( MechanicalVOpVisitor(v, null,v,f, params) );
}

void VectorOperations::v_op(core::MultiVecId v, sofa::core::MultiVecId a, sofa::core::MultiVecId b, double f )
{
    executeVisitor( MechanicalVOpVisitor(v,a,b,f, params), true ); // enable prefetching
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
    MechanicalVDotVisitor(a,b,&result, params).setTags(ctx->getTags()).execute( ctx, true ); // enable prefetching

}

void VectorOperations::v_threshold(sofa::core::MultiVecId a, double threshold)
{
    executeVisitor( VelocityThresholdVisitor(a,threshold,params) );
}

void VectorOperations::print(sofa::core::MultiVecId v, std::ostream &out)
{
    executeVisitor( MechanicalVPrintVisitor( v,params,out ) );
}


}
}
}
