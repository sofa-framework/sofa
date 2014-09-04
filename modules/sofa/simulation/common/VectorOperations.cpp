/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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

VectorOperations::VectorOperations(const sofa::core::ExecParams* params, sofa::core::objectmodel::BaseContext *ctx, bool precomputedTraversalOrder):
    sofa::core::behavior::BaseVectorOperations(params,ctx),
    executeVisitor(*ctx,precomputedTraversalOrder)
{
}

void VectorOperations::v_alloc(sofa::core::MultiVecCoordId& v)
{
    /* template < VecType vtype > MechanicalVAvailVisitor;  */
    /* this can be probably merged in a single operation with the MultiVecId design */
    core::VecCoordId id(core::VecCoordId::V_FIRST_DYNAMIC_INDEX);
    //executeVisitor( MechanicalVAvailVisitor<core:V_COORD>( params /* PARAMS FIRST */, id) );
    //v.assign(id);
    MechanicalVAvailVisitor<core::V_COORD> avail(params /* PARAMS FIRST */, id);
    executeVisitor( &avail );
    //v.assign(id);
    v.setId(avail.states, id);
    executeVisitor( MechanicalVAllocVisitor<core::V_COORD>(params /* PARAMS FIRST */, v) );
}

void VectorOperations::v_alloc(sofa::core::MultiVecDerivId& v)
{
    core::VecDerivId id(core::VecDerivId::V_FIRST_DYNAMIC_INDEX);
    MechanicalVAvailVisitor<core::V_DERIV> avail(params /* PARAMS FIRST */, id);
    executeVisitor( &avail );
    //v.assign(id);
    v.setId(avail.states, id);
    executeVisitor(  MechanicalVAllocVisitor<core::V_DERIV>(params /* PARAMS FIRST */, v) );
}

void VectorOperations::v_free(sofa::core::MultiVecCoordId& id, bool interactionForceField, bool propagate)
{
    if( !id.isNull() ) executeVisitor( MechanicalVFreeVisitor<core::V_COORD>( params /* PARAMS FIRST */, id, interactionForceField, propagate) );
}

void VectorOperations::v_free(sofa::core::MultiVecDerivId& id, bool interactionForceField, bool propagate)
{
    if( !id.isNull() ) executeVisitor( MechanicalVFreeVisitor<core::V_DERIV>(params /* PARAMS FIRST */, id, interactionForceField, propagate) );
}

void VectorOperations::v_realloc(sofa::core::MultiVecCoordId& v, bool interactionForceField, bool propagate)
{
    if( v.isNull() )
    {
        core::VecCoordId id(core::VecCoordId::V_FIRST_DYNAMIC_INDEX);
        MechanicalVAvailVisitor<core::V_COORD> avail(params /* PARAMS FIRST */, id);
        executeVisitor( &avail );
        //v.assign(id);
        v.setId(avail.states, id);
    }
    executeVisitor( MechanicalVReallocVisitor<core::V_COORD>(params /* PARAMS FIRST */, &v, interactionForceField, propagate) );
}

void VectorOperations::v_realloc(sofa::core::MultiVecDerivId& v, bool interactionForceField, bool propagate)
{
    if( v.isNull() )
    {
        core::VecDerivId id(core::VecDerivId::V_FIRST_DYNAMIC_INDEX);
        MechanicalVAvailVisitor<core::V_DERIV> avail(params /* PARAMS FIRST */, id);
        executeVisitor( &avail );
        //v.assign(id);
        v.setId(avail.states, id);
    }
    executeVisitor( MechanicalVReallocVisitor<core::V_DERIV>(params /* PARAMS FIRST */, &v, interactionForceField, propagate) );
}


void VectorOperations::v_clear(sofa::core::MultiVecId v) //v=0
{
    executeVisitor( MechanicalVOpVisitor(params /* PARAMS FIRST */, v, core::ConstMultiVecId::null(), core::ConstMultiVecId::null(), 1.0) );
}

void VectorOperations::v_eq(sofa::core::MultiVecId v, sofa::core::ConstMultiVecId a) // v=a
{
    executeVisitor( MechanicalVOpVisitor(params /* PARAMS FIRST */, v, a, core::ConstMultiVecId::null(), 1.0) );
}

void VectorOperations::v_eq(sofa::core::MultiVecId v, sofa::core::ConstMultiVecId a, double f) // v=f*a
{
    executeVisitor( MechanicalVOpVisitor(params /* PARAMS FIRST */, v, core::ConstMultiVecId::null(), a, f) );
}

#ifndef SOFA_SMP
void VectorOperations::v_peq(sofa::core::MultiVecId v, sofa::core::ConstMultiVecId a, double f)
{
    executeVisitor( MechanicalVOpVisitor(params /* PARAMS FIRST */, v, v, a, f) );
}
#else
void VectorOperations::v_peq(sofa::core::MultiVecId v, sofa::core::ConstMultiVecId a, Shared<double> &fSh,double f)
{
    ParallelMechanicalVOpVisitor(params /* PARAMS FIRST */, v, v, a, f, &fSh).execute( ctx );
}

void VectorOperations::v_peq(sofa::core::MultiVecId v, sofa::core::ConstMultiVecId a, double f)
{
    // ParallelMechanicalVOpVisitor(params /* PARAMS FIRST */, v, v, a, f).execute( ctx );
}

void VectorOperations::v_meq(sofa::core::MultiVecId v, sofa::core::ConstMultiVecId a, Shared<double> &fSh)
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
    MechanicalVDotVisitor(params /* PARAMS FIRST */, a,b,&result).setTags(ctx->getTags()).execute( ctx, executeVisitor.precomputedTraversalOrder );
}

void VectorOperations::v_norm( sofa::core::ConstMultiVecId a, unsigned l)
{
    MechanicalVNormVisitor vis(params, a,l);
    vis.setTags(ctx->getTags()).execute( ctx, executeVisitor.precomputedTraversalOrder );
    result = vis.getResult();
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

void VectorOperations::print(sofa::core::ConstMultiVecId v, std::ostream &out, std::string prefix, std::string suffix)
{
    out << prefix;
    executeVisitor( MechanicalVPrintVisitor( params /* PARAMS FIRST */, v, out ) );
    out << suffix;
}

size_t VectorOperations::v_size(core::MultiVecId v)
{
    size_t result = 0;
    executeVisitor( MechanicalVSizeVisitor(params,&result,v) );
    return result;
}

double VectorOperations::finish()
{
    return result;

}

}
}
}
