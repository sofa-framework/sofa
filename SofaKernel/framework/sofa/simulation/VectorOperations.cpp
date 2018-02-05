/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <sofa/simulation/VectorOperations.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/simulation/MechanicalVisitor.h>


#include <sofa/simulation/VelocityThresholdVisitor.h>
#include <sofa/simulation/MechanicalVPrintVisitor.h>

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
    //executeVisitor( MechanicalVAvailVisitor<core:V_COORD>( params, id) );
    //v.assign(id);
    MechanicalVAvailVisitor<core::V_COORD> avail(params, id);
    executeVisitor( &avail );
    //v.assign(id);
    v.setId(avail.states, id);
    executeVisitor( MechanicalVAllocVisitor<core::V_COORD>(params, v) );
}

void VectorOperations::v_alloc(sofa::core::MultiVecDerivId& v)
{
    core::VecDerivId id(core::VecDerivId::V_FIRST_DYNAMIC_INDEX);
    MechanicalVAvailVisitor<core::V_DERIV> avail(params, id);
    executeVisitor( &avail );
    //v.assign(id);
    v.setId(avail.states, id);
    executeVisitor(  MechanicalVAllocVisitor<core::V_DERIV>(params, v) );
}

void VectorOperations::v_free(sofa::core::MultiVecCoordId& id, bool interactionForceField, bool propagate)
{
    if( !id.isNull() ) executeVisitor( MechanicalVFreeVisitor<core::V_COORD>( params, id, interactionForceField, propagate) );
}

void VectorOperations::v_free(sofa::core::MultiVecDerivId& id, bool interactionForceField, bool propagate)
{
    if( !id.isNull() ) executeVisitor( MechanicalVFreeVisitor<core::V_DERIV>(params, id, interactionForceField, propagate) );
}

void VectorOperations::v_realloc(sofa::core::MultiVecCoordId& v, bool interactionForceField, bool propagate)
{
    if( v.isNull() )
    {
        core::VecCoordId id(core::VecCoordId::V_FIRST_DYNAMIC_INDEX);
        MechanicalVAvailVisitor<core::V_COORD> avail(params, id);
        executeVisitor( &avail );
        //v.assign(id);
        v.setId(avail.states, id);
    }
    executeVisitor( MechanicalVReallocVisitor<core::V_COORD>(params, &v, interactionForceField, propagate) );
}

void VectorOperations::v_realloc(sofa::core::MultiVecDerivId& v, bool interactionForceField, bool propagate)
{
    if( v.isNull() )
    {
        core::VecDerivId id(core::VecDerivId::V_FIRST_DYNAMIC_INDEX);
        MechanicalVAvailVisitor<core::V_DERIV> avail(params, id);
        executeVisitor( &avail );
        //v.assign(id);
        v.setId(avail.states, id);
    }
    executeVisitor( MechanicalVReallocVisitor<core::V_DERIV>(params, &v, interactionForceField, propagate) );
}


void VectorOperations::v_clear(sofa::core::MultiVecId v) //v=0
{
    executeVisitor( MechanicalVOpVisitor(params, v, core::ConstMultiVecId::null(), core::ConstMultiVecId::null(), 1.0) );
}

void VectorOperations::v_eq(sofa::core::MultiVecId v, sofa::core::ConstMultiVecId a) // v=a
{
    executeVisitor( MechanicalVOpVisitor(params, v, a, core::ConstMultiVecId::null(), 1.0) );
}

void VectorOperations::v_eq(sofa::core::MultiVecId v, sofa::core::ConstMultiVecId a, SReal f) // v=f*a
{
    executeVisitor( MechanicalVOpVisitor(params, v, core::ConstMultiVecId::null(), a, f) );
}

void VectorOperations::v_peq(sofa::core::MultiVecId v, sofa::core::ConstMultiVecId a, SReal f)
{
    executeVisitor( MechanicalVOpVisitor(params, v, v, a, f) );
}


void VectorOperations::v_teq(sofa::core::MultiVecId v, SReal f)
{
    executeVisitor( MechanicalVOpVisitor(params, v, core::MultiVecId::null(), v, f) );
}

void VectorOperations::v_op(core::MultiVecId v, sofa::core::ConstMultiVecId a, sofa::core::ConstMultiVecId b, SReal f )
{
    executeVisitor( MechanicalVOpVisitor(params, v, a, b, f) );
}

void VectorOperations::v_multiop(const core::behavior::BaseMechanicalState::VMultiOp& o)
{
    executeVisitor( MechanicalVMultiOpVisitor(params, o) );
}


void VectorOperations::v_dot( sofa::core::ConstMultiVecId a, sofa::core::ConstMultiVecId b)
{
    result = 0;
    MechanicalVDotVisitor(params, a,b,&result).setTags(ctx->getTags()).execute( ctx, executeVisitor.precomputedTraversalOrder );
}

void VectorOperations::v_norm( sofa::core::ConstMultiVecId a, unsigned l)
{
    MechanicalVNormVisitor vis(params, a,l);
    vis.setTags(ctx->getTags()).execute( ctx, executeVisitor.precomputedTraversalOrder );
    result = vis.getResult();
}


void VectorOperations::v_threshold(sofa::core::MultiVecId a, SReal threshold)
{
    executeVisitor( VelocityThresholdVisitor(params, a,threshold) );
}

void VectorOperations::print(sofa::core::ConstMultiVecId v, std::ostream &out, std::string prefix, std::string suffix)
{
    out << prefix;
    executeVisitor( MechanicalVPrintVisitor( params, v, out ) );
    out << suffix;
}

size_t VectorOperations::v_size(core::MultiVecId v)
{
    size_t result = 0;
    executeVisitor( MechanicalVSizeVisitor(params,&result,v) );
    return result;
}

SReal VectorOperations::finish()
{
    return result;

}

}
}
}
