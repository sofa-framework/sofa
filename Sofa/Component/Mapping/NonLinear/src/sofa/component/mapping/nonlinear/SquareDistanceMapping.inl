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
#pragma once

#include <sofa/component/mapping/nonlinear/SquareDistanceMapping.h>
#include <sofa/core/BaseLocalMappingMatrix.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/MechanicalParams.h>
#include <iostream>
#include <sofa/simulation/Node.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/MechanicalState.inl>
#include <sofa/linearalgebra/CompressedRowSparseMatrixConstraintEigenUtils.h>

namespace sofa::component::mapping::nonlinear
{

template <class TIn, class TOut>
SquareDistanceMapping<TIn, TOut>::SquareDistanceMapping()
    : Inherit()
    , d_showObjectScale(initData(&d_showObjectScale, Real(0), "showObjectScale", "Scale for object display"))
    , d_color(initData(&d_color, sofa::type::RGBAColor(1,1,0,1), "showColor", "Color for object display. (default=[1.0,1.0,0.0,1.0])"))
    , l_topology(initLink("topology", "link to the topology container"))
{
}

template <class TIn, class TOut>
SquareDistanceMapping<TIn, TOut>::~SquareDistanceMapping()
{
}


template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::init()
{
    if (l_topology.empty())
    {
        msg_warning() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (l_topology->getNbEdges() < 1)
    {
        msg_error() << "No Topology component containing edges found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    const SeqEdges& links = l_topology->getEdges();

    this->getToModel()->resize( links.size() );

    // only used for warning message
    const bool compliance = ((simulation::Node*)(this->getContext()))->forceField.size() && ((simulation::Node*)(this->getContext()))->forceField[0]->isCompliance.getValue();
    msg_error_when(compliance) << "Null rest Lengths cannot be used for stable compliant constraint, prefer to use a DifferenceMapping if those dofs are used with a compliance";

    baseMatrices.resize( 1 );
    baseMatrices[0] = &jacobian;

    this->Inherit::init();  // applies the mapping, so after the Data init
}

template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::computeCoordPositionDifference( Direction& r, const InCoord& a, const InCoord& b )
{
    r = TIn::getCPos(b)-TIn::getCPos(a);
}

template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteOnlyAccessor< Data<OutVecCoord> >  out = dOut;
    helper::ReadAccessor< Data<InVecCoord> >  in = dIn;
    const SeqEdges& links = l_topology->getEdges();

    jacobian.resizeBlocks(out.size(),in.size());


    Direction gap;

    for(unsigned i=0; i<links.size(); i++ )
    {
        const InCoord& p0 = in[links[i][0]];
        const InCoord& p1 = in[links[i][1]];

        // gap = in[links[i][1]] - in[links[i][0]] (only for position)
        computeCoordPositionDifference( gap, p0, p1 );

        // if d = N - R  ==> d² = N² + R² - 2.N.R
        const Real gapNorm = gap.norm2();
        out[i][0] = gapNorm; // d = N²


        // insert in increasing column order
        gap *= 2; // 2*p[1]-2*p[0]

//        if( restLengths[i] )
//        {
//            out[i][0] -= ( 2*sqrt(gapNorm) + restLengths[i] ) * restLengths[i]; // d = N² + R² - 2.N.R

//            // TODO implement Jacobian when restpos != 0
//            // gap -=  d2NR/dx
//        }


        jacobian.beginRow(i);
        if( links[i][1]<links[i][0] )
        {
            for(unsigned k=0; k<In::spatial_dimensions; k++ )
                jacobian.insertBack( i, links[i][1]*Nin+k, gap[k] );
            for(unsigned k=0; k<In::spatial_dimensions; k++ )
                jacobian.insertBack( i, links[i][0]*Nin+k, -gap[k] );
        }
        else
        {
            for(unsigned k=0; k<In::spatial_dimensions; k++ )
                jacobian.insertBack( i, links[i][0]*Nin+k, -gap[k] );
            for(unsigned k=0; k<In::spatial_dimensions; k++ )
                jacobian.insertBack( i, links[i][1]*Nin+k, gap[k] );
        }
    }

    jacobian.compress();
}


template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    if( jacobian.rowSize() )
    {
        auto dOutWa = sofa::helper::getWriteOnlyAccessor(dOut);
        auto dInRa = sofa::helper::getReadAccessor(dIn);
        jacobian.mult(dOutWa.wref(),dInRa.ref());
    }
}

template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)
{
    if( jacobian.rowSize() )
    {
        auto dOutRa = sofa::helper::getReadAccessor(dOut);
        auto dInWa = sofa::helper::getWriteOnlyAccessor(dIn);
        jacobian.addMultTranspose(dInWa.wref(),dOutRa.ref());
    }
}

template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId )
{
    const unsigned& geometricStiffness = d_geometricStiffness.getValue().getSelectedId();
    if( !geometricStiffness ) return;

    helper::WriteAccessor<Data<InVecDeriv> > parentForce (*parentDfId[this->fromModel.get()].write());
    helper::ReadAccessor<Data<InVecDeriv> > parentDisplacement (*mparams->readDx(this->fromModel.get()));  // parent displacement
    const SReal& kfactor = mparams->kFactor();
    helper::ReadAccessor<Data<OutVecDeriv> > childForce (*mparams->readF(this->toModel.get()));

    if( K.compressedMatrix.nonZeros() )
    {
        K.addMult( parentForce.wref(), parentDisplacement.ref(), (typename In::Real)kfactor );
    }
    else
    {
        const SeqEdges& links = l_topology->getEdges();

        for(unsigned i=0; i<links.size(); i++ )
        {
            // force in compression (>0) can lead to negative eigen values in geometric stiffness
            // this results in a undefinite implicit matrix that causes instabilies
            // if stabilized GS (geometricStiffness==2) -> keep only force in extension
            if( childForce[i][0] < 0 || geometricStiffness==1 )
            {

                SReal tmp = 2*childForce[i][0]*kfactor;

                InDeriv df = (parentDisplacement[links[i][0]]-parentDisplacement[links[i][1]])*tmp;
                // it is symmetric so    -df  = (parentDisplacement[links[i][1]]-parentDisplacement[links[i][0]])*tmp;

                parentForce[links[i][0]] += df;
                parentForce[links[i][1]] -= df;
            }
        }
    }
}

template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::applyJT(const core::ConstraintParams* cparams, Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in)
{
    SOFA_UNUSED(cparams);
    const OutMatrixDeriv& childMat  = sofa::helper::getReadAccessor(in).ref();
    InMatrixDeriv&        parentMat = sofa::helper::getWriteAccessor(out).wref();
    addMultTransposeEigen(parentMat, jacobian.compressedMatrix, childMat);
}


template <class TIn, class TOut>
const sofa::linearalgebra::BaseMatrix* SquareDistanceMapping<TIn, TOut>::getJ()
{
    return &jacobian;
}

template <class TIn, class TOut>
const type::vector<sofa::linearalgebra::BaseMatrix*>* SquareDistanceMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}



template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::updateK(const core::MechanicalParams *mparams, core::ConstMultiVecDerivId childForceId )
{
    SOFA_UNUSED(mparams);
    const unsigned geometricStiffness = d_geometricStiffness.getValue().getSelectedId();
    if( !geometricStiffness ) { K.resize(0,0); return; }

    helper::ReadAccessor<Data<OutVecDeriv> > childForce( *childForceId[this->toModel.get()].read() );
    const SeqEdges& links = l_topology->getEdges();

    unsigned int size = this->fromModel->getSize();
    K.resizeBlocks(size,size);
    for(size_t i=0; i<links.size(); i++)
    {
        // force in compression (>0) can lead to negative eigen values in geometric stiffness
        // this results in a undefinite implicit matrix that causes instabilies
        // if stabilized GS (geometricStiffness==2) -> keep only force in extension
        if( childForce[i][0] < 0 || geometricStiffness==1 )
        {
            SReal tmp = 2*childForce[i][0];

            for(unsigned k=0; k<In::spatial_dimensions; k++)
            {
                K.add( links[i][0]*Nin+k, links[i][0]*Nin+k, tmp );
                K.add( links[i][0]*Nin+k, links[i][1]*Nin+k, -tmp );
                K.add( links[i][1]*Nin+k, links[i][1]*Nin+k, tmp );
                K.add( links[i][1]*Nin+k, links[i][0]*Nin+k, -tmp );
            }
        }
    }
    K.compress();
}

template <class TIn, class TOut>
const linearalgebra::BaseMatrix* SquareDistanceMapping<TIn, TOut>::getK()
{
    return &K;
}

template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::buildGeometricStiffnessMatrix(
    sofa::core::GeometricStiffnessMatrix* matrices)
{
    const unsigned geometricStiffness = d_geometricStiffness.getValue().getSelectedId();
    if( !geometricStiffness )
    {
        return;
    }

    const auto childForce = this->toModel->readTotalForces();
    const SeqEdges& links = l_topology->getEdges();
    const auto dJdx = matrices->getMappingDerivativeIn(this->fromModel).withRespectToPositionsIn(this->fromModel);

    for(sofa::Size i=0; i<links.size(); i++)
    {
        const OutDeriv force_i = childForce[i];

        const sofa::topology::Edge link = links[i];
        // force in compression (>0) can lead to negative eigen values in geometric stiffness
        // this results in a undefinite implicit matrix that causes instabilies
        // if stabilized GS (geometricStiffness==2) -> keep only force in extension
        if( force_i[0] < 0 || geometricStiffness==1 )
        {
            const Real tmp = 2 * force_i[0];

            for(unsigned k=0; k<In::spatial_dimensions; k++)
            {
                dJdx(link[0] * Nin + k, link[0] * Nin + k) += tmp;
                dJdx(link[0] * Nin + k, link[1] * Nin + k) += -tmp;
                dJdx(link[1] * Nin + k, link[0] * Nin + k) += -tmp;
                dJdx(link[1] * Nin + k, link[1] * Nin + k) += tmp;
            }
        }
    }
}

template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if( !vparams->displayFlags().getShowMechanicalMappings() ) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();
    const SeqEdges& links = l_topology->getEdges();

    if( d_showObjectScale.getValue() == 0 )
    {
        vparams->drawTool()->disableLighting();
        type::vector< type::Vec3 > points;
        for(std::size_t i=0; i<links.size(); i++ )
        {
            points.push_back( sofa::type::Vec3( TIn::getCPos(pos[links[i][0]]) ) );
            points.push_back( sofa::type::Vec3( TIn::getCPos(pos[links[i][1]]) ));
        }
        vparams->drawTool()->drawLines ( points, 1, d_color.getValue() );
    }
    else
    {
        vparams->drawTool()->enableLighting();
        for(std::size_t i=0; i<links.size(); i++ )
        {
            type::Vec3 p0 = TIn::getCPos(pos[links[i][0]]);
            type::Vec3 p1 = TIn::getCPos(pos[links[i][1]]);
            vparams->drawTool()->drawCylinder( p0, p1, (float)d_showObjectScale.getValue(), d_color.getValue() );
        }
    }


}

} // namespace sofa::component::mapping::nonlinear
