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

#include <sofa/component/mapping/nonlinear/DistanceMapping.h>
#include <sofa/core/BaseLocalMappingMatrix.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ConstraintParams.h>
#include <iostream>
#include <sofa/simulation/Node.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixConstraintEigenUtils.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/component/mapping/nonlinear/DistanceMultiMapping.inl>

namespace sofa::component::mapping::nonlinear
{

template <class TIn, class TOut>
DistanceMapping<TIn, TOut>::DistanceMapping()
    : Inherit()
    , f_computeDistance(initData(&f_computeDistance, false, "computeDistance", "if 'computeDistance = true', then rest length of each element equal 0, otherwise rest length is the initial lenght of each of them"))
    , f_restLengths(initData(&f_restLengths, "restLengths", "Rest lengths of the connections"))
    , d_showObjectScale(initData(&d_showObjectScale, Real(0), "showObjectScale", "Scale for object display"))
    , d_color(initData(&d_color,sofa::type::RGBAColor::yellow(), "showColor", "Color for object display. (default=[1.0,1.0,0.0,1.0])"))
    , l_topology(initLink("topology", "link to the topology container"))
{
}

template <class TIn, class TOut>
DistanceMapping<TIn, TOut>::~DistanceMapping()
{
}


template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::init()
{
    if (l_topology.empty())
    {
        msg_warning() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (l_topology->getNbEdges() < 1)
    {
        msg_error() << "No topology component containg edges found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    const SeqEdges& links = l_topology->getEdges();
    typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();

    this->getToModel()->resize( links.size() );
    jacobian.resizeBlocks(links.size(),pos.size());

    directions.resize(links.size());
    invlengths.resize(links.size());

    // only used for warning message
    bool compliance = ((simulation::Node*)(this->getContext()))->forceField.size() && ((simulation::Node*)(this->getContext()))->forceField[0]->isCompliance.getValue();

    // compute the rest lengths if they are not known
    if( f_restLengths.getValue().size() != links.size() )
    {
        helper::WriteOnlyAccessor< Data<type::vector<Real> > > restLengths(f_restLengths);
        restLengths.resize( links.size() );
        if(!(f_computeDistance.getValue()))
        {
            for(unsigned i=0; i<links.size(); i++ )
            {
                restLengths[i] = (pos[links[i][0]] - pos[links[i][1]]).norm();

                msg_error_when(restLengths[i] <= std::numeric_limits<SReal>::epsilon() && compliance) << "Null rest Length cannot be used for stable compliant constraint, prefer to use a DifferenceMapping for this dof " << i << " if used with a compliance";
            }
        }
        else
        {
            msg_error_when(compliance) << "Null rest Lengths cannot be used for stable compliant constraint, prefer to use a DifferenceMapping if those dofs are used with a compliance";
            for(unsigned i=0; i<links.size(); i++ )
                restLengths[i] = (Real)0.;
        }
    }
    else if (compliance) // for warning message
    {
        helper::ReadAccessor< Data<type::vector<Real> > > restLengths(f_restLengths);
        std::stringstream sstream;
        for (unsigned i = 0; i < links.size(); i++)
            if (restLengths[i] <= std::numeric_limits<SReal>::epsilon()) 
                sstream << "Null rest Length cannot be used for stable compliant constraint, prefer to use a DifferenceMapping for this dof " << i << " if used with a compliance \n";
        msg_error_when(!sstream.str().empty()) << sstream.str();
    }

    baseMatrices.resize( 1 );
    baseMatrices[0] = &jacobian;

    this->Inherit::init();  // applies the mapping, so after the Data init
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::computeCoordPositionDifference( Direction& r, const InCoord& a, const InCoord& b )
{
    r = TIn::getCPos(b)-TIn::getCPos(a);
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteOnlyAccessor< Data<OutVecCoord> >  out = dOut;
    helper::ReadAccessor< Data<InVecCoord> >  in = dIn;
    helper::ReadAccessor<Data<type::vector<Real> > > restLengths(f_restLengths);
    const SeqEdges& links = l_topology->getEdges();

    jacobian.clear();

    for(unsigned i=0; i<links.size(); i++ )
    {
        Direction& gap = directions[i];

        // gap = in[links[i][1]] - in[links[i][0]] (only for position)
        computeCoordPositionDifference( gap, in[links[i][0]], in[links[i][1]] );

        Real gapNorm = gap.norm();
        out[i] = gapNorm - restLengths[i];  // output

        // normalize
        if( gapNorm>std::numeric_limits<SReal>::epsilon() )
        {
            invlengths[i] = 1/gapNorm;
            gap *= invlengths[i];
        }
        else
        {
            invlengths[i] = 0;

            // arbritary vector mapping all directions
            static const Real p = static_cast<Real>(1) / std::sqrt(static_cast<Real>(In::spatial_dimensions));
            gap.fill(p);
        }

        // insert in increasing column order
        if( links[i][1]<links[i][0])
        {
            jacobian.beginRow(i);
            for(unsigned k=0; k<In::spatial_dimensions; k++ )
            {
                jacobian.insertBack( i, links[i][1]*Nin+k, gap[k] );
            }
            for(unsigned k=0; k<In::spatial_dimensions; k++ )
            {
                jacobian.insertBack( i, links[i][0]*Nin+k, -gap[k] );
            }
        }
        else
        {
            jacobian.beginRow(i);
            for(unsigned k=0; k<In::spatial_dimensions; k++ )
            {
                jacobian.insertBack( i, links[i][0]*Nin+k, -gap[k] );
            }
            for(unsigned k=0; k<In::spatial_dimensions; k++ )
            {
                jacobian.insertBack( i, links[i][1]*Nin+k, gap[k] );
            }
        }
    }

    jacobian.compress();
}


template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& out, const Data<InVecDeriv>& in)
{
    if( jacobian.rowSize() )
    {
        auto dOutWa = sofa::helper::getWriteOnlyAccessor(out);
        auto dInRa = sofa::helper::getReadAccessor(in);
        jacobian.mult(dOutWa.wref(),dInRa.ref());
    }
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& out, const Data<OutVecDeriv>& in)
{
    if( jacobian.rowSize() )
    {
        auto dOutRa = sofa::helper::getReadAccessor(in);
        auto dInWa = sofa::helper::getWriteOnlyAccessor(out);
        jacobian.addMultTranspose(dInWa.wref(),dOutRa.ref());
    }
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId)
{
    const unsigned geometricStiffness = d_geometricStiffness.getValue().getSelectedId();
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
                sofa::type::Mat<Nin,Nin,Real> b;  // = (I - uu^T)
                for(unsigned j=0; j<In::spatial_dimensions; j++)
                {
                    for(unsigned k=0; k<In::spatial_dimensions; k++)
                    {
                        b[j][k] = static_cast<Real>(1) * ( j==k ) - directions[i][j]*directions[i][k];
                    }
                }
                // (I - uu^T)*f/l*kfactor  --  do not forget kfactor !
                b *= (Real)(childForce[i][0] * invlengths[i] * kfactor);
                // note that computing a block is not efficient here, but it
                // would make sense for storing a stiffness matrix

                InDeriv dx = parentDisplacement[links[i][1]] - parentDisplacement[links[i][0]];
                InDeriv df;
                for(unsigned j=0; j<Nin; j++)
                {
                    for(unsigned k=0; k<Nin; k++)
                    {
                        df[j]+=b[j][k]*dx[k];
                    }
                }
                parentForce[links[i][0]] -= df;
                parentForce[links[i][1]] += df;
            }
        }
    }
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::applyJT(const core::ConstraintParams* cparams, Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in)
{
    SOFA_UNUSED(cparams);
    const OutMatrixDeriv& childMat  = sofa::helper::getReadAccessor(in).ref();
    InMatrixDeriv&        parentMat = sofa::helper::getWriteAccessor(out).wref();
    addMultTransposeEigen(parentMat, jacobian.compressedMatrix, childMat);
}


template <class TIn, class TOut>
const sofa::linearalgebra::BaseMatrix* DistanceMapping<TIn, TOut>::getJ()
{
    return &jacobian;
}

template <class TIn, class TOut>
const type::vector<sofa::linearalgebra::BaseMatrix*>* DistanceMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}



template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::updateK(const core::MechanicalParams *mparams, core::ConstMultiVecDerivId childForceId )
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
            sofa::type::Mat<Nin,Nin,Real> b;  // = (I - uu^T)

            for(unsigned j=0; j<In::spatial_dimensions; j++)
            {
                for(unsigned k=0; k<In::spatial_dimensions; k++)
                {
                    b[j][k] = static_cast<Real>(1) * ( j==k ) - directions[i][j]*directions[i][k];
                }
            }
            b *= childForce[i][0] * invlengths[i];  // (I - uu^T)*f/l

            // Note that 'links' is not sorted so the matrix can not be filled-up in order
            K.addBlock(links[i][0],links[i][0],b);
            K.addBlock(links[i][0],links[i][1],-b);
            K.addBlock(links[i][1],links[i][0],-b);
            K.addBlock(links[i][1],links[i][1],b);
        }
    }
    K.compress();
}

template <class TIn, class TOut>
const linearalgebra::BaseMatrix* DistanceMapping<TIn, TOut>::getK()
{
    return &K;
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::buildGeometricStiffnessMatrix(
    sofa::core::GeometricStiffnessMatrix* matrices)
{
    const unsigned& geometricStiffness = d_geometricStiffness.getValue().getSelectedId();
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

        // force in compression (>0) can lead to negative eigen values in geometric stiffness
        // this results in a undefinite implicit matrix that causes instabilies
        // if stabilized GS (geometricStiffness==2) -> keep only force in extension
        if( force_i[0] < 0 || geometricStiffness==1 )
        {
            const sofa::topology::Edge link = links[i];
            const Direction& dir = directions[i];
            sofa::type::Mat<In::spatial_dimensions, In::spatial_dimensions, Real> b;  // = (I - uu^T)

            for(unsigned j=0; j<In::spatial_dimensions; j++)
            {
                for(unsigned k=0; k<In::spatial_dimensions; k++)
                {
                    b[j][k] = static_cast<Real>(1) * ( j==k ) - dir[j] * dir[k];
                }
            }
            b *= force_i[0] * invlengths[i];  // (I - uu^T)*f/l

            dJdx(link[0] * Nin, link[0] * Nin) += b;
            dJdx(link[0] * Nin, link[1] * Nin) += -b;
            dJdx(link[1] * Nin, link[0] * Nin) += -b;
            dJdx(link[1] * Nin, link[1] * Nin) += b;
        }
    }
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
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
