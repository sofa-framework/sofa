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
#include <sofa/component/mapping/nonlinear/BaseNonLinearMapping.inl>
#include <sofa/core/BaseLocalMappingMatrix.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/visual/VisualParams.h>
#include <iostream>
#include <sofa/simulation/Node.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixConstraintEigenUtils.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/component/mapping/nonlinear/DistanceMultiMapping.inl>

namespace sofa::component::mapping::nonlinear
{

template <class TIn, class TOut>
DistanceMapping<TIn, TOut>::DistanceMapping()
    : d_computeDistance(initData(&d_computeDistance, false, "computeDistance", "if 'computeDistance = true', then rest length of each element equal 0, otherwise rest length is the initial length of each of them"))
    , d_restLengths(initData(&d_restLengths, "restLengths", "Rest lengths of the connections"))
    , d_showObjectScale(initData(&d_showObjectScale, Real(0), "showObjectScale", "Scale for object display"))
    , d_color(initData(&d_color, sofa::type::RGBAColor::yellow(), "showColor", "Color for object display. (default=[1.0,1.0,0.0,1.0])"))
    , l_topology(initLink("topology", "link to the topology container"))
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

    if (!l_topology)
    {
        msg_error() << "No topology found";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (l_topology->getNbEdges() < 1)
    {
        msg_error() << "No topology component containing edges found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    const SeqEdges& links = l_topology->getEdges();
    typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();

    this->getToModel()->resize( links.size() );
    this->m_jacobian.resizeBlocks(links.size(),pos.size());

    directions.resize(links.size());
    invlengths.resize(links.size());

    // compute the rest lengths if they are not known
    if(d_restLengths.getValue().size() != links.size() )
    {
        helper::WriteOnlyAccessor< Data<type::vector<Real> > > restLengths(d_restLengths);
        restLengths->resize( links.size(), 0);

        if(!d_computeDistance.getValue())
        {
            for (std::size_t i = 0; i < links.size(); i++)
            {
                const auto& edge = links[i];
                restLengths[i] = (pos[edge[0]] - pos[edge[1]]).norm();
            }
        }
    }

    this->Inherit1::init();  // applies the mapping, so after the Data init
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::computeCoordPositionDifference( Direction& r, const Coord_t<In>& a, const Coord_t<In>& b )
{
    r = TIn::getCPos(b) - TIn::getCPos(a);
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ , DataVecCoord_t<Out>& dOut, const DataVecCoord_t<In>& dIn)
{
    helper::WriteOnlyAccessor<DataVecCoord_t<Out>> out(dOut);
    helper::ReadAccessor in(dIn);
    helper::ReadAccessor<Data<type::vector<Real> > > restLengths(d_restLengths);
    const SeqEdges& links = l_topology->getEdges();

    this->m_jacobian.clear();

    for (unsigned int i = 0; i < links.size(); ++i)
    {
        Direction& direction = directions[i];
        const auto& link = links[i];

        // gap = in[link[1]] - in[link[0]] (only for position)
        computeCoordPositionDifference( direction, in[link[0]], in[link[1]] );

        const Real distance = direction.norm();
        out[i] = distance - restLengths[i];  // output

        // normalize
        if (distance > std::numeric_limits<SReal>::epsilon())
        {
            invlengths[i] = 1 / distance;
            direction *= invlengths[i];
        }
        else
        {
            invlengths[i] = 0;

            // arbitrary vector mapping all directions
            static const Real p = static_cast<Real>(1) / std::sqrt(static_cast<Real>(In::spatial_dimensions));
            direction.fill(p);
        }

        sofa::type::fixed_array<JacobianEntry, 2> jacobianEntries {
            JacobianEntry{link[0], -direction},
            JacobianEntry{link[1], direction}
        };

        //invert to insert in increasing column order
        std::sort(jacobianEntries.begin(), jacobianEntries.end());

        this->m_jacobian.beginRow(i);
        for (const auto& [vertexId, jacobianValue] : jacobianEntries)
        {
            for (unsigned k = 0; k < In::spatial_dimensions; ++k)
            {
                this->m_jacobian.insertBack(i, vertexId * Nin + k, jacobianValue[k]);
            }
        }
    }

    this->m_jacobian.compress();
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::matrixFreeApplyDJT(
    const core::MechanicalParams* mparams, Real kFactor,
    Data<VecDeriv_t<In>>& parentForce,
    const Data<VecDeriv_t<In>>& parentDisplacement,
    const Data<VecDeriv_t<Out>>& childForce)
{
    SOFA_UNUSED(mparams);
    const unsigned geometricStiffness = this->d_geometricStiffness.getValue().getSelectedId();

    helper::WriteAccessor parentForceAccessor(parentForce);
    helper::ReadAccessor parentDisplacementAccessor(parentDisplacement);
    helper::ReadAccessor childForceAccessor(childForce);

    const SeqEdges& links = l_topology->getEdges();

    for (unsigned i = 0; i < links.size(); ++i)
    {
        // force in compression (>0) can lead to negative eigen values in geometric stiffness
        // this results in an undefinite implicit matrix that causes instabilities
        // if stabilized GS (geometricStiffness==2) -> keep only force in extension
        if( childForceAccessor[i][0] < 0 || geometricStiffness==1 )
        {
            sofa::type::Mat<Nin,Nin,Real> b;  // = (I - uu^T)
            for(unsigned j=0; j<In::spatial_dimensions; j++)
            {
                for(unsigned k=0; k<In::spatial_dimensions; k++)
                {
                    b(j,k) = static_cast<Real>(1) * ( j==k ) - directions[i][j]*directions[i][k];
                }
            }
            // (I - uu^T)*f/l*kfactor  --  do not forget kfactor !
            b *= (Real)(childForceAccessor[i][0] * invlengths[i] * kFactor);
            // note that computing a block is not efficient here, but it
            // would make sense for storing a stiffness matrix

            Deriv_t<In> dx = parentDisplacementAccessor[links[i][1]] - parentDisplacementAccessor[links[i][0]];
            Deriv_t<In> df;
            for(unsigned j=0; j<Nin; j++)
            {
                for(unsigned k=0; k<Nin; k++)
                {
                    df[j]+=b(j,k)*dx[k];
                }
            }
            parentForceAccessor[links[i][0]] -= df;
            parentForceAccessor[links[i][1]] += df;
        }
    }
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::doUpdateK(
    const core::MechanicalParams* mparams,
    const Data<VecDeriv_t<Out>>& childForce, SparseKMatrixEigen& matrix)
{
    SOFA_UNUSED(mparams);
    const unsigned geometricStiffness = this->d_geometricStiffness.getValue().getSelectedId();


    const helper::ReadAccessor childForceAccessor(childForce);
    const SeqEdges& links = l_topology->getEdges();

    for (size_t i = 0; i < links.size(); i++)
    {
        // force in compression (>0) can lead to negative eigen values in geometric stiffness
        // this results in an undefinite implicit matrix that causes instabilities
        // if stabilized GS (geometricStiffness==2) -> keep only force in extension
        if( childForceAccessor[i][0] < 0 || geometricStiffness==1 )
        {
            sofa::type::Mat<Nin,Nin,Real> b;  // = (I - uu^T)

            for(unsigned j=0; j<In::spatial_dimensions; j++)
            {
                for(unsigned k=0; k<In::spatial_dimensions; k++)
                {
                    b(j,k) = static_cast<Real>(1) * ( j==k ) - directions[i][j]*directions[i][k];
                }
            }
            b *= childForceAccessor[i][0] * invlengths[i];  // (I - uu^T)*f/l

            // Note that 'links' is not sorted so the matrix can not be filled-up in order
            matrix.addBlock(links[i][0],links[i][0],b);
            matrix.addBlock(links[i][0],links[i][1],-b);
            matrix.addBlock(links[i][1],links[i][0],-b);
            matrix.addBlock(links[i][1],links[i][1],b);
        }
    }
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::buildGeometricStiffnessMatrix(
    sofa::core::GeometricStiffnessMatrix* matrices)
{
    const unsigned& geometricStiffness = this->d_geometricStiffness.getValue().getSelectedId();
    if( !geometricStiffness )
    {
        return;
    }

    const auto childForce = this->toModel->readTotalForces();
    const SeqEdges& links = l_topology->getEdges();
    const auto dJdx = matrices->getMappingDerivativeIn(this->fromModel).withRespectToPositionsIn(this->fromModel);

    for(sofa::Size i=0; i<links.size(); i++)
    {
        const Deriv_t<Out> force_i = childForce[i];

        // force in compression (>0) can lead to negative eigen values in geometric stiffness
        // this results in an undefinite implicit matrix that causes instabilities
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
                    b(j,k) = static_cast<Real>(1) * ( j==k ) - dir[j] * dir[k];
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
void DistanceMapping<TIn, TOut>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    SOFA_UNUSED(params);

    if (!onlyVisible) return;
    if (!this->getFromModel()) return;

    const auto bbox = this->getFromModel()->computeBBox(); //this may compute twice the mstate bbox, but there is no way to determine if the bbox has already been computed
    this->f_bbox.setValue(std::move(bbox));
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
        points.reserve(2 * links.size());
        for (const auto& link : links)
        {
            points.emplace_back( TIn::getCPos(pos[link[0]]) );
            points.emplace_back( TIn::getCPos(pos[link[1]]) );
        }
        vparams->drawTool()->drawLines ( points, 1, d_color.getValue() );
    }
    else
    {
        vparams->drawTool()->enableLighting();
        for (const auto& link : links)
        {
            const type::Vec3 p0 = TIn::getCPos(pos[link[0]]);
            const type::Vec3 p1 = TIn::getCPos(pos[link[1]]);
            vparams->drawTool()->drawCylinder( p0, p1, (float)d_showObjectScale.getValue(), d_color.getValue() );
        }
    }

}

} // namespace sofa::component::mapping::nonlinear
