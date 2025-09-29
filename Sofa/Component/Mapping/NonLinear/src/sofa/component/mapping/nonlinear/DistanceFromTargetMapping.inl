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

#include <sofa/component/mapping/nonlinear/DistanceFromTargetMapping.h>
#include <sofa/component/mapping/nonlinear/BaseNonLinearMapping.inl>
#include <sofa/core/BaseLocalMappingMatrix.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixConstraintEigenUtils.h>
#include <iostream>

namespace sofa::component::mapping::nonlinear
{

template <class TIn, class TOut>
DistanceFromTargetMapping<TIn, TOut>::DistanceFromTargetMapping()
    : d_indices(initData(&d_indices, "indices", "Indices of the parent points"))
    , d_targetPositions(initData(&d_targetPositions, "targetPositions", "Positions to compute the distances from"))
    , d_restDistances(initData(&d_restDistances, "restLengths", "Rest lengths of the connections"))
    , d_showObjectScale(initData(&d_showObjectScale, 0.f, "showObjectScale", "Scale for object display"))
    , d_color(initData(&d_color, sofa::type::RGBAColor::yellow(), "showColor", "Color for object display."))
{
}

template <class TIn, class TOut>
DistanceFromTargetMapping<TIn, TOut>::~DistanceFromTargetMapping()
{
}

template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::createTarget(unsigned index, const InCoord &position, Real distance)
{
    helper::WriteAccessor< Data< type::vector<Real> > > distances(d_restDistances);
    helper::WriteAccessor< Data<type::vector<unsigned> > > indices(d_indices);
    helper::WriteAccessor< Data<InVecCoord > > targetPositions(d_targetPositions);

    indices.push_back(index);
    targetPositions.push_back(position);
    distances.push_back(distance);
}

template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::updateTarget(unsigned index, const InCoord &position)
{
    helper::WriteAccessor< Data<InVecCoord > > targetPositions(d_targetPositions);
    helper::WriteAccessor< Data<type::vector<unsigned> > > indices(d_indices);

    // find the target with given index
    unsigned i=0; while(i<indices.size() && indices[i]!=index) i++;

    targetPositions[i] = position;
}

template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::updateTarget(unsigned index, SReal x, SReal y, SReal z)
{
    InCoord pos;
    TIn::set( pos, x, y, z );
    updateTarget( index, pos );
}

template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::clear()
{
    helper::WriteAccessor< Data< type::vector<Real> > > distances(d_restDistances);
    helper::WriteAccessor< Data<InVecCoord > > positions(d_targetPositions);
    helper::WriteAccessor< Data<type::vector<unsigned> > > indices(d_indices);

    distances.clear();
    positions.clear();
    indices.clear();

    this->getToModel()->resize( 0 );
}



template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::init()
{
    auto indices = sofa::helper::getWriteAccessor(d_indices);
    auto targetPositions = sofa::helper::getReadAccessor(d_targetPositions);
    if (indices.size() < targetPositions.size())
    {
        for (std::size_t i = indices.size(); i < targetPositions.size(); ++i)
        {
            indices.push_back(i);
        }
    }
    assert(d_indices.getValue().size() == d_targetPositions.getValue().size()) ;

    // unset distances are set to 0
    if(d_restDistances.getValue().size() != d_indices.getValue().size())
    {
        helper::WriteAccessor< Data< type::vector<Real> > > distances(d_restDistances);
        const unsigned prevsize = distances.size();
        distances.resize(d_indices.getValue().size() );
        for(unsigned i=prevsize; i<distances.size(); i++ )
            distances[i] = 0;
    }

    this->getToModel()->resize(d_indices.getValue().size() );

    Inherit1::init();  // applies the mapping, so after the Data init
}

template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::computeCoordPositionDifference( Direction& r, const InCoord& a, const InCoord& b )
{
    r = type::toVecN<Direction>(TIn::getCPos(b)-TIn::getCPos(a));
}

template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ , DataVecCoord_t<Out>& dOut, const DataVecCoord_t<In>& dIn)
{
    helper::WriteAccessor out(dOut);
    helper::WriteAccessor restDistances(d_restDistances);

    const helper::ReadAccessor in(dIn);
    const helper::ReadAccessor indices(d_indices);
    const helper::ReadAccessor targetPositions(d_targetPositions);

    this->m_jacobian.resizeBlocks(out.size(),in.size());
    directions.resize(out.size());
    invlengths.resize(out.size());


    for(unsigned i=0; i<indices.size() ; i++ )
    {
        Direction& gap = directions[i];

        // gap = in[indices[i]] - targetPositions[i] (only for position)
        computeCoordPositionDifference( gap, targetPositions[i], in[indices[i]] );

        Real gapNorm = gap.norm();
        out[i] = gapNorm - restDistances[i];  // output

        if( gapNorm>1.e-10 )
        {
            invlengths[i] = 1/gapNorm;
            gap *= invlengths[i];
        }
        else
        {
            invlengths[i] = 0;
            gap = Direction();
            gap[0]=1.0;  // arbitrary unit vector
        }

        for(unsigned j=0; j<Nout; j++)
        {
            this->m_jacobian.beginRow(i*Nout+j);
            for(unsigned k=0; k<Nin; k++ )
            {
                this->m_jacobian.insertBack( i*Nout+j, indices[i]*Nin+k, gap[k] );
            }
        }
    }

    this->m_jacobian.compress();
}

template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::matrixFreeApplyDJT(
    const core::MechanicalParams* mparams, Real kFactor,
    Data<VecDeriv_t<In>>& parentForce,
    const Data<VecDeriv_t<In>>& parentDisplacement,
    const Data<VecDeriv_t<Out>>& childForce)
{
    SOFA_UNUSED( mparams );
    const unsigned geometricStiffness = this->d_geometricStiffness.getValue().getSelectedId();

    helper::WriteAccessor parentForceAccessor(parentForce);
    const helper::ReadAccessor parentDisplacementAccessor(parentDisplacement);
    const helper::ReadAccessor childForceAccessor (childForce);
    const helper::ReadAccessor indices(d_indices);

    for (unsigned i = 0; i < indices.size(); ++i)
    {
        // force in compression (>0) can lead to negative eigen values in geometric stiffness
        // this results in an undefinite implicit matrix that causes instabilities
        // if stabilized GS (geometricStiffness==2) -> keep only force in extension
        if( childForceAccessor[i][0] < 0 || geometricStiffness==1 )
        {
            sofa::type::Mat<Nin,Nin,Real> b;  // = (I - uu^T)
            for(unsigned j=0; j<Nin; j++)
            {
                for(unsigned k=0; k<Nin; k++)
                {
                    b(j,k) = static_cast<Real>(1) * ( j==k ) - directions[i][j]*directions[i][k];
                }
            }
            // (I - uu^T)*f/l*kfactor  --  do not forget kfactor !
            b *= (Real)(childForceAccessor[i][0] * invlengths[i] * kFactor);
            // note that computing a block is not efficient here, but it would
            // makes sense for storing a stiffness matrix

            const auto& dx = parentDisplacementAccessor[indices[i]];
            Deriv_t<In> df;
            for(unsigned j=0; j<Nin; j++)
            {
                for(unsigned k=0; k<Nin; k++)
                {
                    df[j]+=b(j,k)*dx[k];
                }
            }
           // Deriv_t<In> df = b*dx;
            parentForceAccessor[indices[i]] += df;
        }
    }
}

template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::buildGeometricStiffnessMatrix(
    sofa::core::GeometricStiffnessMatrix* matrices)
{
    const unsigned geometricStiffness = this->d_geometricStiffness.getValue().getSelectedId();
    if( !geometricStiffness )
    {
        return;
    }

    const auto childForce = this->toModel->readTotalForces();
    helper::ReadAccessor< Data<type::vector<unsigned> > > indices(d_indices);
    const auto dJdx = matrices->getMappingDerivativeIn(this->fromModel).withRespectToPositionsIn(this->fromModel);

    for(sofa::Size i=0; i<indices.size(); i++)
    {
        const auto& force_i = childForce[i];

        // force in compression (>0) can lead to negative eigen values in geometric stiffness
        // this results in an undefinite implicit matrix that causes instabilities
        // if stabilized GS (geometricStiffness==2) -> keep only force in extension
        if( force_i[0] < 0 || geometricStiffness==1 )
        {
            size_t idx = indices[i];

            sofa::type::MatNoInit<Nin,Nin,Real> b;  // = (I - uu^T)
            for(unsigned j=0; j<Nin; j++)
            {
                for(unsigned k=0; k<Nin; k++)
                {
                    b(j,k) = static_cast<Real>(1) * ( j==k ) - directions[i][j]*directions[i][k];
                }
            }
            b *= force_i[0] * invlengths[i];  // (I - uu^T)*f/l

            dJdx(idx * Nin, idx * Nin) += b;
        }
    }
}

template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::doUpdateK(
    const core::MechanicalParams* mparams,
    const Data<VecDeriv_t<Out>>& childForce, SparseKMatrixEigen& matrix)
{
    SOFA_UNUSED(mparams);
    const unsigned geometricStiffness = this->d_geometricStiffness.getValue().getSelectedId();

    const helper::ReadAccessor childForceAccessor(childForce);
    const helper::ReadAccessor indices(d_indices);

    for (size_t i = 0; i < indices.size(); i++)
    {
        // force in compression (>0) can lead to negative eigen values in geometric stiffness
        // this results in an undefinite implicit matrix that causes instabilities
        // if stabilized GS (geometricStiffness==2) -> keep only force in extension
        if( childForceAccessor[i][0] < 0 || geometricStiffness==1 )
        {
            size_t idx = indices[i];

            sofa::type::Mat<Nin,Nin,Real> b;  // = (I - uu^T)
            for(unsigned j=0; j<Nin; j++)
            {
                for(unsigned k=0; k<Nin; k++)
                {
                    b(j,k) = static_cast<Real>(1) * ( j==k ) - directions[i][j]*directions[i][k];
                }
            }
            b *= childForceAccessor[i][0] * invlengths[i];  // (I - uu^T)*f/l

            matrix.addBlock(idx,idx,b);
        }
    }
}



template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if( !vparams->displayFlags().getShowMechanicalMappings() ) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    const float arrowsize = d_showObjectScale.getValue();
    if( arrowsize<0 ) return;

    typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();
    helper::ReadAccessor< Data<InVecCoord > > targetPositions(d_targetPositions);
    const helper::ReadAccessor< Data<type::vector<unsigned> > > indices(d_indices);

    type::vector< sofa::type::Vec3 > points;

    for(unsigned i=0; i<indices.size(); i++ )
    {
        points.push_back( sofa::type::Vec3(TIn::getCPos(targetPositions[i]) ) );
        points.push_back( sofa::type::Vec3(TIn::getCPos(pos[indices[i]]) ) );
    }

    if( !arrowsize )
        vparams->drawTool()->drawLines ( points, 1, d_color.getValue() );
    else
        for (unsigned int i=0; i<points.size()/2; ++i)
            vparams->drawTool()->drawArrow( points[2*i+1], points[2*i], arrowsize, d_color.getValue() );

}

} // namespace sofa::component::mapping::nonlinear
