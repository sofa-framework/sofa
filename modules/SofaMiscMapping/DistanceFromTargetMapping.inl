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
#ifndef SOFA_COMPONENT_MAPPING_DistanceFromTargetMapping_INL
#define SOFA_COMPONENT_MAPPING_DistanceFromTargetMapping_INL

#include "DistanceFromTargetMapping.h"
#include <sofa/core/visual/VisualParams.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
DistanceFromTargetMapping<TIn, TOut>::DistanceFromTargetMapping()
    : Inherit()
    , f_indices(initData(&f_indices, "indices", "Indices of the parent points"))
    , f_targetPositions(initData(&f_targetPositions, "targetPositions", "Positions to compute the distances from"))
    , f_restDistances(initData(&f_restDistances, "restLengths", "Rest lengths of the connections."))
    , _arrowSize(-1)
    , _color( 1,0,0,1 )
{
}

template <class TIn, class TOut>
DistanceFromTargetMapping<TIn, TOut>::~DistanceFromTargetMapping()
{
}

template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::createTarget( unsigned index, InCoord position, Real distance)
{
    helper::WriteAccessor< Data< vector<Real> > > distances(f_restDistances);
    helper::WriteAccessor< Data<vector<unsigned> > > indices(f_indices);
    helper::WriteAccessor< Data<InVecCoord > > targetPositions(f_targetPositions);

    indices.push_back(index);
    targetPositions.push_back(position);
    distances.push_back(distance);

//    cerr<<"DistanceFromTargetMapping<TIn, TOut>::createTarget index " << index << " at position " << position << ", distance = " << distances << endl;
}

template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::updateTarget( unsigned index, InCoord position)
{
    helper::WriteAccessor< Data<InVecCoord > > targetPositions(f_targetPositions);
    helper::WriteAccessor< Data<vector<unsigned> > > indices(f_indices);
//    cerr<<"DistanceFromTargetMapping<TIn, TOut>::updateTarget index " << index << " at position " << position << endl;

    // find the target with given index
    unsigned i=0; while(i<indices.size() && indices[i]!=index) i++;

    targetPositions[i] = position;
}

template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::clear()
{
    helper::WriteAccessor< Data< vector<Real> > > distances(f_restDistances);
    helper::WriteAccessor< Data<InVecCoord > > positions(f_targetPositions);
    helper::WriteAccessor< Data<vector<unsigned> > > indices(f_indices);

    distances.clear();
    positions.clear();
    indices.clear();

    this->getToModel()->resize( 0 );
}



template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::init()
{
    assert( f_indices.getValue().size()==f_targetPositions.getValue().size()) ;

    // unset distances are set to 0
    if(f_restDistances.getValue().size() != f_indices.getValue().size())
    {
        helper::WriteAccessor< Data< vector<Real> > > distances(f_restDistances);
        unsigned prevsize = distances.size();
        distances.resize( f_indices.getValue().size() );
        for(unsigned i=prevsize; i<distances.size(); i++ )
            distances[i] = 0;
    }

    this->getToModel()->resize( f_indices.getValue().size() );


    baseMatrices.resize( 1 );
    baseMatrices[0] = &jacobian;

    stiffnessBaseMatrices.resize(1);
    stiffnessBaseMatrices[0] = &K;

    this->Inherit::init();  // applies the mapping, so after the Data init
}

template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::computeCoordPositionDifference( InDeriv& r, const InCoord& a, const InCoord& b )
{
    // default implementation
    TIn::setDPos(r, TIn::getDPos(TIn::coordDifference(b,a))); //Generic code working also for type!=particles but not optimize for particles
}

template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteAccessor< Data<OutVecCoord> >  out = dOut;
    helper::ReadAccessor< Data<InVecCoord> >  in = dIn;
    helper::WriteAccessor<Data<vector<Real> > > restDistances(f_restDistances);
    helper::ReadAccessor< Data<vector<unsigned> > > indices(f_indices);
    helper::ReadAccessor< Data<InVecCoord > > targetPositions(f_targetPositions);

    jacobian.resizeBlocks(out.size(),in.size());
    directions.resize(out.size());
    invlengths.resize(out.size());

    for(unsigned i=0; i<indices.size(); i++ )
    {
        InDeriv& gap = directions[i];

        // gap = in[links[i][1]] - in[links[i][0]] (only for position)
        computeCoordPositionDifference( gap, targetPositions[i], in[indices[i]] );

        Real gapNorm = TIn::getDPos(gap).norm();
//        cerr<<"DistanceFromTargetMapping<TIn, TOut>::apply, gap = " << gap <<", norm = " << gapNorm << endl;
        out[i] = gapNorm - restDistances[i];  // output

        if( gapNorm>1.e-10 )
        {
            invlengths[i] = 1/gapNorm;
            gap *= invlengths[i];
        }
        else
        {
            invlengths[i] = 0;
            gap = InDeriv();
            gap[0]=1.0;  // arbitrary unit vector
        }

//        jacobian.beginRow(i);
        for(unsigned j=0; j<Nout; j++)
        {
            for(unsigned k=0; k<Nin; k++ )
            {
                jacobian.insertBack( i*Nout+j, indices[i]*Nin+k, gap[k] );
//                jacobian.add( i*Nout+j, indices[i]*Nin+k, gap[k] );
            }
        }

    }
//    cerr<<"DistanceFromTargetMapping<TIn, TOut>::apply, in = " << in << endl;
//    cerr<<"DistanceFromTargetMapping<TIn, TOut>::apply, target positions = " << positions << endl;
//    cerr<<"DistanceFromTargetMapping<TIn, TOut>::apply, out = " << out << endl;

    jacobian.compress();
//    cerr<<"DistanceFromTargetMapping<TIn, TOut>::apply, jacobian: "<<endl<< jacobian << endl;

}



template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    if( jacobian.rowSize() > 0 )
        jacobian.mult(dOut,dIn);
}

template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)
{
    if( jacobian.rowSize() > 0 )
        jacobian.addMultTranspose(dIn,dOut);
}

template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::applyJT(const core::ConstraintParams*, Data<InMatrixDeriv>& , const Data<OutMatrixDeriv>& )
{
    //    cerr<<"DistanceFromTargetMapping<TIn, TOut>::applyJT does nothing " << endl;
}


template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId )
{
    helper::WriteAccessor<Data<InVecDeriv> > parentForce (*parentDfId[this->fromModel.get(mparams)].write());
    helper::ReadAccessor<Data<InVecDeriv> > parentDisplacement (*mparams->readDx(this->fromModel));  // parent displacement
    Real kfactor = mparams->kFactor();
    helper::ReadAccessor<Data<OutVecDeriv> > childForce (*mparams->readF(this->toModel));
    helper::ReadAccessor< Data<vector<unsigned> > > indices(f_indices);

    for(unsigned i=0; i<indices.size(); i++ )
    {
        sofa::defaulttype::Mat<Nin,Nin,Real> b;  // = (I - uu^T)
        for(unsigned j=0; j<Nin; j++)
        {
            for(unsigned k=0; k<Nin; k++)
            {
                if( j==k )
                    b[j][k] = 1. - directions[i][j]*directions[i][k];
                else
                    b[j][k] =    - directions[i][j]*directions[i][k];
            }
        }
        b *= childForce[i][0] * invlengths[i] * kfactor;  // (I - uu^T)*f/l*kfactor     do not forget kfactor !
        // note that computing a block is not efficient here, but it would makes sense for storing a stiffness matrix

        InDeriv dx = parentDisplacement[indices[i]];
        InDeriv df;
        for(unsigned j=0; j<Nin; j++)
        {
            for(unsigned k=0; k<Nin; k++)
            {
                df[j]+=b[j][k]*dx[k];
            }
        }
       // InDeriv df = b*dx;
        parentForce[indices[i]] += df;
//        cerr<<"DistanceFromTargetMapping<TIn, TOut>::applyDJT, df = " << df << endl;
    }
}




template <class TIn, class TOut>
const sofa::defaulttype::BaseMatrix* DistanceFromTargetMapping<TIn, TOut>::getJ()
{
    return &jacobian;
}

template <class TIn, class TOut>
const vector<sofa::defaulttype::BaseMatrix*>* DistanceFromTargetMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}

template <class TIn, class TOut>
const vector<defaulttype::BaseMatrix*>* DistanceFromTargetMapping<TIn, TOut>::getKs()
{
//    helper::ReadAccessor<Data<OutVecDeriv> > childForce (*this->toModel->read(core::ConstVecDerivId::force()));
    const OutVecDeriv& childForce = this->toModel->readForces().ref();
    helper::ReadAccessor< Data<vector<unsigned> > > indices(f_indices);
    helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));

    K.resizeBlocks(in.size(),in.size());
    for(size_t i=0; i<indices.size(); i++)
    {
        size_t idx = indices[i];

        sofa::defaulttype::Mat<Nin,Nin,Real> b;  // = (I - uu^T)
        for(unsigned j=0; j<Nin; j++)
        {
            for(unsigned k=0; k<Nin; k++)
            {
                if( j==k )
                    b[j][k] = 1. - directions[i][j]*directions[i][k];
                else
                    b[j][k] =    - directions[i][j]*directions[i][k];
            }
        }
        b *= childForce[i][0] * invlengths[i];  // (I - uu^T)*f/l

//        std::cerr<<SOFA_CLASS_METHOD<<childForce[i][0]<<std::endl;

        K.beginBlockRow(idx);
        K.createBlock(idx,b);
        K.endBlockRow();
    }
    K.compress();

    return &stiffnessBaseMatrices;
}



template <class TIn, class TOut>
void DistanceFromTargetMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if( _arrowSize<0 ) return;

    typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();
    helper::ReadAccessor< Data<InVecCoord > > targetPositions(f_targetPositions);
    helper::ReadAccessor< Data<vector<unsigned> > > indices(f_indices);

    vector< sofa::defaulttype::Vector3 > points;

    for(unsigned i=0; i<indices.size(); i++ )
    {
        points.push_back( sofa::defaulttype::Vector3(TIn::getCPos(targetPositions[i]) ) );
        points.push_back( sofa::defaulttype::Vector3(TIn::getCPos(pos[indices[i]]) ) );
    }

    if( !_arrowSize )
        vparams->drawTool()->drawLines ( points, 1, _color );
    else
        for (unsigned int i=0; i<points.size()/2; ++i)
            vparams->drawTool()->drawArrow( points[2*i+1], points[2*i], _arrowSize, _color );

}



} // namespace mapping

} // namespace component

} // namespace sofa

#endif
