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
#ifndef SOFA_COMPONENT_MAPPING_DistanceMapping_INL
#define SOFA_COMPONENT_MAPPING_DistanceMapping_INL

#include "../deformationMapping/DistanceMapping.h"
#include <sofa/core/visual/VisualParams.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;


template <class TIn, class TOut>
DistanceMapping<TIn, TOut>::DistanceMapping()
    : Inherit()
    , f_indices(initData(&f_indices, "indices", "Indices of the parent points"))
    , f_positions(initData(&f_positions, "positions", "Positions to compute the distances from"))
    , f_restDistances(initData(&f_restDistances, "restLengths", "Rest lengths of the connections."))
{
}

template <class TIn, class TOut>
DistanceMapping<TIn, TOut>::~DistanceMapping()
{
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::createTarget( unsigned index, InCoord position, Real distance)
{
    helper::WriteAccessor< Data< vector<Real> > > distances(f_restDistances);
    helper::WriteAccessor< Data<vector<unsigned> > > indices(f_indices);
    helper::WriteAccessor< Data<InVecCoord > > positions(f_positions);

    indices.push_back(index);
    positions.push_back(position);
    distances.push_back(distance);

//    cerr<<"DistanceMapping<TIn, TOut>::createTarget index " << index << " at position " << position << endl;
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::updateTarget( unsigned index, InCoord position)
{
    helper::WriteAccessor< Data<InVecCoord > > positions(f_positions);
    helper::WriteAccessor< Data<vector<unsigned> > > indices(f_indices);
//    cerr<<"DistanceMapping<TIn, TOut>::updateTarget index " << index << " at position " << position << endl;

    // find the target with given index
    unsigned i=0; while(i<indices.size() && indices[i]!=index) i++;

    positions[i] = position;
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::clear()
{
    helper::WriteAccessor< Data< vector<Real> > > distances(f_restDistances);
    helper::WriteAccessor< Data<InVecCoord > > positions(f_positions);
    helper::WriteAccessor< Data<vector<unsigned> > > indices(f_indices);

    distances.clear();
    positions.clear();
    indices.clear();

    this->getToModel()->resize( 0 );
}



template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::init()
{
    assert( f_indices.getValue().size()==f_positions.getValue().size()) ;

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

    this->Inherit::init();  // applies the mapping, so after the Data init
}



template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteAccessor< Data<OutVecCoord> >  out = dOut;
    helper::ReadAccessor< Data<InVecCoord> >  in = dIn;
    helper::WriteAccessor<Data<vector<Real> > > restDistances(f_restDistances);
    helper::ReadAccessor< Data<vector<unsigned> > > indices(f_indices);
    helper::ReadAccessor< Data<InVecCoord > > positions(f_positions);

    jacobian.resizeBlocks(out.size(),in.size());

    for(unsigned i=0; i<indices.size(); i++ )
    {
        InDeriv gap = in[indices[i]] - positions[i];
        Real gapNorm = gap.norm();
        out[i] = gapNorm - restDistances[i];  // output

        if( gapNorm>1.e-10 )
        {
            gap *= 1/gapNorm;
        }
        else
        {
            gap = InDeriv();
            gap[0]=1.0;  // arbitrary unit vector
        }

        jacobian.beginRow(i);
        for(unsigned j=0; j<Nout; j++)
        {
            for(unsigned k=0; k<Nin; k++ )
            {
                jacobian.set( i*Nout+j, indices[i]*Nin+k, gap[k] );
            }
        }

    }

    jacobian.endEdit();
    //      cerr<<"DistanceMapping<TIn, TOut>::apply, jacobian: "<<endl<< jacobian << endl;

}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    if( jacobian.rowSize() > 0 )
        jacobian.mult(dOut,dIn);
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)
{
    if( jacobian.rowSize() > 0 )
        jacobian.addMultTranspose(dIn,dOut);
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::applyJT(const core::ConstraintParams*, Data<InMatrixDeriv>& , const Data<OutMatrixDeriv>& )
{
    //    cerr<<"DistanceMapping<TIn, TOut>::applyJT does nothing " << endl;
}


template <class TIn, class TOut>
const sofa::defaulttype::BaseMatrix* DistanceMapping<TIn, TOut>::getJ()
{
    return &jacobian;
}

template <class TIn, class TOut>
const vector<sofa::defaulttype::BaseMatrix*>* DistanceMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();
    helper::ReadAccessor< Data<InVecCoord > > positions(f_positions);
    helper::ReadAccessor< Data<vector<unsigned> > > indices(f_indices);

    vector< Vec3d > points;

    for(unsigned i=0; i<indices.size(); i++ )
    {
        points.push_back(positions[i]);
        points.push_back(pos[indices[i]]);
    }
    vparams->drawTool()->drawLines ( points, 1, Vec<4,float> ( 1,0,0,1 ) );
}


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
