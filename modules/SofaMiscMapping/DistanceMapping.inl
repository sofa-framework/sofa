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

#include "DistanceMapping.h"
#include <sofa/core/visual/VisualParams.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
DistanceMapping<TIn, TOut>::DistanceMapping()
    : Inherit()
    , f_computeDistance(initData(&f_computeDistance, false, "computeDistance", "if 'computeDistance = true', then rest length of each element equal 0, otherwise rest length is the initial lenght of each of them"))
    , f_restLengths(initData(&f_restLengths, "restLengths", "Rest lengths of the connections"))
    , d_showObjectScale(initData(&d_showObjectScale, Real(0), "showObjectScale", "Scale for object display"))
    , d_color(initData(&d_color, defaulttype::Vec4f(1,1,0,1), "showColor", "Color for object display"))
{
}

template <class TIn, class TOut>
DistanceMapping<TIn, TOut>::~DistanceMapping()
{
}


template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::init()
{
    edgeContainer = dynamic_cast<topology::EdgeSetTopologyContainer*>( this->getContext()->getMeshTopology() );
    if( !edgeContainer ) serr<<"No EdgeSetTopologyContainer found ! "<<sendl;

    SeqEdges links = edgeContainer->getEdges();

    this->getToModel()->resize( links.size() );

    // compute the rest lengths if they are not known
    if( f_restLengths.getValue().size() != links.size() )
    {
        helper::WriteAccessor< Data<vector<Real> > > restLengths(f_restLengths);
        typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();
        restLengths.resize( links.size() );
        if(!(f_computeDistance.getValue()))
            for(unsigned i=0; i<links.size(); i++ )
                restLengths[i] = (pos[links[i][0]] - pos[links[i][1]]).norm();
        else
            for(unsigned i=0; i<links.size(); i++ )
                restLengths[i] = (Real)0.;
    }

    baseMatrices.resize( 1 );
    baseMatrices[0] = &jacobian;

    stiffnessBaseMatrices.resize(1);
    stiffnessBaseMatrices[0] = &K;

    this->Inherit::init();  // applies the mapping, so after the Data init
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::computeCoordPositionDifference( InDeriv& r, const InCoord& a, const InCoord& b )
{
    // default implementation
    TIn::setDPos(r, TIn::getDPos(TIn::coordDifference(b,a))); //Generic code working also for type!=particles but not optimize for particles
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteAccessor< Data<OutVecCoord> >  out = dOut;
    helper::ReadAccessor< Data<InVecCoord> >  in = dIn;
    helper::ReadAccessor<Data<vector<Real> > > restLengths(f_restLengths);
    SeqEdges links = edgeContainer->getEdges();

    //    jacobian.clear();
    jacobian.resizeBlocks(out.size(),in.size());
    directions.resize(out.size());
    invlengths.resize(out.size());

    for(unsigned i=0; i<links.size(); i++ )
    {
//        Block block;
//        typename Block::Line& gap = block[0];
        InDeriv& gap = directions[i];

        // gap = in[links[i][1]] - in[links[i][0]] (only for position)
        computeCoordPositionDifference( gap, in[links[i][0]], in[links[i][1]] );

        Real gapNorm = gap.norm();
        out[i] = gapNorm - restLengths[i];  // output

        // normalize
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

        // insert in increasing row and column order
//        jacobian.beginRow(i);
        if( links[i][1]<links[i][0])
        {
            for(unsigned j=0; j<Nout; j++)
            {
                for(unsigned k=0; k<Nin; k++ )
                {
                    jacobian.insertBack( i*Nout+j, links[i][1]*Nin+k, gap[k] );
//                    jacobian.add( i*Nout+j, links[i][1]*Nin+k, gap[k] );
                }
                for(unsigned k=0; k<Nin; k++ )
                {
//                    jacobian.add( i*Nout+j, links[i][0]*Nin+k, -gap[k] );
                    jacobian.insertBack( i*Nout+j, links[i][0]*Nin+k, -gap[k] );
                }
            }
        }
        else
        {
            for(unsigned j=0; j<Nout; j++)
            {
                for(unsigned k=0; k<Nin; k++ )
                {
//                    jacobian.add( i*Nout+j, links[i][0]*Nin+k, -gap[k] );
                    jacobian.insertBack( i*Nout+j, links[i][0]*Nin+k, -gap[k] );
                }
                for(unsigned k=0; k<Nin; k++ )
                {
//                    jacobian.add( i*Nout+j, links[i][1]*Nin+k, gap[k] );
                    jacobian.insertBack( i*Nout+j, links[i][1]*Nin+k, gap[k] );
                }
            }
        }
    }

    jacobian.compress();
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
void DistanceMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId )
{
    helper::WriteAccessor<Data<InVecDeriv> > parentForce (*parentDfId[this->fromModel.get(mparams)].write());
    helper::ReadAccessor<Data<InVecDeriv> > parentDisplacement (*mparams->readDx(this->fromModel));  // parent displacement
    Real kfactor = mparams->kFactor();
    helper::ReadAccessor<Data<OutVecDeriv> > childForce (*mparams->readF(this->toModel));
    SeqEdges links = edgeContainer->getEdges();

    for(unsigned i=0; i<links.size(); i++ )
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
//        cerr<<"DistanceMapping<TIn, TOut>::applyDJT, df = " << df << endl;
    }
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::applyJT(const core::ConstraintParams*, Data<InMatrixDeriv>& , const Data<OutMatrixDeriv>& )
{
    //    cerr<<"DistanceMapping<TIn, TOut>::applyJT(const core::ConstraintParams*, Data<InMatrixDeriv>& , const Data<OutMatrixDeriv>& ) does nothing " << endl;
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
const vector<defaulttype::BaseMatrix*>* DistanceMapping<TIn, TOut>::getKs()
{
//    helper::ReadAccessor<Data<OutVecDeriv> > childForce (*this->toModel->read(core::ConstVecDerivId::force()));
    const OutVecDeriv& childForce = this->toModel->readForces().ref();
    helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
    SeqEdges links = edgeContainer->getEdges();

    K.resizeBlocks(in.size(),in.size());
    for(size_t i=0; i<links.size(); i++)
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
        b *= childForce[i][0] * invlengths[i];  // (I - uu^T)*f/l

        K.beginBlockRow(links[i][0]);
        K.createBlock(links[i][0],b);
        K.createBlock(links[i][1],-b);
        K.endBlockRow();
        K.beginBlockRow(links[i][1]);
        K.createBlock(links[i][0],-b);
        K.createBlock(links[i][1],b);
        K.endBlockRow();
    }
    K.compress();

    return &stiffnessBaseMatrices;
}

template <class TIn, class TOut>
void DistanceMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if( !vparams->displayFlags().getShowMechanicalMappings() ) return;

    typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();
    SeqEdges links = edgeContainer->getEdges();


    if( d_showObjectScale.getValue() == 0 )
    {
        vector< defaulttype::Vector3 > points;
        for(unsigned i=0; i<links.size(); i++ )
        {
            points.push_back( sofa::defaulttype::Vector3( TIn::getCPos(pos[links[i][0]]) ) );
            points.push_back( sofa::defaulttype::Vector3( TIn::getCPos(pos[links[i][1]]) ));
        }
        vparams->drawTool()->drawLines ( points, 1, d_color.getValue() );
    }
    else
    {
        for(unsigned i=0; i<links.size(); i++ )
        {
            defaulttype::Vector3 p0 = TIn::getCPos(pos[links[i][0]]);
            defaulttype::Vector3 p1 = TIn::getCPos(pos[links[i][1]]);
            vparams->drawTool()->drawCylinder( p0, p1, d_showObjectScale.getValue(), d_color.getValue() );
        }
    }
}


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
