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
#ifndef SOFA_COMPONENT_MAPPING_ProjectionToLineMapping_INL
#define SOFA_COMPONENT_MAPPING_ProjectionToLineMapping_INL

#include "ProjectionToLineMapping.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/RGBAColor.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
ProjectionToTargetLineMapping<TIn, TOut>::ProjectionToTargetLineMapping()
    : Inherit()
    , f_indices(initData(&f_indices, "indices", "Indices of the parent points (if empty, all input dofs are mapped)"))
    , f_origins(initData(&f_origins, "origins", "Origins of the lines on which the points are projected"))
    , f_directions(initData(&f_directions, "directions", "Directions of the lines on which the points are projected"))
    , d_drawScale(initData(&d_drawScale, SReal(10), "drawScale", "Draw scale"))
    , d_drawColor(initData(&d_drawColor, defaulttype::RGBAColor(0,1,0,1), "drawColor", "Draw color. (default=[0.0,1.0,0.0,1.0])"))
{
}

template <class TIn, class TOut>
void ProjectionToTargetLineMapping<TIn, TOut>::init()
{
    baseMatrices.resize( 1 );
    baseMatrices[0] = &jacobian;

    reinit();
    this->Inherit::init();
}


template <class TIn, class TOut>
void ProjectionToTargetLineMapping<TIn, TOut>::reinit()
{
    helper::ReadAccessor< Data<helper::vector<unsigned> > > indices(f_indices);

    size_t nb = indices.empty() ? this->getFromModel()->getSize() : indices.size(); // if indices is empty, mapping every input dofs

    this->getToModel()->resize( nb );

    // ensuring direction are normalized
    helper::WriteAccessor< Data<OutVecCoord> > directions(f_directions);
    for( size_t i=0 ; i<directions.size() ; ++i )
        directions[i].normalize( OutCoord(1,0,0) ); // failsafe for null norms


    // precompute constant jacobian
    jacobian.resizeBlocks(nb,this->getFromModel()->getSize());
    for(unsigned i=0; i<nb; i++ )
    {
        const OutCoord& n = i<directions.size() ? directions[i] : directions.ref().back();
        const unsigned& index = indices.empty() ? i : indices[i] ;

        for(unsigned j=0; j<Nout; j++)
        {
            jacobian.beginRow( i*Nout+j );
            for(unsigned k=0; k<Nout; k++ )
            {
                jacobian.insertBack( i*Nout+j, index*Nin+k, n[j]*n[k] );
            }
        }
    }
    jacobian.compress();

    this->Inherit::reinit();
}


template <class TIn, class TOut>
void ProjectionToTargetLineMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteAccessor< Data<OutVecCoord> >  out = dOut;
    helper::ReadAccessor< Data<InVecCoord> >  in = dIn;
    helper::ReadAccessor< Data<helper::vector<unsigned> > > indices(f_indices);
    helper::ReadAccessor< Data<OutVecCoord> > origins(f_origins);
    helper::ReadAccessor< Data<OutVecCoord> > directions(f_directions);

    size_t nb = indices.empty() ? this->getFromModel()->getSize() : indices.size(); // if indices is empty, mapping every input dofs

    for(unsigned i=0; i<nb; i++ )
    {
        const unsigned& index = indices.empty() ? i : indices[i] ;
        const InCoord& p = in[index];
        const OutCoord& o = i<origins.size() ? origins[i] : origins.ref().back();
        const OutCoord& n = i<directions.size() ? directions[i] : directions.ref().back();

        out[i] = o + n * ( n * ( TIn::getCPos(p) - o ) ); // projection on the line
    }
}



template <class TIn, class TOut>
void ProjectionToTargetLineMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    jacobian.mult(dOut,dIn);
}

template <class TIn, class TOut>
void ProjectionToTargetLineMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)
{
    jacobian.addMultTranspose(dIn,dOut);
}

template <class TIn, class TOut>
void ProjectionToTargetLineMapping<TIn, TOut>::applyJT(const core::ConstraintParams*, Data<InMatrixDeriv>& , const Data<OutMatrixDeriv>& )
{
}



template <class TIn, class TOut>
const sofa::defaulttype::BaseMatrix* ProjectionToTargetLineMapping<TIn, TOut>::getJ()
{
    return &jacobian;
}

template <class TIn, class TOut>
const helper::vector<sofa::defaulttype::BaseMatrix*>* ProjectionToTargetLineMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}



template <class TIn, class TOut>
void ProjectionToTargetLineMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if( !vparams->displayFlags().getShowMechanicalMappings() ) return;

    const SReal& scale = d_drawScale.getValue();
    if(!scale) return;

    const defaulttype::Vec4f color = d_drawColor.getValue();

    helper::ReadAccessor< Data<OutVecCoord> > origins(f_origins);
    helper::ReadAccessor< Data<OutVecCoord> > directions(f_directions);

    helper::vector< defaulttype::Vector3 > points;

    size_t nb = std::max( directions.size(), origins.size() );

    for(unsigned i=0; i<nb; i++ )
    {
        const OutCoord& o = i<origins.size() ? origins[i] : origins.ref().back();
        const OutCoord& n = i<directions.size() ? directions[i] : directions.ref().back();

        points.push_back( o - n*scale );
        points.push_back( o + n*scale );
    }

    vparams->drawTool()->drawLines( points, 1, color );
}


template <class TIn, class TOut>
void ProjectionToTargetLineMapping<TIn, TOut>::updateForceMask()
{
    helper::ReadAccessor< Data<helper::vector<unsigned> > > indices(f_indices);
    if( indices.empty() ) return Inherit::updateForceMask(); // all dofs are mapped

    for( unsigned i=0 ; i<indices.size() ; i++ )
        if( this->maskTo->getEntry(i) )
            this->maskFrom->insertEntry( indices[i] );
}


//////////////////


template <class TIn, class TOut>
ProjectionToLineMultiMapping<TIn, TOut>::ProjectionToLineMultiMapping()
    : Inherit1()
    , f_indices(initData(&f_indices, "indices", "Indices of the parent points (if empty, all input dofs are mapped)"))
    , d_drawScale(initData(&d_drawScale, SReal(10), "drawScale", "Draw scale"))
    , d_drawColor(initData(&d_drawColor, defaulttype::RGBAColor(0,1,0,1), "drawColor", "Draw color. (default=[0.0,1.0,0.0,1.0])"))
{
}

template <class TIn, class TOut>
void ProjectionToLineMultiMapping<TIn, TOut>::init()
{
    baseMatrices.resize( 2 );
    baseMatrices[0] = &jacobian0;
    baseMatrices[1] = &jacobian1;

    reinit();
    this->Inherit1::init();
}


template <class TIn, class TOut>
void ProjectionToLineMultiMapping<TIn, TOut>::reinit()
{
    helper::ReadAccessor< Data<helper::vector<unsigned> > > indices(f_indices);

    size_t nb = indices.empty() ? this->getFromModels()[0]->getSize() : indices.size(); // if indices is empty, mapping every input dofs

    this->getToModels()[0]->resize( nb );

    assert(this->getFromModels()[1]->getSize()==2); // center + direction
    jacobian1.resizeBlocks(nb,this->getFromModels()[1]->getSize());

    this->Inherit1::reinit();
}


template <class TIn, class TOut>
void ProjectionToLineMultiMapping<TIn, TOut>::apply(const core::MechanicalParams */*mparams*/, const helper::vector<OutDataVecCoord*>& dataVecOutPos, const helper::vector<const InDataVecCoord*>& dataVecInPos)
{
    helper::WriteAccessor< Data<OutVecCoord> >  out = *dataVecOutPos[0];
    helper::ReadAccessor< Data<InVecCoord> >  in = *dataVecInPos[0];
    helper::ReadAccessor< Data<InVecCoord> >  line = *dataVecInPos[1];
    helper::ReadAccessor< Data<helper::vector<unsigned> > > indices(f_indices);

    size_t nb = indices.empty() ? this->getFromModels()[0]->getSize() : indices.size(); // if indices is empty, mapping every input dofs

    jacobian0.resizeBlocks(nb,this->getFromModels()[0]->getSize());
    jacobian1.compressedMatrix.setZero();

    const OutCoord& o = line[0];
    const OutCoord& d = line[1];
    OutCoord dn = d;
    Real d_norm2 = dn.norm2();

    assert( d_norm2 > std::numeric_limits<Real>::epsilon() );

    Real d_norm = std::sqrt(d_norm2 );
    dn.normalizeWithNorm( d_norm );


    for(unsigned i=0; i<nb; i++ )
    {
        const unsigned& index = indices.empty() ? i : indices[i] ;
        const InCoord& p = in[index];

        out[i] = o + dn * ( dn * ( p - o ) ); // projection on the line

        defaulttype::Matrix3 Jp = dyad( dn, dn );
        defaulttype::Matrix3 Jo = defaulttype::Matrix3::Identity() - Jp;
        defaulttype::Matrix3 Jd;

        // generated by maple
        // because of the direction normalization that complexifies
        // TODO this code is improvable
        const Real& n0 = d[0];
        const Real& n1 = d[1];
        const Real& n2 = d[2];
        const Real& t1 = Jp[2][2];//n2 * n2;
        const Real& t2 = Jp[1][1];//n1 * n1;
        const Real& t3 = Jp[0][0];//n0 * n0;
        const Real& t4 = d_norm2; //t1 + t2 + t3; // squared norm
        Real t5 = 1./(d_norm2 * d_norm);//pow(t4, -0.3e1 / 0.2e1); // 1/(squared norm * norm)
        Real t40 = t4 * t5; //
        Real t6 = o[0] - p[0];
        Real t7 = p[1] - o[1];
        Real t8 = p[2] - o[2];
        Real t9 = n1 * t7;
        Real t10 = n2 * t8;
        Real t30 = t3 * t5;
        Real t11 = n0 * t6;
        Real t12 = (-t9 - t10 + t11) * t40;
        t6 = (t30 * t6 - t40 * t6 + (-t9 - t10) * t5 * n0) * t40;
        Real t20 = t2 * t5;
        t7 = (t20 * t7 - t40 * t7 + (t10 - t11) * t5 * n1) * t40;
        t10 = -t12 * n1 * t5 + t7;
        Real t100 = t1 * t5;
        t8 = (t100 * t8 - t40 * t8 + (t9 - t11) * t5 * n2) * t40;
        t9 = -t12 * n2 * t5 + t8;
        t5 = t12 * t5 * n0 + t6;
        Jd[0][0] = t12 * (t30 - t40) + n0 * t6;
        Jd[0][1] = -n0 * t10;
        Jd[0][2] = -n0 * t9;
        Jd[1][0] = n1 * t5;
        Jd[1][1] = t12 * (t20 - t40) - n1 * t7;
        Jd[1][2] = -n1 * t9;
        Jd[2][0] = n2 * t5;
        Jd[2][1] = -n2 * t10;
        Jd[2][2] = t12 * (t100 - t40) - n2 * t8;


        for(unsigned j=0; j<Nout; j++)
        {
            jacobian0.beginRow( i*Nout+j );
            jacobian1.beginRow( i*Nout+j );
            for(unsigned k=0; k<Nout; k++ )
            {
                jacobian0.insertBack( i*Nout+j, index*Nin+k, Jp[j][k] ); // dp
                jacobian1.insertBack( i*Nout+j, k, Jo[j][k] ); // do
            }
            for(unsigned k=0; k<Nout; k++ )
                jacobian1.insertBack( i*Nout+j, Nout+k, Jd[j][k] ); // do
        }
    }

    jacobian0.compress();
    jacobian1.compress();

}

template <class TIn, class TOut>
void ProjectionToLineMultiMapping<TIn, TOut>::applyJ(const core::MechanicalParams */*mparams*/, const helper::vector<OutDataVecDeriv*>& dataVecOutVel, const helper::vector<const InDataVecDeriv*>& dataVecInVel)
{
    jacobian0.mult(*dataVecOutVel[0],*dataVecInVel[0]);
    jacobian1.mult(*dataVecOutVel[0],*dataVecInVel[1]);
}

template <class TIn, class TOut>
void ProjectionToLineMultiMapping<TIn, TOut>::applyJT(const core::MechanicalParams */*mparams*/, const helper::vector<InDataVecDeriv*>& dataVecOutForce, const helper::vector<const OutDataVecDeriv*>& dataVecInForce)
{
    jacobian0.addMultTranspose(*dataVecOutForce[0],*dataVecInForce[0]);
    jacobian1.addMultTranspose(*dataVecOutForce[1],*dataVecInForce[0]);
}



template <class TIn, class TOut>
const helper::vector<sofa::defaulttype::BaseMatrix*>* ProjectionToLineMultiMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}



template <class TIn, class TOut>
void ProjectionToLineMultiMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if( !vparams->displayFlags().getShowMechanicalMappings() ) return;

    const SReal& scale = d_drawScale.getValue();
    if(!scale) return;

    const defaulttype::Vec4f color = d_drawColor.getValue();

    helper::ReadAccessor<Data<InVecCoord> > line = this->getFromModels()[1]->read(core::ConstVecCoordId::position());


    helper::vector< defaulttype::Vector3 > points;

    const OutCoord& o = line[0];
    OutCoord n = line[1].normalized();

    points.push_back( o - n*scale );
    points.push_back( o + n*scale );

    vparams->drawTool()->drawLines( points, 1, color );
}


template <class TIn, class TOut>
void ProjectionToLineMultiMapping<TIn, TOut>::updateForceMask()
{
    helper::ReadAccessor< Data<helper::vector<unsigned> > > indices(f_indices);
    if( indices.empty() ) return Inherit1::updateForceMask(); // all dofs are mapped

    // the line is always playing a role
    this->maskFrom[1]->insertEntry( 0 );
    this->maskFrom[1]->insertEntry( 1 );

    for( unsigned i=0 ; i<indices.size() ; i++ )
        if( this->maskTo[0]->getEntry(i) )
            this->maskFrom[0]->insertEntry( indices[i] );
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif

