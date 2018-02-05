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
#ifndef SOFA_COMPONENT_MAPPING_ProjectionToPlaneMapping_INL
#define SOFA_COMPONENT_MAPPING_ProjectionToPlaneMapping_INL

#include "ProjectionToPlaneMapping.h"
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
ProjectionToTargetPlaneMapping<TIn, TOut>::ProjectionToTargetPlaneMapping()
    : Inherit()
    , f_indices(initData(&f_indices, "indices", "Indices of the parent points (if empty, all input dofs are mapped)"))
    , f_origins(initData(&f_origins, "origins", "Origins of the planes on which the points are projected"))
    , f_normals(initData(&f_normals, "normals", "Normals of the planes on which the points are projected"))
    , d_factor(initData(&d_factor, Real(1), "factor", "Projection factor (0->nothing, 1->projection on the plane (default), 2->planar symmetry, ..."))
    , d_drawScale(initData(&d_drawScale, SReal(10), "drawScale", "Draw scale"))
    , d_drawColor(initData(&d_drawColor, defaulttype::RGBAColor(1.0f,0.0f,0.0f,0.5f), "drawColor", "Draw color. (default=[1.0,0.0,0.0,0.5]))"))
{
    d_drawScale.setGroup("Visualization");
    d_drawColor.setGroup("Visualization");
}

template <class TIn, class TOut>
void ProjectionToTargetPlaneMapping<TIn, TOut>::init()
{
    baseMatrices.resize( 1 );
    baseMatrices[0] = &jacobian;

    reinit();
    this->Inherit::init();
}

template <class TIn, class TOut>
void ProjectionToTargetPlaneMapping<TIn, TOut>::reinit()
{
    helper::ReadAccessor< Data<helper::vector<unsigned> > > indices(f_indices);
    size_t nb = indices.empty() ? this->getFromModel()->getSize() : indices.size(); // if indices is empty, mapping every input dofs

    this->getToModel()->resize( nb );

    // ensuring direction are normalized
    helper::WriteAccessor< Data<OutVecCoord> > normals(f_normals);
    for( size_t i=0 ; i<normals.size() ; ++i )
        normals[i].normalize( OutCoord(1,0,0) ); // failsafe for null norms


    Real factor = d_factor.getValue();
    // precompute constant jacobian
    jacobian.resizeBlocks(nb,this->getFromModel()->getSize());
    for(unsigned i=0; i<nb; i++ )
    {
        const OutCoord& n = i<normals.size() ? normals[i] : normals.ref().back();
        const unsigned& index = indices.empty() ? i : indices[i] ;

        for(unsigned j=0; j<Nout; j++)
        {
            jacobian.beginRow(i*Nout+j);
            for(unsigned k=0; k<Nout; k++ )
            {
                if( j == k )
                    jacobian.insertBack( i*Nout+j, index*Nin+k, 1-factor*n[j]*n[k] );
                else
                    jacobian.insertBack( i*Nout+j, index*Nin+k, -factor*n[j]*n[k] );
            }
        }
    }
    jacobian.compress();

    this->Inherit::reinit();
}

template <class TIn, class TOut>
void ProjectionToTargetPlaneMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteAccessor< Data<OutVecCoord> >  out = dOut;
    helper::ReadAccessor< Data<InVecCoord> >  in = dIn;
    helper::ReadAccessor< Data<helper::vector<unsigned> > > indices(f_indices);
    helper::ReadAccessor< Data<OutVecCoord> > origins(f_origins);
    helper::ReadAccessor< Data<OutVecCoord> > normals(f_normals);

    size_t nb = indices.empty() ? this->getFromModel()->getSize() : indices.size(); // if indices is empty, mapping every input dofs

    Real factor = d_factor.getValue();

    for(unsigned i=0; i<nb; i++ )
    {
        const unsigned& index = indices.empty() ? i : indices[i] ;
        const InCoord& p = in[index];
        typename In::CPos x =  TIn::getCPos(p);
        const OutCoord& o = i<origins.size() ? origins[i] : origins.ref().back();
        const OutCoord& n = i<normals.size() ? normals[i] : normals.ref().back();

        out[i] = x - factor * n * ( (x-o) * n ); // projection on the plane
    }
}



template <class TIn, class TOut>
void ProjectionToTargetPlaneMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    jacobian.mult(dOut,dIn);
}

template <class TIn, class TOut>
void ProjectionToTargetPlaneMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)
{
    jacobian.addMultTranspose(dIn,dOut);
}

template <class TIn, class TOut>
void ProjectionToTargetPlaneMapping<TIn, TOut>::applyJT(const core::ConstraintParams*, Data<InMatrixDeriv>& , const Data<OutMatrixDeriv>& )
{
}



template <class TIn, class TOut>
const sofa::defaulttype::BaseMatrix* ProjectionToTargetPlaneMapping<TIn, TOut>::getJ()
{
    return &jacobian;
}

template <class TIn, class TOut>
const helper::vector<sofa::defaulttype::BaseMatrix*>* ProjectionToTargetPlaneMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}



template <class TIn, class TOut>
void ProjectionToTargetPlaneMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if( !vparams->displayFlags().getShowMechanicalMappings() ) return;

    const SReal& scale = d_drawScale.getValue();
    if(!scale) return;

    const defaulttype::Vec4f color = d_drawColor.getValue();


    helper::ReadAccessor< Data<OutVecCoord> > origins(f_origins);
    helper::ReadAccessor< Data<OutVecCoord> > normals(f_normals);

    vparams->drawTool()->saveLastState();
    vparams->drawTool()->setLightingEnabled(false);

    size_t nb = std::max( normals.size(), origins.size() );
    for(unsigned i=0; i<nb; i++ )
    {
        const OutCoord& n = i<normals.size() ? normals[i] : normals.ref().back();
        const OutCoord& o = i<origins.size() ? origins[i] : origins.ref().back();
        OutCoord t0, t1;

        if( !helper::isNull(n[1]) ) t0.set( 0, -n[2], n[1] );
        else if( !helper::isNull(n[2]) ) t0.set( n[2], 0, -n[0] );
        else if( !helper::isNull(n[0]) ) t0.set( n[1], -n[0], 0 );

        t0.normalize();
        t1 = n.cross( t0 );

        vparams->drawTool()->drawQuad( o -t0*scale -t1*scale, o +t0*scale -t1*scale, o +t0*scale +t1*scale, o -t0*scale +t1*scale, n, color );

    }

    vparams->drawTool()->restoreLastState();
}


template <class TIn, class TOut>
void ProjectionToTargetPlaneMapping<TIn, TOut>::updateForceMask()
{
    helper::ReadAccessor< Data<helper::vector<unsigned> > > indices(f_indices);
    if( indices.empty() ) return Inherit::updateForceMask(); // all dofs are mapped

    for( size_t i=0 ; i<indices.size() ; i++ )
        if( this->maskTo->getEntry(i) )
            this->maskFrom->insertEntry( indices[i] );
}


//////////////////


template <class TIn, class TOut>
ProjectionToPlaneMultiMapping<TIn, TOut>::ProjectionToPlaneMultiMapping()
    : Inherit1()
    , f_indices(initData(&f_indices, "indices", "Indices of the parent points (if empty, all input dofs are mapped)"))
    , d_factor(initData(&d_factor, Real(1), "factor", "Projection factor (0->nothing, 1->projection on the plane (default), 2->planar symmetry, ..."))
    , d_drawScale(initData(&d_drawScale, SReal(10), "drawScale", "Draw scale"))
    , d_drawColor(initData(&d_drawColor, defaulttype::RGBAColor(0,1,0,1), "drawColor", "Draw color. (default=[0.0,1.0,0.0,1.0])"))
{
}

template <class TIn, class TOut>
void ProjectionToPlaneMultiMapping<TIn, TOut>::init()
{
    baseMatrices.resize( 2 );
    baseMatrices[0] = &jacobian0;
    baseMatrices[1] = &jacobian1;

    reinit();
    this->Inherit1::init();
}


template <class TIn, class TOut>
void ProjectionToPlaneMultiMapping<TIn, TOut>::reinit()
{
    helper::ReadAccessor< Data<helper::vector<unsigned> > > indices(f_indices);

    size_t nb = indices.empty() ? this->getFromModels()[0]->getSize() : indices.size(); // if indices is empty, mapping every input dofs

    this->getToModels()[0]->resize( nb );

    assert(this->getFromModels()[1]->getSize()==2); // center + normal
    jacobian1.resizeBlocks(nb,this->getFromModels()[1]->getSize());

    this->Inherit1::reinit();
}


template <class TIn, class TOut>
void ProjectionToPlaneMultiMapping<TIn, TOut>::apply(const core::MechanicalParams */*mparams*/, const helper::vector<OutDataVecCoord*>& dataVecOutPos, const helper::vector<const InDataVecCoord*>& dataVecInPos)
{
    helper::WriteAccessor< Data<OutVecCoord> >  out = *dataVecOutPos[0];
    helper::ReadAccessor< Data<InVecCoord> >  in = *dataVecInPos[0];
    helper::ReadAccessor< Data<InVecCoord> >  plane = *dataVecInPos[1];
    helper::ReadAccessor< Data<helper::vector<unsigned> > > indices(f_indices);

    size_t nb = indices.empty() ? this->getFromModels()[0]->getSize() : indices.size(); // if indices is empty, mapping every input dofs

    jacobian0.resizeBlocks(nb,this->getFromModels()[0]->getSize());
    jacobian1.compressedMatrix.setZero();

    const OutCoord& o = plane[0];
    const OutCoord& n = plane[1];
    OutCoord nn = n;
    Real n_norm2 = nn.norm2();

    assert( n_norm2 > std::numeric_limits<Real>::epsilon() );

    Real n_norm = std::sqrt( n_norm2 );
    nn.normalizeWithNorm( n_norm );

    Real factor = d_factor.getValue();


    for(unsigned i=0; i<nb; i++ )
    {
        const unsigned& index = indices.empty() ? i : indices[i] ;
        const InCoord& p = in[index];
        out[i] = p - factor * nn * ( ( p - o ) * nn ); // projection on the plane

        defaulttype::Matrix3 Jo = factor * dyad( nn, nn );
        defaulttype::Matrix3 Jp = defaulttype::Matrix3::Identity() - Jo;
        defaulttype::Matrix3 Jd;

        // generated by maple
        // because of the normal normalization that complexifies
        // TODO this code is improvable
        const Real& n0 = n[0];
        const Real& n1 = n[1];
        const Real& n2 = n[2];
        const Real& p0 = p[0];
        const Real& p1 = p[1];
        const Real& p2 = p[2];
        const Real& o0 = o[0];
        const Real& o1 = o[1];
        const Real& o2 = o[2];
        Real t1 = Jp[2][2];//n2 * n2;
        Real t2 = Jp[1][1];//n1 * n1;
        Real t3 = Jp[0][0];//n0 * n0;
        Real t4 = n_norm2; //t1 + t2 + t3; // squared norm
        Real t5 = 1./(n_norm2 * n_norm);//pow(t4, -0.3e1 / 0.2e1); // 1/(squared norm * norm)
        t4 = t4 * t5;
        Real t6 = -p0 + o0;
        Real t7 = p1 - o1;
        Real t8 = p2 - o2;
        Real t9 = n1 * t7;
        Real t10 = n2 * t8;
        t2 = t2 * t5;
        Real t11 = n0 * t6;
        Real t12 = (t11 - t9 - t10) * t4;
        t6 = (-t2 * t6 + t4 * t6 + (t9 + t10) * t5 * n0) * t4;
        t3 = t3 * t5;
        Real t13 = t12 * t5;
        t7 = (-t3 * t7 + t4 * t7 + (t11 - t10) * t5 * n1) * t4;
        t10 = t13 * n1 + t7;
        Real t14 = factor * n0;
        t1 = t1 * t5;
        t5 = (-t1 * t8 + t4 * t8 + (t11 - t9) * t5 * n2) * t4;
        t8 = t13 * n2 + t5;
        t9 = t13 * n0 - t6;
        t11 = factor * n1;
        t13 = factor * n2;
        Jd[0][0] = factor * (t12 * (-t2 + t4) + n0 * t6);
        Jd[0][1] = -t14 * t10;
        Jd[0][2] = -t14 * t8;
        Jd[1][0] = -t11 * t9;
        Jd[1][1] = -factor * (t12 * (t3 - t4) + n1 * t7);
        Jd[1][2] = -t11 * t8;
        Jd[2][0] = -t13 * t9;
        Jd[2][1] = -t13 * t10;
        Jd[2][2] = -factor * (t12 * (t1 - t4) + t5 * n2);

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
void ProjectionToPlaneMultiMapping<TIn, TOut>::applyJ(const core::MechanicalParams */*mparams*/, const helper::vector<OutDataVecDeriv*>& dataVecOutVel, const helper::vector<const InDataVecDeriv*>& dataVecInVel)
{
    jacobian0.mult(*dataVecOutVel[0],*dataVecInVel[0]);
    jacobian1.mult(*dataVecOutVel[0],*dataVecInVel[1]);
}

template <class TIn, class TOut>
void ProjectionToPlaneMultiMapping<TIn, TOut>::applyJT(const core::MechanicalParams */*mparams*/, const helper::vector<InDataVecDeriv*>& dataVecOutForce, const helper::vector<const OutDataVecDeriv*>& dataVecInForce)
{
    jacobian0.addMultTranspose(*dataVecOutForce[0],*dataVecInForce[0]);
    jacobian1.addMultTranspose(*dataVecOutForce[1],*dataVecInForce[0]);
}



template <class TIn, class TOut>
const helper::vector<sofa::defaulttype::BaseMatrix*>* ProjectionToPlaneMultiMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}



template <class TIn, class TOut>
void ProjectionToPlaneMultiMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if( !vparams->displayFlags().getShowMechanicalMappings() ) return;

    const SReal& scale = d_drawScale.getValue();
    if(!scale) return;

    const defaulttype::Vec4f color = d_drawColor.getValue();

    helper::ReadAccessor<Data<InVecCoord> > plane = this->getFromModels()[1]->read(core::ConstVecCoordId::position());


    const OutCoord& o = plane[0];
    OutCoord n = plane[1].normalized();

    vparams->drawTool()->saveLastState();
    vparams->drawTool()->setLightingEnabled(false);

    OutCoord t0, t1;

    if( !helper::isNull(n[1]) ) t0.set( 0, -n[2], n[1] );
    else if( !helper::isNull(n[2]) ) t0.set( n[2], 0, -n[0] );
    else if( !helper::isNull(n[0]) ) t0.set( n[1], -n[0], 0 );

    t0.normalize();
    t1 = n.cross( t0 );

    vparams->drawTool()->drawQuad( o -t0*scale -t1*scale, o +t0*scale -t1*scale, o +t0*scale +t1*scale, o -t0*scale +t1*scale, n, color );

    vparams->drawTool()->restoreLastState();

    // normal
    helper::vector< defaulttype::Vector3 > points;
    points.push_back( o - n*scale );
    points.push_back( o + n*scale );
    vparams->drawTool()->drawLines( points, 1, color );
}


template <class TIn, class TOut>
void ProjectionToPlaneMultiMapping<TIn, TOut>::updateForceMask()
{
    helper::ReadAccessor< Data<helper::vector<unsigned> > > indices(f_indices);
    if( indices.empty() ) return Inherit1::updateForceMask(); // all dofs are mapped

    // the plane is always playing a role
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

