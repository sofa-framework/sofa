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
#ifndef SOFA_COMPONENT_MAPPING_ProjectionToPlaneMapping_INL
#define SOFA_COMPONENT_MAPPING_ProjectionToPlaneMapping_INL

#include "ProjectionToPlaneMapping.h"
#include <sofa/core/visual/VisualParams.h>
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
    , f_indices(initData(&f_indices, "indices", "Indices of the parent points"))
    , f_origins(initData(&f_origins, "origins", "Origins of the planes on which the points are projected"))
    , f_normals(initData(&f_normals, "normals", "Normals of the planes on which the points are projected"))
{
}

template <class TIn, class TOut>
void ProjectionToTargetPlaneMapping<TIn, TOut>::init()
{
    assert( f_indices.getValue().size()==f_origins.getValue().size()) ;
    assert( f_indices.getValue().size()==f_normals.getValue().size()) ;

    this->getToModel()->resize( f_indices.getValue().size() );

    baseMatrices.resize( 1 );
    baseMatrices[0] = &jacobian;


    // ensuring direction are normalized
    helper::WriteAccessor< Data<OutVecCoord> > normals(f_normals);
    for( size_t i=0 ; i<normals.size() ; ++i )
        normals[i].normalize( OutCoord(1,0,0) ); // failsafe for null norms

    this->Inherit::init();
}


template <class TIn, class TOut>
void ProjectionToTargetPlaneMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteAccessor< Data<OutVecCoord> >  out = dOut;
    helper::ReadAccessor< Data<InVecCoord> >  in = dIn;
    helper::ReadAccessor< Data<vector<unsigned> > > indices(f_indices);
    helper::ReadAccessor< Data<OutVecCoord> > origins(f_origins);
    helper::ReadAccessor< Data<OutVecCoord> > normals(f_normals);

    jacobian.resizeBlocks(out.size(),in.size());

    for(unsigned i=0; i<indices.size(); i++ )
    {
        const InCoord& p = in[indices[i]];
        typename In::CPos x =  TIn::getCPos(p);
        const OutCoord& o = origins[i];
        const OutCoord& n = normals[i];

        out[i] = x - n * ( (x-o) * n ); // projection on the plane

        for(unsigned j=0; j<Nout; j++)
        {
            for(unsigned k=0; k<Nout; k++ )
            {
                if( j == k )
                    jacobian.insertBack( i*Nout+j, indices[i]*Nin+k, 1-n[j]*n[k] );
                else
                    jacobian.insertBack( i*Nout+j, indices[i]*Nin+k, -n[j]*n[k] );
            }
        }
    }
    jacobian.compress();
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
    //    cerr<<"ProjectionToTargetPlaneMapping<TIn, TOut>::applyJT is not implemented " << endl;
}



template <class TIn, class TOut>
const sofa::defaulttype::BaseMatrix* ProjectionToTargetPlaneMapping<TIn, TOut>::getJ()
{
    return &jacobian;
}

template <class TIn, class TOut>
const vector<sofa::defaulttype::BaseMatrix*>* ProjectionToTargetPlaneMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}



template <class TIn, class TOut>
void ProjectionToTargetPlaneMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if( !vparams->displayFlags().getShowMechanicalMappings() ) return;

    helper::ReadAccessor< Data<OutVecCoord> > origins(f_origins);
    helper::ReadAccessor< Data<OutVecCoord> > normals(f_normals);

    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);

    glBegin(GL_QUADS);
    for(unsigned i=0; i<origins.size(); i++ )
    {
        const OutCoord& n = normals[i];
        const OutCoord& o = origins[i];
        OutCoord t0, t1;

        if( !helper::isNull(n[1]) ) t0.set( 0, -n[2], n[1] );
        else if( !helper::isNull(n[2]) ) t0.set( n[2], 0, -n[0] );
        else if( !helper::isNull(n[0]) ) t0.set( n[1], -n[0], 0 );

        t0.normalize();
        t1 = n.cross( t0 );

        vparams->drawTool()->drawQuad( o -t0*10000 -t1*10000, o +t0*10000 -t1*10000, o +t0*10000 +t1*10000, o -t0*10000 +t1*10000, n, defaulttype::Vec4f(1,0,0,1) );

    }
    glEnd();

    glPopAttrib();
}



} // namespace mapping

} // namespace component

} // namespace sofa

#endif

