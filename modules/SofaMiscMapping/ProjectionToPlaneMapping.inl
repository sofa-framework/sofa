/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
    , f_indices(initData(&f_indices, "indices", "Indices of the parent points (if empty, all input dofs are mapped)"))
    , f_origins(initData(&f_origins, "origins", "Origins of the planes on which the points are projected"))
    , f_normals(initData(&f_normals, "normals", "Normals of the planes on which the points are projected"))
    , d_drawScale(initData(&d_drawScale, SReal(10), "drawScale", "Draw scale"))
    , d_drawColor(initData(&d_drawColor, defaulttype::Vec4f(1,0,0,0.5), "drawColor", "Draw color"))
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
                    jacobian.insertBack( i*Nout+j, index*Nin+k, 1-n[j]*n[k] );
                else
                    jacobian.insertBack( i*Nout+j, index*Nin+k, -n[j]*n[k] );
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

    for(unsigned i=0; i<nb; i++ )
    {
        const unsigned& index = indices.empty() ? i : indices[i] ;
        const InCoord& p = in[index];
        typename In::CPos x =  TIn::getCPos(p);
        const OutCoord& o = i<origins.size() ? origins[i] : origins.ref().back();
        const OutCoord& n = i<normals.size() ? normals[i] : normals.ref().back();

        out[i] = x - n * ( (x-o) * n ); // projection on the plane
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
    //    cerr<<"ProjectionToTargetPlaneMapping<TIn, TOut>::applyJT is not implemented " << endl;
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

#ifndef SOFA_NO_OPENGL
    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);

    glBegin(GL_QUADS);


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
    glEnd();

    glPopAttrib();
#endif // SOFA_NO_OPENGL
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


} // namespace mapping

} // namespace component

} // namespace sofa

#endif

