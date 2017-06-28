/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
    jacobian.finalize();

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

    const Real factor = d_factor.getValue();

    for(unsigned i=0; i<nb; i++ )
    {
        const unsigned& index = indices.empty() ? i : indices[i] ;
        const InCoord& p = in[index];
        out[i] = p - factor * n * ( ( p - o ) * n ); // projection on the plane

        const defaulttype::Matrix3 Jo = factor * dyad( n, n );
        const defaulttype::Matrix3 Jp = defaulttype::Matrix3::Identity() - Jo;

        const InCoord op = p-o;
        const defaulttype::Matrix3 nop = dyad( n, op );

        defaulttype::Matrix3 Jn;
        Jn[0][0] = -factor * ( nop[1][1] + nop[2][2] + 2*nop[0][0] );
        Jn[0][1] = -factor * ( nop[0][1] );
        Jn[0][2] = -factor * ( nop[0][2] );
        Jn[1][0] = -factor * ( nop[1][0] );
        Jn[1][1] = -factor * ( nop[0][0] + nop[2][2] + 2*nop[1][1] );
        Jn[1][2] = -factor * ( nop[1][2] );
        Jn[2][0] = -factor * ( nop[2][0] );
        Jn[2][1] = -factor * ( nop[2][1] );
        Jn[2][2] = -factor * ( nop[0][0] + nop[1][1] + 2*nop[2][2] );

        for(unsigned j=0; j<Nout; j++)
        {
            size_t row = i*Nout+j;
            jacobian0.beginRow( row );
            jacobian1.beginRow( row );
            for(unsigned k=0; k<Nout; k++ )
            {
                jacobian0.insertBack( row, index*Nin+k, Jp[j][k] ); // dp
                jacobian1.insertBack( row, k, Jo[j][k] ); // do
            }

            for(unsigned k=0; k<Nout; k++ )
                jacobian1.insertBack( row, Nout+k, Jn[j][k] ); // dn
        }
    }

    jacobian0.finalize();
    jacobian1.finalize();

}

template <class TIn, class TOut>
void ProjectionToPlaneMultiMapping<TIn, TOut>::applyJ(const core::MechanicalParams */*mparams*/, const helper::vector<OutDataVecDeriv*>& dataVecOutVel, const helper::vector<const InDataVecDeriv*>& dataVecInVel)
{
    jacobian0.mult(*dataVecOutVel[0],*dataVecInVel[0]);
    jacobian1.addMult(*dataVecOutVel[0],*dataVecInVel[1]);
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
void ProjectionToPlaneMultiMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId)
{
    const SReal& kfactor = mparams->kFactor();

    unsigned size = this->getFromModels().size();

    // TODO not optimized but at least it is implemented

    // merge all parent displacements
    InVecDeriv parentDisplacements;
    for( unsigned i=0; i< size ; i++ )
    {
        const core::State<In>* fromModel = this->getFromModels()[i];
        const InVecDeriv& parentDisplacement = mparams->readDx(fromModel)->getValue();
        parentDisplacements.insert(parentDisplacements.end(), parentDisplacement.begin(), parentDisplacement.end());
    }

    // merged parent forces
    InVecDeriv parentForces(parentDisplacements.size());
    K.addMult( parentForces, parentDisplacements, kfactor );

    // un-merge parent forces
    size_t offset=0;
    for( unsigned i=0; i< size ; i++ )
    {
        core::State<In>* fromModel = this->getFromModels()[i];
        InVecDeriv& parentForce = *parentDfId[fromModel].write()->beginEdit();
        for( size_t j=0;j<parentForce.size();++j)
            parentForce[j] += parentForces[offset+j];
        offset += parentForce.size();
        parentDfId[fromModel].write()->endEdit();
    }
}


template <class TIn, class TOut>
void ProjectionToPlaneMultiMapping<TIn, TOut>::updateK(const core::MechanicalParams* /*mparams*/, core::ConstMultiVecDerivId childForceId )
{
    helper::ReadAccessor<Data<OutVecDeriv> > childForces( *childForceId[(const core::State<TOut>*)this->getToModels()[0]].read() );

    static const core::ConstMultiVecCoordId pos = core::ConstVecCoordId::position();

    const core::State<TIn>* pointState = this->getFromModels()[0];
    helper::ReadAccessor< Data<InVecCoord> > points( *pos[pointState].read() );

    const core::State<TIn>* planeState = this->getFromModels()[1];
    helper::ReadAccessor< Data<InVecCoord> > plane( *pos[planeState].read() );

    const InCoord& o = plane[0];
    const InCoord& n = plane[1];

    helper::ReadAccessor< Data<helper::vector<unsigned> > > indices(f_indices);


    size_t nb = indices.empty() ? this->getFromModels()[0]->getSize() : indices.size(); // if indices is empty, mapping every input dofs

    K.resizeBlocks(this->getFromModels()[0]->getSize()+2, this->getFromModels()[0]->getSize()+2); // all projected point + plane center + plane normal

    const Real factor = d_factor.getValue();



    typedef defaulttype::Mat<Nin,Nin,Real> Block;
    typedef defaulttype::Mat<Nin*Nout,Nin,Real> HessianBlock;

    HessianBlock dodn;
    Block dodnf, d2nf;

    // dout_l/doj.dnk
    for(unsigned l=0; l<Nout; l++) // dout
    {
        for(unsigned j=0; j<Nin; j++) // do
        {
            for(unsigned k=0; k<Nin; k++) // dn
            {
                if( l==j && k==l )
                    dodn[l*Nout+j][k] = factor * 2 * n[l];
                else if( l==k )
                    dodn[l*Nout+j][k] = factor * n[j];
                else if( j==k )
                    dodn[l*Nout+j][k] = factor * n[l];
            }
        }
    }


    for(unsigned i=0; i<nb; i++ )
    {
        const unsigned& index = indices.empty() ? i : indices[i] ;

        const OutDeriv& childForce = childForces[i];

        const InCoord& p = points[index];

        Block dpdnf; // dout_l/dpj.dnk x f

        for(unsigned l=0; l<Nout; l++) // dout
        {
            for(unsigned j=0; j<Nin; j++) // do
            {
                for(unsigned k=0; k<Nin; k++) // dn
                {
                    if( l!=k && j!=k ) continue; // dodn[l*Nout+j][k]==0

                    dpdnf[j][k] -= dodn[l*Nout+j][k] * childForce[l];
                }
            }
        }

        K.addBlock( index, nb+1, dpdnf ); // position index -> current point to project, position nb+1 -> plane normal
        K.addBlock( nb+1, index, dpdnf.transposed() );

        // dout_l/doj.dnk x f  (note that dout_l/doj.dnk == -dout_l/dpj.dnk)
        dodnf -= dpdnf;


        const InCoord op = p-o;
        // d2out/d2n
        for(unsigned l=0; l<Nout; l++) // dout
        {
            for(unsigned j=0; j<Nin; j++) // dn
            {
                for(unsigned k=0; k<Nin; k++) // dn
                {
                    Real d2n;
                    if( l==j && k==l )
                        d2n = 2 * op[l];
                    else if( l==j )
                        d2n = op[k];
                    else if( l==k )
                        d2n = op[j];
                    else
                        continue; // 0

                    d2nf[j][k] -= factor * d2n * childForce[l];
                }
            }
        }

    }

    K.addBlock( nb, nb+1, dodnf ); // position nb -> plane center, position nb+1 -> plane normal
    K.addBlock( nb+1, nb, dodnf.transposed() );

    K.addBlock( nb+1, nb+1, d2nf );

    K.compress();
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

