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
#ifndef SOFA_COMPONENT_MAPPING_ExtensionMapping_INL
#define SOFA_COMPONENT_MAPPING_ExtensionMapping_INL

#include "../deformationMapping/ExtensionMapping.h"
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
ExtensionMapping<TIn, TOut>::ExtensionMapping()
    : Inherit()
    , f_restLengths(initData(&f_restLengths, "restLengths", "Rest lengths of the connections."))
{
}

template <class TIn, class TOut>
ExtensionMapping<TIn, TOut>::~ExtensionMapping()
{
}


template <class TIn, class TOut>
void ExtensionMapping<TIn, TOut>::init()
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
        for(unsigned i=0; i<links.size(); i++ )
        {
            restLengths[i] = (pos[links[i][0]] - pos[links[i][1]]).norm();
        }
    }

    baseMatrices.resize( 1 );
    baseMatrices[0] = &jacobian;

    this->Inherit::init();  // applies the mapping, so after the Data init
}



template <class TIn, class TOut>
void ExtensionMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteAccessor< Data<OutVecCoord> >  out = dOut;
    helper::ReadAccessor< Data<InVecCoord> >  in = dIn;
    helper::WriteAccessor<Data<vector<Real> > > restLengths(f_restLengths);
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

        gap = in[links[i][1]] - in[links[i][0]];
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
        jacobian.beginRow(i);
        if( links[i][1]<links[i][0])
        {
            for(unsigned j=0; j<Nout; j++)
            {
                for(unsigned k=0; k<Nin; k++ )
                {
                    jacobian.set( i*Nout+j, links[i][1]*Nin+k, gap[k] );
                }
                for(unsigned k=0; k<Nin; k++ )
                {
                    jacobian.set( i*Nout+j, links[i][0]*Nin+k, -gap[k] );
                }
            }
        }
        else
        {
            for(unsigned j=0; j<Nout; j++)
            {
                for(unsigned k=0; k<Nin; k++ )
                {
                    jacobian.set( i*Nout+j, links[i][0]*Nin+k, -gap[k] );
                }
                for(unsigned k=0; k<Nin; k++ )
                {
                    jacobian.set( i*Nout+j, links[i][1]*Nin+k, gap[k] );
                }
            }
        }
    }

    jacobian.endEdit();
    //      cerr<<"ExtensionMapping<TIn, TOut>::apply, jacobian: "<<endl<< jacobian << endl;

}

//template <class TIn, class TOut>
//void ExtensionMapping<TIn, TOut>::computeGeometricStiffness(const core::MechanicalParams *mparams)
//{

//}

template <class TIn, class TOut>
void ExtensionMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    if( jacobian.rowSize() > 0 )
        jacobian.mult(dOut,dIn);
}

template <class TIn, class TOut>
void ExtensionMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)
{
    if( jacobian.rowSize() > 0 )
        jacobian.addMultTranspose(dIn,dOut);
}

template <class TIn, class TOut>
void ExtensionMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId )
{
    helper::WriteAccessor<Data<InVecDeriv> > parentForce (*parentDfId[this->fromModel.get(mparams)].write());
    helper::ReadAccessor<Data<InVecDeriv> > parentDisplacement (*mparams->readDx(this->fromModel));  // parent displacement
    Real kfactor = mparams->kFactor();
    helper::ReadAccessor<Data<OutVecDeriv> > childForce (*mparams->readF(this->toModel));
    SeqEdges links = edgeContainer->getEdges();

    for(unsigned i=0; i<links.size(); i++ )
    {
        Mat<Nin,Nin,Real> b;  // = (I - uu^T)
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
        InDeriv df = b*dx;
        parentForce[links[i][0]] -= df;
        parentForce[links[i][1]] += df;
//        cerr<<"ExtensionMapping<TIn, TOut>::applyDJT, df = " << df << endl;
    }
}

template <class TIn, class TOut>
void ExtensionMapping<TIn, TOut>::applyJT(const core::ConstraintParams*, Data<InMatrixDeriv>& , const Data<OutMatrixDeriv>& )
{
    //    cerr<<"ExtensionMapping<TIn, TOut>::applyJT(const core::ConstraintParams*, Data<InMatrixDeriv>& , const Data<OutMatrixDeriv>& ) does nothing " << endl;
}


template <class TIn, class TOut>
const sofa::defaulttype::BaseMatrix* ExtensionMapping<TIn, TOut>::getJ()
{
    return &jacobian;
}

template <class TIn, class TOut>
const vector<sofa::defaulttype::BaseMatrix*>* ExtensionMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}

template <class TIn, class TOut>
void ExtensionMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();
    SeqEdges links = edgeContainer->getEdges();

    vector< Vec3d > points;

    for(unsigned i=0; i<links.size(); i++ )
    {
        points.push_back(pos[links[i][0]]);
        points.push_back(pos[links[i][1]]);
    }
    vparams->drawTool()->drawLines ( points, 1, Vec<4,float> ( 1,1,0,1 ) );
}


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
