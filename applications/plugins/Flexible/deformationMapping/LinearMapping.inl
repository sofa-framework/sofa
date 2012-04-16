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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_LINEARMAPPING_INL
#define SOFA_COMPONENT_MAPPING_LINEARMAPPING_INL

#include "LinearMapping.h"
#include "LinearJacobianBlock.inl"
#include <sofa/core/Mapping.inl>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/component/topology/TopologyData.inl>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/core/visual/VisualParams.h>
#include <iostream>
#include <sofa/helper/gl/Color.h>
#include <sofa/helper/vector.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;

using helper::WriteAccessor;
using helper::ReadAccessor;
using helper::vector;

template <class TIn, class TOut>
LinearMapping<TIn, TOut>::LinearMapping (core::State<In>* from, core::State<Out>* to )
    : Inherit ( from, to )
    , assembleJ ( initData ( &assembleJ,false, "assembleJ","Construct the Jacobian matrix or use optimized Jacobian/vector multiplications" ) )
    , ShapeFunction(NULL)
    , f_index ( initData ( &f_index,"indices","parent indices for each child" ) )
    , f_w ( initData ( &f_w,"weights","influence weights of the Dofs" ) )
    , f_dw ( initData ( &f_dw,"weightGradients","weight gradients" ) )
    , f_ddw ( initData ( &f_ddw,"weightHessians","weight Hessians" ) )
    , maskFrom(NULL)
    , maskTo(NULL)
{
}

template <class TIn, class TOut>
LinearMapping<TIn, TOut>::~LinearMapping ()
{
}

template <class TIn, class TOut>
void LinearMapping<TIn, TOut>::init()
{
    if (core::behavior::BaseMechanicalState* stateFrom = dynamic_cast<core::behavior::BaseMechanicalState*>(this->fromModel.get()))
        maskFrom = &stateFrom->forceMask;
    if (core::behavior::BaseMechanicalState* stateTo = dynamic_cast<core::behavior::BaseMechanicalState*>(this->toModel.get()))
        maskTo = &stateTo->forceMask;

    ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
    ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));

    // init shape function
    this->getContext()->get(ShapeFunction,core::objectmodel::BaseContext::SearchUp);
    if ( !ShapeFunction ) serr << "ShapeFunction<"<<ShapeFunctionType::Name()<<"> component not found" << sendl;
    else
    {
        vector<mCoord> initmPos; initmPos.resize(out.size());  for(unsigned int i=0; i<out.size(); i++ )  Out::get(initmPos[i][0],initmPos[i][1],initmPos[i][2],out[i]);
        ShapeFunction->computeShapeFunction(initmPos,*this->f_index.beginEdit(),*this->f_w.beginEdit(),*this->f_dw.beginEdit(),*this->f_ddw.beginEdit());
        this->f_index.endEdit();        this->f_w.endEdit();        this->f_dw.endEdit();        this->f_ddw.endEdit();
    }

    // init jacobians
    jacobian.resize(out.size());
    for(unsigned int i=0; i<out.size(); i++ )
    {
        unsigned int nbref=this->f_index.getValue()[i].size();
        jacobian[i].resize(nbref);
        for(unsigned int j=0; j<nbref; j++ )
        {
            unsigned int index=this->f_index.getValue()[i][j];
            jacobian[i][j].init( in[index],out[i],f_w.getValue()[i][j],f_dw.getValue()[i][j],f_ddw.getValue()[i][j]);
        }
    }
    Inherit::init();
}

template <class TIn, class TOut>
void LinearMapping<TIn, TOut>::reinit()
{
    if(this->assembleJ.getValue()) updateJ();
}


template <class TIn, class TOut>
void LinearMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    OutVecCoord&  out = *dOut.beginEdit();
    const InVecCoord&  in = dIn.getValue();

    for(unsigned int i=0; i<jacobian.size(); i++)
    {
        out[i]=OutCoord();
        for(unsigned int j=0; j<jacobian[i].size(); j++)
        {
            unsigned int index=this->f_index.getValue()[i][j];
            jacobian[i][j].addapply(out[i],in[index]);
        }
    }
    dOut.endEdit();

    if(this->assembleJ.getValue()) if(!Block::constantJ) updateJ();
}

template <class TIn, class TOut>
void LinearMapping<TIn, TOut>::updateJ()
{
    ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
    ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
    eigenJacobian.resizeBlocks(out.size(),in.size());
    for(unsigned int i=0; i<jacobian.size(); i++)
    {
//        for(unsigned int j=0;j<jacobian[i].size();j++)
//        {
//            unsigned int index=this->f_index.getValue()[i][j];
//            eigenJacobian.setBlock( i, index, jacobian[i][j].getJ());
//        }

        // Put all the blocks of the row in an array, then send the array to the matrix
        // Not very efficient: MatBlock creations could be avoided.
        vector<MatBlock> blocks;
        vector<unsigned> columns;
        for(unsigned int j=0; j<jacobian[i].size(); j++)
        {
            columns.push_back( this->f_index.getValue()[i][j] );
            blocks.push_back( jacobian[i][j].getJ() );
        }
        eigenJacobian.appendBlockRow( i, columns, blocks );
    }
    eigenJacobian.endEdit();
}


template <class TIn, class TOut>
void LinearMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{

    if(this->assembleJ.getValue())  eigenJacobian.mult(dOut,dIn);
    else
    {
        OutVecDeriv&  out = *dOut.beginEdit();
        const InVecDeriv&  in = dIn.getValue();

        for(unsigned int i=0; i<jacobian.size(); i++)
        {
            out[i]=OutDeriv();
            for(unsigned int j=0; j<jacobian[i].size(); j++)
            {
                unsigned int index=this->f_index.getValue()[i][j];
                jacobian[i][j].addmult(out[i],in[index]);
            }
        }

        dOut.endEdit();
    }
}

template <class TIn, class TOut>
void LinearMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)
{
    if(this->assembleJ.getValue())  eigenJacobian.addMultTranspose(dIn,dOut);
    else
    {
        InVecDeriv&  in = *dIn.beginEdit();
        const OutVecDeriv&  out = dOut.getValue();

        for(unsigned int i=0; i<jacobian.size(); i++)
        {
            for(unsigned int j=0; j<jacobian[i].size(); j++)
            {
                unsigned int index=this->f_index.getValue()[i][j];
                jacobian[i][j].addMultTranspose(in[index],out[i]);
            }
        }

        dIn.endEdit();
    }
}


template <class TIn, class TOut>
void LinearMapping<TIn, TOut>::applyJT(const core::ConstraintParams */*cparams*/ /* PARAMS FIRST */, Data<InMatrixDeriv>& /*out*/, const Data<OutMatrixDeriv>& /*in*/)
{

}


template <class TIn, class TOut>
const defaulttype::BaseMatrix* LinearMapping<TIn, TOut>::getJ(const core::MechanicalParams */*mparams*/)
{
    if(!this->assembleJ.getValue()) updateJ();
    return &eigenJacobian;
}


template <class TIn, class TOut>
void LinearMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowMechanicalMappings()) return;

    ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
    ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
    ReadAccessor<Data<vector<VRef> > > ref (this->f_index);
    ReadAccessor<Data<vector<VReal> > > w (this->f_w);

    vector< Vec3d > edge;     edge.resize(2);
    Vec<4,float> col;

    for(unsigned i=0; i<ref.size(); i++ )
        for(unsigned j=0; j<ref[i].size(); j++ )
            if(w[i][j])
            {
                In::get(edge[0][0],edge[0][1],edge[0][2],in[ref[i][j]]);
                Out::get(edge[1][0],edge[1][1],edge[1][2],out[i]);
                sofa::helper::gl::Color::getHSVA(&col[0],240.*w[i][j],1.,.8,1.);
                vparams->drawTool()->drawLines ( edge, 1, col );
            }
}



} // namespace mapping

} // namespace component

} // namespace sofa

#endif
