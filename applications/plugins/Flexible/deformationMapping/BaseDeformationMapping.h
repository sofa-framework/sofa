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
#ifndef SOFA_COMPONENT_MAPPING_BaseDeformationMAPPING_H
#define SOFA_COMPONENT_MAPPING_BaseDeformationMAPPING_H

#include "../initFlexible.h"
#include <sofa/core/Mapping.h>
#include <sofa/component/component.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/gl/Color.h>
#include <sofa/helper/vector.h>

#include "../shapeFunction/BaseShapeFunction.h"
#include <sofa/component/topology/TopologyData.inl>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/core/visual/VisualParams.h>
#include <iostream>
#include <sofa/helper/gl/Color.h>
#include <sofa/helper/vector.h>
#include <sofa/core/Mapping.inl>

#include <sofa/component/linearsolver/EigenSparseMatrix.h>


namespace sofa
{
namespace component
{
namespace mapping
{

using helper::vector;

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class InDataTypes, class OutDataTypes>
class BaseDeformationMappingInternalData
{
public:
};


/** Abstract mapping (one parent->several children with different influence) using JacobianBlocks or sparse eigen matrix
*/

template <class JacobianBlockType>
class SOFA_Flexible_API BaseDeformationMapping : public core::Mapping<typename JacobianBlockType::In,typename JacobianBlockType::Out>
{
public:
    typedef core::Mapping<typename JacobianBlockType::In, typename JacobianBlockType::Out> Inherit;
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE(BaseDeformationMapping,JacobianBlockType), SOFA_TEMPLATE2(core::Mapping,typename JacobianBlockType::In,typename JacobianBlockType::Out));

    /** @name  Input types    */
    //@{
    typedef typename JacobianBlockType::In In;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Real Real;
    //@}

    /** @name  Output types    */
    //@{
    typedef typename JacobianBlockType::Out Out;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    enum { spatial_dimensions = Out::spatial_dimensions };
    //@}

    /** @name  Shape Function types    */
    //@{
    typedef core::behavior::ShapeFunctionTypes<spatial_dimensions,Real> ShapeFunctionType;
    typedef core::behavior::BaseShapeFunction<ShapeFunctionType> BaseShapeFunction;
    typedef typename BaseShapeFunction::VReal VReal;
    typedef typename BaseShapeFunction::VGradient VGradient;
    typedef typename BaseShapeFunction::VHessian VHessian;
    typedef typename BaseShapeFunction::VRef VRef;
    typedef typename BaseShapeFunction::Coord mCoord; ///< material coordinates
    //@}

    /** @name  Jacobian types    */
    //@{
    typedef JacobianBlockType BlockType;
    typedef vector<vector<BlockType> >  SparseMatrix;

    typedef typename BlockType::MatBlock  MatBlock;  ///< Jacobian block matrix
    typedef linearsolver::EigenSparseMatrix<In,Out>    SparseMatrixEigen;

    typedef typename BlockType::KBlock  KBlock;  ///< stiffness block matrix
    typedef linearsolver::EigenSparseMatrix<In,In>    SparseKMatrixEigen;
    //@}

    /** @name Mapping functions */
    //@{
    virtual void init()
    {
        if (core::behavior::BaseMechanicalState* stateFrom = dynamic_cast<core::behavior::BaseMechanicalState*>(this->fromModel.get()))
            maskFrom = &stateFrom->forceMask;
        if (core::behavior::BaseMechanicalState* stateTo = dynamic_cast<core::behavior::BaseMechanicalState*>(this->toModel.get()))
            maskTo = &stateTo->forceMask;

        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
        helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));

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

        baseMatrices.resize( 1 ); // just a wrapping for getJs()
        baseMatrices[0] = &eigenJacobian;

        reinit();

        Inherit::init();
    }

    virtual void reinit()
    {
        if(this->assembleJ.getValue()) updateJ();

        Inherit::reinit();
    }

    virtual void apply(const core::MechanicalParams */*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
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

        if(!BlockType::constantJ) if(this->assembleJ.getValue()) updateJ();
    }

    virtual void applyJ(const core::MechanicalParams */*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
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


    virtual void applyJT(const core::MechanicalParams */*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)
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

    virtual void applyJT(const core::ConstraintParams */*cparams*/ , Data<InMatrixDeriv>& /*out*/, const Data<OutMatrixDeriv>& /*in*/)
    {

    }

    virtual void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId )
    {
        if(BlockType::constantJ) return;

        Data<InVecDeriv>& parentForceData = *parentDfId[this->fromModel.get(mparams)].write();
        const Data<InVecDeriv>& parentDisplacementData = *mparams->readDx(this->fromModel);
        const Data<OutVecDeriv>& childForceData = *mparams->readF(this->toModel);

        helper::WriteAccessor<Data<InVecDeriv> > parentForce (parentForceData);
        helper::ReadAccessor<Data<InVecDeriv> > parentDisplacement (parentDisplacementData);
        helper::ReadAccessor<Data<OutVecDeriv> > childForce (childForceData);

        if(this->assembleK.getValue())
        {
            updateK(childForce.ref());
            K.addMult(parentForceData,parentDisplacementData,mparams->kFactor());
        }
        else
        {
            for(unsigned int i=0; i<jacobian.size(); i++)
            {
                for(unsigned int j=0; j<jacobian[i].size(); j++)
                {
                    unsigned int index=this->f_index.getValue()[i][j];
                    jacobian[i][j].addDForce( parentForce[index], parentDisplacement[index], childForce[i], mparams->kFactor() );
                }
            }
        }
    }

    const defaulttype::BaseMatrix* getJ(const core::MechanicalParams */*mparams*/)
    {
        if(!this->assembleJ.getValue()) updateJ();
        return &eigenJacobian;
    }

    // Compliant plugin experimental API
    virtual const vector<sofa::defaulttype::BaseMatrix*>* getJs() { return &baseMatrices; }

    void draw(const core::visual::VisualParams* vparams)
    {
        if (!vparams->displayFlags().getShowMechanicalMappings()) return;

        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
        helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
        helper::ReadAccessor<Data<vector<VRef> > > ref (this->f_index);
        helper::ReadAccessor<Data<vector<VReal> > > w (this->f_w);

        vector< defaulttype::Vec3d > edge;     edge.resize(2);
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

    //@}



protected:
    BaseDeformationMapping (core::State<In>* from = NULL, core::State<Out>* to= NULL)
        : Inherit ( from, to )
        , ShapeFunction(NULL)
        , f_index ( initData ( &f_index,"indices","parent indices for each child" ) )
        , f_w ( initData ( &f_w,"weights","influence weights of the Dofs" ) )
        , f_dw ( initData ( &f_dw,"weightGradients","weight gradients" ) )
        , f_ddw ( initData ( &f_ddw,"weightHessians","weight Hessians" ) )
        , maskFrom(NULL)
        , maskTo(NULL)
        , assembleJ ( initData ( &assembleJ,false, "assembleJ","Assemble the Jacobian matrix or use optimized Jacobian/vector multiplications" ) )
        , assembleK ( initData ( &assembleK,false, "assembleK","Assemble the geometric stiffness matrix or use optimized Jacobian/vector multiplications" ) )
    {

    }

    virtual ~BaseDeformationMapping()     { }

    BaseShapeFunction* ShapeFunction;        ///< where the weights are computed
    Data<vector<VRef> > f_index;            ///< The numChildren * numRefs column indices. index[i][j] is the index of the j-th parent influencing child i.
    Data<vector<VReal> >       f_w;
    Data<vector<VGradient> >   f_dw;
    Data<vector<VHessian> >    f_ddw;

    SparseMatrix jacobian;   ///< Jacobian of the mapping

    helper::ParticleMask* maskFrom;  ///< Subset of master DOF, to cull out computations involving null forces or displacements
    helper::ParticleMask* maskTo;    ///< Subset of slave DOF, to cull out computations involving null forces or displacements

    Data<bool> assembleJ;
    SparseMatrixEigen eigenJacobian;  ///< Assembled Jacobian matrix
    vector<defaulttype::BaseMatrix*> baseMatrices;      ///< Vector of jacobian matrices, for the Compliant plugin API
    void updateJ()
    {
        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
        helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
        eigenJacobian.resizeBlocks(out.size(),in.size());
        for(unsigned int i=0; i<jacobian.size(); i++)
        {
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

    Data<bool> assembleK;
    SparseKMatrixEigen K;  ///< Assembled geometric stiffness matrix
    void updateK(const OutVecDeriv& childForce)
    {
        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
        K.resizeBlocks(in.size(),in.size());
        vector<KBlock> diagonalBlocks; diagonalBlocks.resize(in.size());

        for(unsigned int i=0; i<jacobian.size(); i++)
        {
            for(unsigned int j=0; j<jacobian[i].size(); j++)
            {
                unsigned int index=this->f_index.getValue()[i][j];
                diagonalBlocks[index] += jacobian[i][j].getK(childForce[i]);
            }
        }
        for(unsigned int i=0; i<in.size(); i++)
        {
            vector<KBlock> blocks;
            vector<unsigned> columns;
            columns.push_back( i );
            blocks.push_back( diagonalBlocks[i] );
            K.appendBlockRow( i, columns, blocks );
        }
        K.endEdit();
    }
};


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
