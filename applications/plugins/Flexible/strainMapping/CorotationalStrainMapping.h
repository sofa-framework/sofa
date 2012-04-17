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
#ifndef SOFA_COMPONENT_MAPPING_CorotationalStrainMAPPING_H
#define SOFA_COMPONENT_MAPPING_CorotationalStrainMAPPING_H

#include "../initFlexible.h"
#include <sofa/core/Mapping.h>
#include <sofa/component/component.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/simulation/common/Simulation.h>

#include "../strainMapping/CorotationalStrainJacobianBlock.inl"
#include <sofa/component/container/MechanicalObject.inl>
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
class CorotationalStrainMappingInternalData
{
public:
};


/** Deformation Gradient to Corotational Lagrangian Strain mapping
*/

template <class TIn, class TOut>
class SOFA_Flexible_API CorotationalStrainMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CorotationalStrainMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;

    /** @name  Input types    */
    //@{
    typedef TIn In;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Real Real;
    //@}

    /** @name  Output types    */
    //@{
    typedef TOut Out;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    enum { spatial_dimensions = Out::spatial_dimensions };
    //@}

    //@name quadrature method
    /**@{*/
    typedef enum { POLAR, QR, SMALL } RotationDecompositionMethod;
    Data<std::string> f_method;
    RotationDecompositionMethod decompositionMethod;
    /**@}*/


    /** @name  Jacobian types    */
    //@{
    typedef defaulttype::CorotationalStrainJacobianBlock<In,Out>  Block;  ///< Jacobian block object
    typedef vector<Block>  SparseMatrix;

    typedef typename Block::MatBlock  MatBlock;  ///< Jacobian block matrix
    typedef linearsolver::EigenSparseMatrix<In,Out>    SparseMatrixEigen;
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

        // resize out
        this->toModel->resize(in.size());

        // init jacobians
        jacobian.resize(in.size());

        reinit();

        Inherit::init();
    }

    virtual void reinit()
    {
        if (f_method.getValue() == "small") decompositionMethod= SMALL;
        else if (f_method.getValue() == "large") decompositionMethod= QR;
        else decompositionMethod= POLAR;

        for(unsigned int i=0; i<jacobian.size(); i++) jacobian[i].decompositionMethod=decompositionMethod;
        if(this->assembleJ.getValue()) updateJ();
    }

    virtual void apply(const core::MechanicalParams */*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
    {
        OutVecCoord&  out = *dOut.beginEdit();
        const InVecCoord&  in = dIn.getValue();

        for(unsigned int i=0; i<jacobian.size(); i++)
        {
            out[i]=OutCoord();
            jacobian[i].addapply(out[i],in[i]);
        }
        dOut.endEdit();

        if(this->assembleJ.getValue()) updateJ(); // this must be updated because the mapping is non-linear
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
                jacobian[i].addmult(out[i],in[i]);
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
                jacobian[i].addMultTranspose(in[i],out[i]);
            }

            dIn.endEdit();
        }
    }

    virtual void applyJT(const core::ConstraintParams */*cparams*/ , Data<InMatrixDeriv>& /*out*/, const Data<OutMatrixDeriv>& /*in*/)
    {

    }

    //    virtual void applyDJT(const core::MechanicalParams* mparams /* PARAMS FIRST  = core::MechanicalParams::defaultInstance()*/, core::MultiVecDerivId parentForce, core::ConstMultiVecDerivId  childForce );
    const defaulttype::BaseMatrix* getJ(const core::MechanicalParams */*mparams*/)
    {
        if(!this->assembleJ.getValue()) updateJ();
        return &eigenJacobian;
    }

    void draw(const core::visual::VisualParams* /*vparams*/)
    {
    }

    //@}



protected:
    CorotationalStrainMapping (core::State<In>* from = NULL, core::State<Out>* to= NULL)
        : Inherit ( from, to )
        , f_method(initData(&f_method,std::string("polar"),"method","\"large\" (by QR), \"polar\" or \"small\" displacements"))
        , assembleJ ( initData ( &assembleJ,false, "assembleJ","Construct the Jacobian matrix or use optimized Jacobian/vector multiplications" ) )
        , maskFrom(NULL)
        , maskTo(NULL)
    {
    }

    virtual ~CorotationalStrainMapping()     { }

    Data<bool> assembleJ;
    void updateJ()
    {
        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
        helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
        eigenJacobian.resizeBlocks(out.size(),in.size());
        for(unsigned int i=0; i<jacobian.size(); i++)
        {
            //        eigenJacobian.setBlock( i, i, jacobian[i].getJ());

            // Put all the blocks of the row in an array, then send the array to the matrix
            // Not very efficient: MatBlock creations could be avoided.
            vector<MatBlock> blocks;
            vector<unsigned> columns;
            columns.push_back( i );
            blocks.push_back( jacobian[i].getJ() );
            eigenJacobian.appendBlockRow( i, columns, blocks );
        }
        eigenJacobian.endEdit();
    }

    SparseMatrix jacobian;   ///< Jacobian of the mapping
    SparseMatrixEigen eigenJacobian;  ///< Assembled Jacobian matrix

    helper::ParticleMask* maskFrom;  ///< Subset of master DOF, to cull out computations involving null forces or displacements
    helper::ParticleMask* maskTo;    ///< Subset of slave DOF, to cull out computations involving null forces or displacements

};


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
