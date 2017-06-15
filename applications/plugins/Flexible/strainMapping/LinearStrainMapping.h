/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_MAPPING_LinearStrainMAPPING_H
#define SOFA_COMPONENT_MAPPING_LinearStrainMAPPING_H

#include <Flexible/config.h>

#include <sofa/core/Mapping.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/simulation/Simulation.h>

#include <sofa/core/Mapping.h>

#include <SofaEigen2Solver/EigenSparseMatrix.h>

#include <sofa/helper/IndexOpenMP.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../types/StrainTypes.h"
#include "LinearStrainJacobianBlock.h"


namespace sofa
{
namespace component
{
namespace mapping
{


/**  Map strain positions as a linear combination of strains, for smoothing.
*/

template <class TStrain>
class LinearStrainMapping : public core::Mapping<TStrain,TStrain>
{
public:
    typedef defaulttype::LinearStrainJacobianBlock<TStrain> BlockType;

    typedef core::Mapping<TStrain,TStrain> Inherit;
    SOFA_CLASS(SOFA_TEMPLATE(LinearStrainMapping,TStrain), SOFA_TEMPLATE2(core::Mapping,TStrain,TStrain));

    /** @name  Input types    */
    //@{
    typedef typename TStrain::Coord Coord;
    typedef typename TStrain::Deriv Deriv;
    typedef typename TStrain::VecCoord VecCoord;
    typedef typename TStrain::VecDeriv VecDeriv;
    typedef typename TStrain::MatrixDeriv MatrixDeriv;
    typedef typename TStrain::Real Real;
    //@}

    /** @name  Shape Function types    */
    //@{
    typedef helper::vector<Real> VReal;
    typedef helper::vector< helper::SVector<Real> > VecVReal;
    typedef helper::vector<unsigned int> VRef;
    typedef helper::vector< helper::SVector<unsigned int> > VecVRef;
    //@}

    /** @name  Jacobian types    */
    //@{
    typedef helper::vector<helper::vector<BlockType> >  SparseMatrix;
    typedef linearsolver::EigenSparseMatrix<TStrain,TStrain>    SparseMatrixEigen;
    //@}

    virtual void resizeOut()
    {
        if(this->f_printLog.getValue()) std::cout<<this->getName()<<"::resizeOut()"<<std::endl;

        const VecVRef& indices = this->d_index.getValue();
        const VecVReal& w = this->d_w.getValue();
        this->toModel->resize(indices.size());

        // init jacobian blocks
        jacobian.resize(indices.size());
        for( size_t i=0 ; i<indices.size() ; ++i)
        {
            jacobian[i].resize(indices[i].size());
            for(size_t j=0; j<indices[i].size(); j++)
                jacobian[i][j].init( w[i][j]);
        }

        reinit();
    }

    /** @name Mapping functions */
    //@{
    virtual void init()
    {
        if( core::behavior::BaseMechanicalState* stateFrom = this->fromModel.get()->toBaseMechanicalState() )
            maskFrom = &stateFrom->forceMask;
        if( core::behavior::BaseMechanicalState* stateTo = this->toModel.get()->toBaseMechanicalState() )
            maskTo = &stateTo->forceMask;

        // init jacobians
        baseMatrices.resize( 1 ); // just a wrapping for getJs()
        baseMatrices[0] = &eigenJacobian;

        resizeOut();
        Inherit::init();
    }

    virtual void reinit()
    {
        if(this->d_assemble.getValue()) updateJ();

        Inherit::reinit();
    }


    virtual void apply(const core::MechanicalParams * /*mparams*/ , Data<VecCoord>& dOut, const Data<VecCoord>& dIn)
    {
        if(this->f_printLog.getValue()) std::cout<<this->getName()<<":apply"<<std::endl;

        VecCoord& out = *dOut.beginWriteOnly();
        const VecCoord& in = dIn.getValue();
            const VecVRef& indices = this->d_index.getValue();

#ifdef _OPENMP
#pragma omp parallel for if (this->d_parallel.getValue())
#endif
        for(helper::IndexOpenMP<unsigned int>::type i=0; i<jacobian.size(); i++)
        {
            out[i]=Coord();
            for(size_t j=0; j<jacobian[i].size(); j++)
            {
                size_t index=indices[i][j];
                jacobian[i][j].addapply(out[i],in[index]);
            }
        }
        dOut.endEdit();
    }

    virtual void applyJ(const core::MechanicalParams * /*mparams*/ , Data<VecDeriv>& dOut, const Data<VecDeriv>& dIn)
    {
        if(this->d_assemble.getValue())
        {
            if( !eigenJacobian.rows() ) updateJ();
            eigenJacobian.mult(dOut,dIn);
        }
        else
        {
            VecDeriv& out = *dOut.beginWriteOnly();
            const VecDeriv& in = dIn.getValue();
            const VecVRef& indices = this->d_index.getValue();

#ifdef _OPENMP
#pragma omp parallel for if (this->d_parallel.getValue())
#endif
            //        for( size_t i=0 ; i<this->maskTo->size() ; ++i)
            //            if( !this->maskTo->isActivated() || this->maskTo->getEntry(i) )
            for(helper::IndexOpenMP<unsigned int>::type i=0; i<jacobian.size(); i++)
            {
                out[i]=Deriv();
                for(size_t j=0; j<jacobian[i].size(); j++)
                {
                    size_t index=indices[i][j];
                    jacobian[i][j].addmult(out[i],in[index]);
                }
            }
            dOut.endEdit();
        }
    }


    virtual void applyJT(const core::MechanicalParams * /*mparams*/ , Data<VecDeriv>& dIn, const Data<VecDeriv>& dOut)
    {
        if(this->d_assemble.getValue())
        {
            if( !eigenJacobian.rows() ) updateJ();
            eigenJacobian.addMultTranspose(dIn,dOut);
        }
        else
        {
            VecDeriv& in = *dIn.beginEdit();
            const VecDeriv& out = dOut.getValue();
            const VecVRef& indices = this->d_index.getValue();

//#ifdef _OPENMP
//#pragma omp parallel for if (this->d_parallel.getValue())
//#endif
            //        for( size_t i=0 ; i<this->maskTo->size() ; ++i)
            //            if( this->maskTo->getEntry(i) )
            for(helper::IndexOpenMP<unsigned int>::type i=0; i<jacobian.size(); i++)
            {
                for(size_t j=0; j<jacobian[i].size(); j++)
                {
                    size_t index=indices[i][j];
                    jacobian[i][j].addMultTranspose(in[index],out[i]);
                }
            }

            dIn.endEdit();
        }
    }

    virtual void applyJT(const core::ConstraintParams * /*cparams*/ , Data<MatrixDeriv>& /*out*/, const Data<MatrixDeriv>& /*in*/)
    {

    }

    virtual void applyDJT(const core::MechanicalParams* /*mparams*/, core::MultiVecDerivId /*parentDfId*/, core::ConstMultiVecDerivId /*childForceId*/ )
    {
    }

    const defaulttype::BaseMatrix* getJ(const core::MechanicalParams * /*mparams*/)
    {
        if(!this->d_assemble.getValue())  // J should have been updated in apply() that is call before (when assemble==1)
        {
            updateJ();
            serr<<"Please, with an assembled solver, set assemble=1\n";
        }
        return &eigenJacobian;
    }

    // Compliant plugin API
    virtual const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs()
    {
        if(!this->d_assemble.getValue())  // J should have been updated in apply() that is call before (when assemble==1)
        {
            updateJ();
            serr<<"Please, with an assembled solver, set assemble=1\n";
        }
        return &baseMatrices;
    }



    virtual const defaulttype::BaseMatrix* getK()
    {
        return NULL;
    }


    void draw(const core::visual::VisualParams* /*vparams*/)
    {
    }

    //void updateForceMask()
    //{
    //    const VecVRef& indices = this->d_index.getValue();
    //    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    //    {
    //        if( this->maskTo->getEntry(i) )
    //        {
    //            for(size_t j=0; j<jacobian[i].size(); j++)
    //            {
    //                size_t index = indices[i][j];
    //                this->maskFrom->insertEntry( index );
    //            }
    //        }
    //    }

    //    //    serr<<"updateForceMask "<<this->maskTo->nbActiveDofs()<<" "<<this->maskFrom->nbActiveDofs()<<sendl;
    //}
    //@}

    Data<bool> d_assemble;
    Data< bool > d_parallel;		///< use openmp ?

protected:
    LinearStrainMapping (core::State<TStrain>* from = NULL, core::State<TStrain>* to= NULL)
        : Inherit ( from, to )
        , d_assemble ( initData ( &d_assemble,false, "assemble","Assemble the matrices (Jacobian and Geometric Stiffness) or use optimized matrix/vector multiplications" ) )
        , d_parallel(initData(&d_parallel, false, "parallel", "use openmp parallelisation?"))
        , d_index ( initData ( &d_index,"indices","parent indices for each child" ) )
        , d_w ( initData ( &d_w,"weights","influence weights of the Dofs" ) )
        , maskFrom(NULL)
        , maskTo(NULL)
    {

    }

    virtual ~LinearStrainMapping()     { }

    SparseMatrix jacobian;   ///< Jacobian of the mapping

    Data<VecVRef > d_index;      ///< Store child to parent relationship. index[i][j] is the index of the j-th parent influencing child i.
    Data<VecVReal > d_w;      ///< Influence weights of the parent for each child

    helper::StateMask* maskFrom;  ///< Subset of master DOF, to cull out computations involving null forces or displacements
    helper::StateMask* maskTo;    ///< Subset of slave DOF, to cull out computations involving null forces or displacements

    SparseMatrixEigen eigenJacobian;  ///< Assembled Jacobian matrix
    helper::vector<defaulttype::BaseMatrix*> baseMatrices;      ///< Vector of jacobian matrices, for the Compliant plugin API

    void updateJ()
    {
        unsigned int insize = this->fromModel->getSize();
        //        unsigned int outsize = this->toModel->getSize();

        SparseMatrixEigen& J = eigenJacobian;
        const VecVRef& indices = this->d_index.getValue();

        J.resizeBlocks(jacobian.size(),insize);

        for( size_t i=0 ; i<this->maskTo->size() ; ++i)
        {
            J.beginBlockRow(i);
            for(size_t j=0; j<jacobian[i].size(); j++)
                J.createBlock( indices[i][j], jacobian[i][j].getJ());
            J.endBlockRow();
        }

        J.compress();
    }





};



} // namespace mapping

} // namespace component

} // namespace sofa


#endif
