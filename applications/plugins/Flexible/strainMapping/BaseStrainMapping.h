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
#ifndef SOFA_COMPONENT_MAPPING_BaseStrainMAPPING_H
#define SOFA_COMPONENT_MAPPING_BaseStrainMAPPING_H

#include "../initFlexible.h"
#include <sofa/core/Mapping.h>
#include <sofa/component/component.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/simulation/common/Simulation.h>

//#include <SofaBaseMechanics/MechanicalObject.inl>
#include <sofa/core/Mapping.inl>

#include <SofaEigen2Solver/EigenSparseMatrix.h>


namespace sofa
{
namespace component
{
namespace mapping
{

using helper::vector;

/** Abstract interface to allow for resizing
*/
class SOFA_Flexible_API BaseStrainMapping : public virtual core::objectmodel::BaseObject
{
public:
    virtual void resizeOut()=0;
    virtual void applyJT()=0;
};



/** Abstract one-to-one mapping (one parent->one child) using JacobianBlocks or sparse eigen matrix
*/

template <class JacobianBlockType>
class BaseStrainMappingT : public core::Mapping<typename JacobianBlockType::In,typename JacobianBlockType::Out>, public BaseStrainMapping
{
public:
    typedef core::Mapping<typename JacobianBlockType::In, typename JacobianBlockType::Out> Inherit;
    SOFA_ABSTRACT_CLASS2(SOFA_TEMPLATE(BaseStrainMappingT,JacobianBlockType), SOFA_TEMPLATE2(core::Mapping,typename JacobianBlockType::In,typename JacobianBlockType::Out),BaseStrainMapping);

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
    //    enum { spatial_dimensions = Out::spatial_dimensions };
    //@}

    /** @name  Jacobian types    */
    //@{
    typedef JacobianBlockType BlockType;
    typedef vector<BlockType>  SparseMatrix;

    typedef typename BlockType::MatBlock  MatBlock;  ///< Jacobian block matrix
    typedef linearsolver::EigenSparseMatrix<In,Out>    SparseMatrixEigen;

    typedef typename BlockType::KBlock  KBlock;  ///< stiffness block matrix
    typedef linearsolver::EigenSparseMatrix<In,In>    SparseKMatrixEigen;
    //@}


    virtual void resizeOut()
    {
        if(this->f_printLog.getValue()) std::cout<<this->getName()<<"::resizeOut()"<<std::endl;

        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
        this->toModel->resize(in.size());

        jacobian.resize(in.size());

        reinit();
    }


    /** @name Mapping functions */
    //@{
    virtual void init()
    {
        if (core::behavior::BaseMechanicalState* stateFrom = dynamic_cast<core::behavior::BaseMechanicalState*>(this->fromModel.get()))
            maskFrom = &stateFrom->forceMask;
        if (core::behavior::BaseMechanicalState* stateTo = dynamic_cast<core::behavior::BaseMechanicalState*>(this->toModel.get()))
            maskTo = &stateTo->forceMask;

        // init jacobians
        baseMatrices.resize( 1 ); // just a wrapping for getJs()
        baseMatrices[0] = &eigenJacobian;

        // init geometric stiffnesses
        stiffnessBaseMatrices.resize(1);
        stiffnessBaseMatrices[0] = &K;

        resizeOut();
        Inherit::init();
    }

    virtual void reinit()
    {
        if(this->assemble.getValue()) updateJ();

        // clear forces
        helper::WriteAccessor<Data< OutVecDeriv > >  f(*this->toModel->write(core::VecDerivId::force())); for(unsigned int i=0;i<f.size();i++) f[i].clear();

        apply(NULL, *this->toModel->write(core::VecCoordId::position()), *this->fromModel->read(core::ConstVecCoordId::position()));
        applyJ(NULL, *this->toModel->write(core::VecDerivId::velocity()), *this->fromModel->read(core::ConstVecDerivId::velocity()));

        Inherit::reinit();
    }

    virtual void applyJT()
    {
        applyJT(NULL, *this->fromModel->write(core::VecDerivId::force()), *this->toModel->read(core::ConstVecDerivId::force()));
       //TODO applyDJT(NULL, *this->fromModel->write(core::VecDerivId::force()), *this->toModel->read(core::ConstVecDerivId::force()));
    }

    virtual void apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
    {
        if(this->f_printLog.getValue()) std::cout<<this->getName()<<":apply"<<std::endl;

        helper::ReadAccessor<Data<InVecCoord> > inpos (*this->fromModel->read(core::ConstVecCoordId::position()));
        helper::ReadAccessor<Data<OutVecCoord> > outpos (*this->toModel->read(core::ConstVecCoordId::position()));
        if(inpos.size()!=outpos.size()) this->resizeOut();

        OutVecCoord&  out = *dOut.beginEdit();
        const InVecCoord&  in = dIn.getValue();

#ifdef USING_OMP_PRAGMAS
        #pragma omp parallel for
#endif
        for(int i=0; i < static_cast<int>(jacobian.size()); i++)
        {
            out[i]=OutCoord();
            jacobian[i].addapply(out[i],in[i]);
        }
        dOut.endEdit();

        if(!BlockType::constant) if(this->assemble.getValue()) updateJ();
    }

    virtual void applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
    {
        if(this->assemble.getValue())  eigenJacobian.mult(dOut,dIn);
        else
        {
            OutVecDeriv&  out = *dOut.beginEdit();
            const InVecDeriv&  in = dIn.getValue();

#ifdef USING_OMP_PRAGMAS
        #pragma omp parallel for
#endif
            for(int i=0; i < static_cast<int>(jacobian.size()); i++)
            {
                out[i]=OutDeriv();
                jacobian[i].addmult(out[i],in[i]);
            }
            dOut.endEdit();
        }
    }


    virtual void applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)
    {
        if(this->assemble.getValue())  eigenJacobian.addMultTranspose(dIn,dOut);
        else
        {
            InVecDeriv&  in = *dIn.beginEdit();
            const OutVecDeriv&  out = dOut.getValue();

#ifdef USING_OMP_PRAGMAS
        #pragma omp parallel for
#endif
            for(int i=0; i < static_cast<int>(jacobian.size()); i++)
            {
                jacobian[i].addMultTranspose(in[i],out[i]);
            }

            dIn.endEdit();
        }
    }

    virtual void applyJT(const core::ConstraintParams * /*cparams*/ , Data<InMatrixDeriv>& /*out*/, const Data<OutMatrixDeriv>& /*in*/)
    {

    }


    virtual void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId )
    {
        if(BlockType::constant) return;

        Data<InVecDeriv>& parentForceData = *parentDfId[this->fromModel.get(mparams)].write();
        const Data<InVecDeriv>& parentDisplacementData = *mparams->readDx(this->fromModel);
        const Data<OutVecDeriv>& childForceData = *mparams->readF(this->toModel);

        helper::WriteAccessor<Data<InVecDeriv> > parentForce (parentForceData);
        helper::ReadAccessor<Data<InVecDeriv> > parentDisplacement (parentDisplacementData);
        helper::ReadAccessor<Data<OutVecDeriv> > childForce (childForceData);

//        cerr<<"BaseStrainMapping::applyDJT, parentForce before = " << parentForce << endl;

        if(this->assemble.getValue())
        {
            updateK(childForce.ref());
            K.addMult(parentForceData,parentDisplacementData,mparams->kFactor());
        }
        else
        {
#ifdef USING_OMP_PRAGMAS
			#pragma omp parallel for
#endif
            for(int i=0; i < static_cast<int>(jacobian.size()); i++)
            {
                jacobian[i].addDForce( parentForce[i], parentDisplacement[i], childForce[i], mparams->kFactor() );
            }
        }
//        cerr<<"BaseStrainMapping::applyDJT, parentForce after = " << parentForce << endl;
    }

    const defaulttype::BaseMatrix* getJ(const core::MechanicalParams * /*mparams*/)
    {
        if(!this->assemble.getValue()/* || !BlockType::constant*/)  // J should have been updated in apply() that is call before (when assemble==1)
        {
            updateJ();
            serr<<"Please, with an assembled solver, set assemble=1\n";
        }
        return &eigenJacobian;
    }

    // Compliant plugin experimental API
    virtual const vector<sofa::defaulttype::BaseMatrix*>* getJs()
    {
        if(!this->assemble.getValue()/* || !BlockType::constant*/)  // J should have been updated in apply() that is call before (when assemble==1)
        {
            updateJ();
            serr<<"Please, with an assembled solver, set assemble=1\n";
        }
        return &baseMatrices;
    }


    void draw(const core::visual::VisualParams* /*vparams*/)
    {
    }

    //@}



    Data<bool> assemble;


protected:
    BaseStrainMappingT (core::State<In>* from = NULL, core::State<Out>* to= NULL)
        : Inherit ( from, to )
        , assemble ( initData ( &assemble,false, "assemble","Assemble the matrices (Jacobian and Geometric Stiffness) or use optimized matrix/vector multiplications" ) )
        , maskFrom(NULL)
        , maskTo(NULL)
    {

    }

    virtual ~BaseStrainMappingT()     { }

    SparseMatrix jacobian;   ///< Jacobian of the mapping

    helper::ParticleMask* maskFrom;  ///< Subset of master DOF, to cull out computations involving null forces or displacements
    helper::ParticleMask* maskTo;    ///< Subset of slave DOF, to cull out computations involving null forces or displacements

    SparseMatrixEigen eigenJacobian;  ///< Assembled Jacobian matrix
    vector<defaulttype::BaseMatrix*> baseMatrices;      ///< Vector of jacobian matrices, for the Compliant plugin API
    void updateJ()
    {
        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
        helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
        eigenJacobian.resizeBlocks(out.size(),in.size());
        for(size_t i=0; i<jacobian.size(); i++)
        {
//            vector<MatBlock> blocks;
//            vector<unsigned> columns;
//            columns.push_back( i );
//            blocks.push_back( jacobian[i].getJ() );
//            eigenJacobian.appendBlockRow( i, columns, blocks );
            eigenJacobian.beginBlockRow(i);
            eigenJacobian.createBlock(i,jacobian[i].getJ());
            eigenJacobian.endBlockRow();
        }
//        eigenJacobian.endEdit();
        eigenJacobian.compress();
    }

    SparseKMatrixEigen K;  ///< Assembled geometric stiffness matrix
    vector<defaulttype::BaseMatrix*> stiffnessBaseMatrices;      ///< Vector of geometric stiffness matrices, for the Compliant plugin API
    void updateK(const OutVecDeriv& childForce)
    {
        helper::ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
        K.resizeBlocks(in.size(),in.size());
        for(size_t i=0; i<jacobian.size(); i++)
        {
//            vector<KBlock> blocks;
//            vector<unsigned> columns;
//            columns.push_back( i );
//            blocks.push_back( jacobian[i].getK(childForce[i]) );
//            K.appendBlockRow( i, columns, blocks );
            K.beginBlockRow(i);
            K.createBlock(i,jacobian[i].getK(childForce[i]));
            K.endBlockRow();
        }
//        K.endEdit();
        K.compress();
    }
    virtual const vector<defaulttype::BaseMatrix*>* getKs()
    {
        updateK(this->toModel->readForces().ref());
        return &stiffnessBaseMatrices;
    }
};



} // namespace mapping

} // namespace component

} // namespace sofa

#endif
