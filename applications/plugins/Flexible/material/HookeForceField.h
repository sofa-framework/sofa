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
#ifndef SOFA_HookeFORCEFIELD_H
#define SOFA_HookeFORCEFIELD_H

#include "../initFlexible.h"
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/MechanicalState.h>
#include "HookeMaterialBlock.inl"
#include "../quadrature/BaseGaussPointSampler.h"

#include <sofa/component/linearsolver/EigenSparseMatrix.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

using helper::vector;

/** Compute stress from strain (=apply material law)
  * using Hooke's Law for isotropic homogeneous materials:
*/

template <class _DataTypes>
class HookeForceField : public core::behavior::ForceField<_DataTypes>
{
public:
    typedef core::behavior::ForceField<_DataTypes> Inherit;
    SOFA_CLASS(SOFA_TEMPLATE(HookeForceField,_DataTypes),SOFA_TEMPLATE(core::behavior::ForceField, _DataTypes));

    /** @name  Input types    */
    //@{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef Data<typename DataTypes::VecCoord> DataVecCoord;
    typedef Data<typename DataTypes::VecDeriv> DataVecDeriv;
    typedef core::behavior::MechanicalState<DataTypes> mstateType;
    //@}

    /** @name  material types    */
    //@{
    typedef defaulttype::HookeMaterialBlock<DataTypes> Block;  ///< Material block object
    typedef vector<Block >  SparseMatrix;

    typedef typename Block::MatBlock  MatBlock;  ///< Material block matrix
    typedef linearsolver::EigenSparseMatrix<DataTypes,DataTypes>    SparseMatrixEigen;
    //@}


protected:
    Data<Real> _youngModulus;
    Data<Real> _poissonRatio;
    Data<Real> _viscosity;
    Data<bool> assembleC;
    Data<bool> assembleK;

    SparseMatrix material;
    SparseMatrixEigen C;
    SparseMatrixEigen K;

    HookeForceField(core::behavior::MechanicalState<DataTypes> *mm = NULL)
        : Inherit(mm)
        , _youngModulus(core::objectmodel::BaseObject::initData(&_youngModulus,(Real)5000,"youngModulus","Young Modulus"))
        , _poissonRatio(core::objectmodel::BaseObject::initData(&_poissonRatio,(Real)0.45f,"poissonRatio","Poisson Ratio"))
        , _viscosity(core::objectmodel::BaseObject::initData(&_viscosity,(Real)0,"viscosity","Viscosity (stress/strainRate)"))
        , assembleC ( initData ( &assembleC,false, "assembleC","Assemble the Compliance matrix" ) )
        , assembleK ( initData ( &assembleK,false, "assembleK","Assemble the Stifness matrix" ) )
    {
        _poissonRatio.setWidget("poissonRatio");
    }

    virtual ~HookeForceField()
    {

    }

public:
    virtual void init()
    {
        if(!(this->mstate)) this->mstate = dynamic_cast<mstateType*>(this->getContext()->getMechanicalState());
        if(!(this->mstate)) { serr<<"state not found"<< sendl; return; }
        reinit();
        Inherit::init();
    }

    virtual void reinit()
    {
        // retrieve volume integrals
        engine::BaseGaussPointSampler* sampler=NULL;
        this->getContext()->get(sampler,core::objectmodel::BaseContext::SearchUp);
        if( !sampler ) serr<<"Gauss sampler not found -> use unit volumes"<< sendl;

        // reinit material
        typename mstateType::ReadVecCoord X = this->mstate->readPositions();
        material.resize(X.size());

        for(unsigned int i=0; i<material.size(); i++)
        {
            Real vol=0;
            if(sampler) vol=sampler->f_volume.getValue()[i][0];
            material[i].init(this->_youngModulus.getValue(),this->_poissonRatio.getValue(),this->_viscosity.getValue(),vol);
        }

        // reinit matrices
        if(this->assembleC.getValue()) updateC();
        if(this->assembleK.getValue()) updateK();

        Inherit::reinit();
    }

    virtual void addForce(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& _f , const DataVecCoord& _x , const DataVecDeriv& _v)
    {
        VecDeriv&  f = *_f.beginEdit();
        const VecCoord&  x = _x.getValue();
        const VecDeriv&  v = _v.getValue();

        for(unsigned int i=0; i<material.size(); i++)
        {
            material[i].addForce(f[i],x[i],v[i]);
        }
        _f.endEdit();

        if(this->assembleC.getValue()) updateC();
        if(this->assembleK.getValue()) updateK();
    }

    virtual void addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv&   _df , const DataVecDeriv&   _dx )
    {
        if(this->assembleK.getValue())  K.mult(_df,_dx);
        else
        {
            VecDeriv&  df = *_df.beginEdit();
            const VecDeriv&  dx = _dx.getValue();

            for(unsigned int i=0; i<material.size(); i++)
            {
                material[i].addDForce(df[i],dx[i],mparams->kFactor(),mparams->bFactor());
            }
            _df.endEdit();
        }
    }

    const defaulttype::BaseMatrix* getC(const core::MechanicalParams */*mparams*/)
    {
        if(!this->assembleC.getValue()) updateC();
        return &C;
    }

    const defaulttype::BaseMatrix* getK(const core::MechanicalParams */*mparams*/)
    {
        if(!this->assembleK.getValue()) updateK();
        return &K;
    }

    void updateC()
    {
        //        ReadAccessor<Data<InVecCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
        //        ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
        //        eigenJacobian.resizeBlocks(out.size(),in.size());
        //        for(unsigned int i=0;i<jacobian.size();i++)
        //        {
        //            //        eigenJacobian.setBlock( i, i, jacobian[i].getJ());

        //            // Put all the blocks of the row in an array, then send the array to the matrix
        //            // Not very efficient: MatBlock creations could be avoided.
        //            vector<MatBlock> blocks;
        //            vector<unsigned> columns;
        //            columns.push_back( i );
        //            blocks.push_back( jacobian[i].getJ() );
        //            eigenJacobian.appendBlockRow( i, columns, blocks );
        //        }
        //        eigenJacobian.endEdit();
    }

    void updateK()
    {

    }

};


}
}
}

#endif
