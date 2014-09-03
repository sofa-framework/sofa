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
#ifndef SOFA_BaseMaterialFORCEFIELD_H
#define SOFA_BaseMaterialFORCEFIELD_H

#include "../initFlexible.h"
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/MechanicalState.h>

#include "../material/BaseMaterial.h"
#include "../quadrature/BaseGaussPointSampler.h"

#include <SofaEigen2Solver/EigenSparseMatrix.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

using helper::vector;

/** Abstract interface to allow for resizing
*/
class SOFA_Flexible_API BaseMaterialForceField : public virtual core::objectmodel::BaseObject
{
public:
    virtual void resize()=0;
    virtual double getPotentialEnergy( const unsigned int index ) const=0;
};


/** Abstract forcefield using MaterialBlocks or sparse eigen matrix
*/

template <class MaterialBlockType>
class BaseMaterialForceFieldT : public core::behavior::ForceField<typename MaterialBlockType::T>, public BaseMaterialForceField
{
public:
    typedef core::behavior::ForceField<typename MaterialBlockType::T> Inherit;
    SOFA_ABSTRACT_CLASS2(SOFA_TEMPLATE(BaseMaterialForceFieldT,MaterialBlockType),SOFA_TEMPLATE(core::behavior::ForceField,typename MaterialBlockType::T),BaseMaterialForceField);

    /** @name  Input types    */
    //@{
    typedef typename MaterialBlockType::T DataTypes;
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
    typedef MaterialBlockType BlockType;  ///< Material block object
    typedef vector<BlockType >  SparseMatrix;

    typedef typename BlockType::MatBlock  MatBlock;  ///< Material block matrix
    typedef linearsolver::EigenSparseMatrix<DataTypes,DataTypes>    SparseMatrixEigen;
    //@}


    virtual void resize()
    {
        if(!(this->mstate)) return;

        if(this->f_printLog.getValue()) std::cout<<"Material::resize()"<<std::endl;

        // init material
        typename mstateType::ReadVecCoord X = this->mstate->readPositions();
        material.resize(X.size());

        // retrieve volume integrals
        engine::BaseGaussPointSampler* sampler=NULL;
        this->getContext()->get(sampler,core::objectmodel::BaseContext::SearchUp);
        if( !sampler ) { serr<<"Gauss point sampler not found -> use unit volumes"<< sendl; for(unsigned int i=0; i<material.size(); i++) material[i].volume=NULL; }
        else for(unsigned int i=0; i<material.size(); i++) material[i].volume=&sampler->f_volume.getValue()[i];

        reinit();
    }


    /** @name forceField functions */
    //@{
    virtual void init()
    {
        if(!(this->mstate))
        {
            this->mstate = dynamic_cast<mstateType*>(this->getContext()->getMechanicalState());
            if(!(this->mstate)) { serr<<"state not found"<< sendl; return; }
        }

        resize();

        Inherit::init();
    }

    virtual void reinit()
    {

        addForce(NULL, *this->mstate->write(core::VecDerivId::force()), *this->mstate->read(core::ConstVecCoordId::position()), *this->mstate->read(core::ConstVecDerivId::velocity()));

        // reinit matrices
        if(this->assemble.getValue())
        {
            updateC();
            updateK();
            updateB();
        }

        Inherit::reinit();
    }

    virtual void addForce(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& _f , const DataVecCoord& _x , const DataVecDeriv& _v)
    {
        if(this->mstate->getSize()!=(int)material.size()) resize();

        VecDeriv&  f = *_f.beginEdit();
        const VecCoord&  x = _x.getValue();
        const VecDeriv&  v = _v.getValue();

        for(unsigned int i=0; i<material.size(); i++)
        {
            material[i].addForce(f[i],x[i],v[i]);
        }
        _f.endEdit();

//        if(!BlockType::constantK && this->assemble.getValue())
//        {
////            updateC(); // no need to update C on regular basis, if C is needed, the forcefield is a compliance and addForce is not call
//            updateK();
//            updateB();
//        }

        if(this->f_printLog.getValue())
        {
            Real W=0;  for(unsigned int i=0; i<material.size(); i++) W+=material[i].getPotentialEnergy(x[i]);
            std::cout<<this->getName()<<":addForce, potentialEnergy="<<W<<std::endl;
        }
    }

    virtual void addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv&   _df , const DataVecDeriv&   _dx )
    {
        VecDeriv&  df = *_df.beginEdit();
        const VecDeriv&  dx = _dx.getValue();

        if(this->assemble.getValue())
        {
            B.addMult(df,dx,mparams->bFactor());
            K.addMult(df,dx,mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue()));
        }
        else
        {
            for(unsigned int i=0; i<material.size(); i++)
            {
                material[i].addDForce(df[i],dx[i],mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue()),mparams->bFactor());
            }
        }

        _df.endEdit();
    }


    const defaulttype::BaseMatrix* getComplianceMatrix(const core::MechanicalParams * /*mparams*/)
    {
        if(!this->assemble.getValue() || !BlockType::constantK)
        {
            // MattN: quick and dirty fix to update the compliance matrix for a non-linear material
            // C is generally computed as K^{-1}, K is computed in addForce that is not call for compliances...
            // A deeper modification in forcefield API is required to fix this for all forcedields
            // maybe a cleaner fix is possible only for flexible
            {
                const DataVecCoord& xx = *this->mstate->read(core::ConstVecCoordId::position());
                const DataVecCoord& vv = *this->mstate->read(core::ConstVecDerivId::velocity());
                const VecCoord&  x = xx.getValue();
                const VecDeriv&  v = vv.getValue();
                VecDeriv f_bidon; f_bidon.resize( x.size() );
                for(unsigned int i=0; i<material.size(); i++)
                    material[i].addForce(f_bidon[i],x[i],v[i]); // too much stuff is computed there but at least C is updated
            }

            updateC();
        }
        return &C;
    }

    virtual void addKToMatrix( sofa::defaulttype::BaseMatrix * matrix, double kFact, unsigned int &offset )
    {
        if(!this->assemble.getValue() || !BlockType::constantK) updateK();

        K.addToBaseMatrix( matrix, kFact, offset );
    }

    virtual void addBToMatrix(sofa::defaulttype::BaseMatrix *matrix, SReal bFact, unsigned int &offset)
    {
        if(!this->assemble.getValue() || !BlockType::constantK) updateB();

        B.addToBaseMatrix( matrix, bFact, offset );
    }

    void draw(const core::visual::VisualParams* /*vparams*/)
    {
    }
    //@}


    virtual double getPotentialEnergy( const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, const DataVecCoord& x ) const
    {
        double e = 0;
        const VecCoord& _x = x.getValue();

        for( unsigned int i=0 ; i<material.size() ; i++ )
        {
            e += material[i].getPotentialEnergy( _x[i] );
        }
        return e;
    }

    virtual double getPotentialEnergy( const unsigned int index ) const
    {
        if(!this->mstate) return 0;
        helper::ReadAccessor<Data< VecCoord > >  x(*this->mstate->read(core::ConstVecCoordId::position()));
        if(index>=material.size()) return 0;
        if(index>=x.size()) return 0;
        return material[index].getPotentialEnergy( x[index] );
    }


    Data<bool> assemble;

protected:

    BaseMaterialForceFieldT(core::behavior::MechanicalState<DataTypes> *mm = NULL)
        : Inherit(mm)
        , assemble ( initData ( &assemble,false, "assemble","Assemble the needed material matrices (compliance C,stiffness K,damping B)" ) )
    {

    }

    virtual ~BaseMaterialForceFieldT()    {     }

    SparseMatrix material;

    SparseMatrixEigen C;

    void updateC()
    {
        if(!(this->mstate)) { serr<<"state not found"<< sendl; return; }
        typename mstateType::ReadVecCoord X = this->mstate->readPositions();

        C.resizeBlocks(X.size(),X.size());
        for(unsigned int i=0; i<material.size(); i++)
        {
//            vector<MatBlock> blocks;
//            vector<unsigned> columns;
//            columns.push_back( i );
//            blocks.push_back( material[i].getC() );
//            C.appendBlockRow( i, columns, blocks );
            C.beginBlockRow(i);
            C.createBlock(i,material[i].getC());
            C.endBlockRow();
        }
//        C.endEdit();
        C.compress();
    }

    SparseMatrixEigen K;

    void updateK()
    {
        if(!(this->mstate)) { serr<<"state not found"<< sendl; return; }
        typename mstateType::ReadVecCoord X = this->mstate->readPositions();

        K.resizeBlocks(X.size(),X.size());
        for(unsigned int i=0; i<material.size(); i++)
        {
//            vector<MatBlock> blocks;
//            vector<unsigned> columns;
//            columns.push_back( i );
//            blocks.push_back( material[i].getK() );
//            K.appendBlockRow( i, columns, blocks );
            K.beginBlockRow(i);
            K.createBlock(i,material[i].getK());
            K.endBlockRow();
        }
//        K.endEdit();
        K.compress();
    }


    SparseMatrixEigen B;

    void updateB()
    {
        if(!(this->mstate)) { serr<<"state not found"<< sendl; return; }
        typename mstateType::ReadVecCoord X = this->mstate->readPositions();

        B.resizeBlocks(X.size(),X.size());
        for(unsigned int i=0; i<material.size(); i++)
        {
//            vector<MatBlock> blocks;
//            vector<unsigned> columns;
//            columns.push_back( i );
//            blocks.push_back( material[i].getB() );
//            B.appendBlockRow( i, columns, blocks );
            B.beginBlockRow(i);
            B.createBlock(i,material[i].getB());
            B.endBlockRow();
        }
//        B.endEdit();
        B.compress();
    }

};


}
}
}

#endif
