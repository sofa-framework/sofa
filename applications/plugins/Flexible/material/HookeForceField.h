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
#ifndef SOFA_HookeFORCEFIELD_H
#define SOFA_HookeFORCEFIELD_H

#include <Flexible/config.h>
#include "BaseMaterialForceField.h"
#include "HookeMaterialBlock.inl"


namespace sofa
{
namespace component
{
namespace forcefield
{


/** Compute stress from strain (=apply material law)
  * using Hooke's Law for isotropic homogeneous materials:
*/

template <class _DataTypes>
class HookeForceField : public BaseMaterialForceFieldT<defaulttype::HookeMaterialBlock<_DataTypes, defaulttype::IsotropicHookeLaw<typename _DataTypes::Real, _DataTypes::material_dimensions, _DataTypes::strain_size> > >
{
public:
    typedef defaulttype::IsotropicHookeLaw<typename _DataTypes::Real, _DataTypes::material_dimensions, _DataTypes::strain_size> LawType;
    typedef defaulttype::HookeMaterialBlock<_DataTypes, LawType > BlockType;
    typedef BaseMaterialForceFieldT<BlockType> Inherit;

    SOFA_CLASS(SOFA_TEMPLATE(HookeForceField,_DataTypes),SOFA_TEMPLATE(BaseMaterialForceFieldT, BlockType));

    typedef typename Inherit::Real Real;

    /** @name  Material parameters */
    //@{
    Data<helper::vector<Real> > _youngModulus;
    Data<helper::vector<Real> > _poissonRatio;
    Data<helper::vector<Real> > _viscosity;
    //@}

    virtual void reinit()
    {
        Real youngModulus=0,poissonRatio=0,viscosity=0;
        for(unsigned int i=0; i<this->material.size(); i++)
        {
            if(i<_youngModulus.getValue().size()) youngModulus=_youngModulus.getValue()[i]; else if(_youngModulus.getValue().size()) youngModulus=_youngModulus.getValue()[0];
            if(i<_poissonRatio.getValue().size()) poissonRatio=_poissonRatio.getValue()[i]; else if(_poissonRatio.getValue().size()) poissonRatio=_poissonRatio.getValue()[0];
            if(i<_viscosity.getValue().size())    viscosity=_viscosity.getValue()[i];       else if(_viscosity.getValue().size())    viscosity=_viscosity.getValue()[0];

            assert( helper::isClamped<Real>( poissonRatio, -1+std::numeric_limits<Real>::epsilon(), 0.5-std::numeric_limits<Real>::epsilon() ) );

            std::vector<Real> params; params.push_back(youngModulus); params.push_back(poissonRatio);
            this->material[i].init( params, viscosity );
        }
        Inherit::reinit();
    }

    //Pierre-Luc : I added this function to be able to evaluate forces without using visitors
    virtual void addForce(typename Inherit::DataVecDeriv& _f , const typename Inherit::DataVecCoord& _x , const typename Inherit::DataVecDeriv& _v, const helper::vector<SReal> _vol)
     {
         if(this->f_printLog.getValue()==true)
             std::cout << SOFA_CLASS_METHOD << std::endl;

         typename Inherit::VecDeriv&  f = *_f.beginEdit();
         const typename Inherit::VecCoord&  x = _x.getValue();
         const typename Inherit::VecDeriv&  v = _v.getValue();

         //Init material block
         sofa::helper::vector< BlockType > materialBlock;
         typedef component::engine::BaseGaussPointSampler::volumeIntegralType volumeIntegralType;
         helper::vector<volumeIntegralType> _volIntType;
         _volIntType.resize(_vol.size());
         for(size_t i=0; i<_volIntType.size(); ++i)
         {
             _volIntType[i].resize(1);
             _volIntType[i][0] = _vol[i];
         }

         materialBlock.resize(_volIntType.size());
         for(unsigned int i=0; i<materialBlock.size(); i++)
             materialBlock[i].volume=&_volIntType[i];


         const helper::vector<Real>& vecYoungModulus = _youngModulus.getValue();
         const helper::vector<Real>& vecPoissonRatio = _poissonRatio.getValue();
         const helper::vector<Real>& vecViscosity = _viscosity.getValue();

         Real youngModulus=0,poissonRatio=0,viscosity=0;
         for(unsigned int i=0; i<materialBlock.size(); i++)
         {
             if(i<vecYoungModulus.size()) youngModulus=vecYoungModulus[i]; else if(vecYoungModulus.size()) youngModulus=vecYoungModulus[0];
             if(i<vecPoissonRatio.size()) poissonRatio=vecPoissonRatio[i]; else if(vecPoissonRatio.size()) poissonRatio=vecPoissonRatio[0];
             if(i<vecViscosity.size())    viscosity=vecViscosity[i];       else if(vecViscosity.size())    viscosity=vecViscosity[0];
             assert( helper::isClamped<Real>( poissonRatio, -1+std::numeric_limits<Real>::epsilon(), 0.5-std::numeric_limits<Real>::epsilon() ) );
             std::vector<Real> params; params.push_back(youngModulus); params.push_back(poissonRatio);
             materialBlock[i].init( params, viscosity );
         }

         //Compute
         for(unsigned int  i=0; i<_volIntType.size(); i++)
         {
             materialBlock[i].addForce(f[i],x[i],v[i]);
         }
         _f.endEdit();
     }


    /// Uniform damping ratio (i.e. viscosity/stiffness) applied to all the constrained values.
    virtual SReal getDampingRatio()
    {
        return this->_viscosity.getValue()[0]/this->_youngModulus.getValue()[0]; // somehow arbitrary. todo: check this.
    }


protected:
    HookeForceField(core::behavior::MechanicalState<_DataTypes> *mm = NULL)
        : Inherit(mm)
        , _youngModulus(initData(&_youngModulus,helper::vector<Real>((int)1,(Real)5000),"youngModulus","Young Modulus"))
        , _poissonRatio(initData(&_poissonRatio,helper::vector<Real>((int)1,(Real)0),"poissonRatio","Poisson Ratio ]-1,0.5["))
        , _viscosity(initData(&_viscosity,helper::vector<Real>((int)1,(Real)0),"viscosity","Viscosity (stress/strainRate)"))
    {
        // _poissonRatio.setWidget("poissonRatio");
    }

    virtual ~HookeForceField()     {    }
};




/** Compute stress from strain (=apply material law)
  * using Hooke's Law for orthotropic homogeneous materials:
*/

template <class _DataTypes>
class HookeOrthotropicForceField : public BaseMaterialForceFieldT<defaulttype::HookeMaterialBlock<_DataTypes, defaulttype::OrthotropicHookeLaw<typename _DataTypes::Real, _DataTypes::material_dimensions, _DataTypes::strain_size> > >
{
public:
    typedef defaulttype::OrthotropicHookeLaw<typename _DataTypes::Real, _DataTypes::material_dimensions, _DataTypes::strain_size> LawType;
    typedef defaulttype::HookeMaterialBlock<_DataTypes, LawType > BlockType;
    typedef BaseMaterialForceFieldT<BlockType> Inherit;

    SOFA_CLASS(SOFA_TEMPLATE(HookeOrthotropicForceField,_DataTypes),SOFA_TEMPLATE(BaseMaterialForceFieldT, BlockType));

    typedef typename Inherit::Real Real;

    /** @name  Material parameters */
    //@{
    Data<helper::vector<Real> > _youngModulusX;
    Data<helper::vector<Real> > _youngModulusY;
    Data<helper::vector<Real> > _youngModulusZ;
    Data<helper::vector<Real> > _poissonRatioXY;
    Data<helper::vector<Real> > _poissonRatioYZ;
    Data<helper::vector<Real> > _poissonRatioZX;
    Data<helper::vector<Real> > _shearModulusXY;
    Data<helper::vector<Real> > _shearModulusYZ;
    Data<helper::vector<Real> > _shearModulusZX;
    Data<helper::vector<Real> > _viscosity;
    //@}

    virtual void reinit()
    {
        if(_DataTypes::material_dimensions==3)
        {
            Real youngModulusX=0,youngModulusY=0,youngModulusZ=0,poissonRatioXY=0,poissonRatioYZ=0,poissonRatioZX=0,shearModulusXY=0,shearModulusYZ=0,shearModulusZX=0,viscosity=0;
            for(unsigned int i=0; i<this->material.size(); i++)
            {
                if(i<_youngModulusX.getValue().size()) youngModulusX=_youngModulusX.getValue()[i]; else if(_youngModulusX.getValue().size()) youngModulusX=_youngModulusX.getValue()[0];
                if(i<_youngModulusY.getValue().size()) youngModulusY=_youngModulusY.getValue()[i]; else if(_youngModulusY.getValue().size()) youngModulusY=_youngModulusY.getValue()[0];
                if(i<_youngModulusZ.getValue().size()) youngModulusZ=_youngModulusZ.getValue()[i]; else if(_youngModulusZ.getValue().size()) youngModulusZ=_youngModulusZ.getValue()[0];
                if(i<_poissonRatioXY.getValue().size()) poissonRatioXY=_poissonRatioXY.getValue()[i]; else if(_poissonRatioXY.getValue().size()) poissonRatioXY=_poissonRatioXY.getValue()[0];
                if(i<_poissonRatioYZ.getValue().size()) poissonRatioYZ=_poissonRatioYZ.getValue()[i]; else if(_poissonRatioYZ.getValue().size()) poissonRatioYZ=_poissonRatioYZ.getValue()[0];
                if(i<_poissonRatioZX.getValue().size()) poissonRatioZX=_poissonRatioZX.getValue()[i]; else if(_poissonRatioZX.getValue().size()) poissonRatioZX=_poissonRatioZX.getValue()[0];
                if(i<_shearModulusXY.getValue().size()) shearModulusXY=_shearModulusXY.getValue()[i]; else if(_shearModulusXY.getValue().size()) shearModulusXY=_shearModulusXY.getValue()[0];
                if(i<_shearModulusYZ.getValue().size()) shearModulusYZ=_shearModulusYZ.getValue()[i]; else if(_shearModulusYZ.getValue().size()) shearModulusYZ=_shearModulusYZ.getValue()[0];
                if(i<_shearModulusZX.getValue().size()) shearModulusZX=_shearModulusZX.getValue()[i]; else if(_shearModulusZX.getValue().size()) shearModulusZX=_shearModulusZX.getValue()[0];
                if(i<_viscosity.getValue().size())    viscosity=_viscosity.getValue()[i];       else if(_viscosity.getValue().size())    viscosity=_viscosity.getValue()[0];

                assert( helper::isClamped<Real>( poissonRatioXY, -1+std::numeric_limits<Real>::epsilon(), 0.5-std::numeric_limits<Real>::epsilon() ) );
                assert( helper::isClamped<Real>( poissonRatioYZ, -1+std::numeric_limits<Real>::epsilon(), 0.5-std::numeric_limits<Real>::epsilon() ) );
                assert( helper::isClamped<Real>( poissonRatioZX, -1+std::numeric_limits<Real>::epsilon(), 0.5-std::numeric_limits<Real>::epsilon() ) );

                std::vector<Real> params;
                params.push_back(youngModulusX); params.push_back(youngModulusY); params.push_back(youngModulusZ);
                params.push_back(poissonRatioXY); params.push_back(poissonRatioYZ); params.push_back(poissonRatioZX);
                params.push_back(shearModulusXY); params.push_back(shearModulusYZ); params.push_back(shearModulusZX);
                this->material[i].init( params, viscosity );
            }
        }
        else if(_DataTypes::material_dimensions==2)
        {
            _youngModulusZ.setDisplayed(false);
            _poissonRatioYZ.setDisplayed(false);
            _poissonRatioZX.setDisplayed(false);
            _shearModulusYZ.setDisplayed(false);
            _shearModulusZX.setDisplayed(false);

            Real youngModulusX=0,youngModulusY=0,poissonRatioXY=0,shearModulusXY=0,viscosity=0;
            for(unsigned int i=0; i<this->material.size(); i++)
            {
                if(i<_youngModulusX.getValue().size()) youngModulusX=_youngModulusX.getValue()[i]; else if(_youngModulusX.getValue().size()) youngModulusX=_youngModulusX.getValue()[0];
                if(i<_youngModulusY.getValue().size()) youngModulusY=_youngModulusY.getValue()[i]; else if(_youngModulusY.getValue().size()) youngModulusY=_youngModulusY.getValue()[0];
                if(i<_poissonRatioXY.getValue().size()) poissonRatioXY=_poissonRatioXY.getValue()[i]; else if(_poissonRatioXY.getValue().size()) poissonRatioXY=_poissonRatioXY.getValue()[0];
                if(i<_shearModulusXY.getValue().size()) shearModulusXY=_shearModulusXY.getValue()[i]; else if(_shearModulusXY.getValue().size()) shearModulusXY=_shearModulusXY.getValue()[0];
                if(i<_viscosity.getValue().size())    viscosity=_viscosity.getValue()[i];       else if(_viscosity.getValue().size())    viscosity=_viscosity.getValue()[0];

                std::vector<Real> params;
                params.push_back(youngModulusX); params.push_back(youngModulusY);
                params.push_back(poissonRatioXY);  params.push_back(shearModulusXY);
                this->material[i].init( params, viscosity );
            }
        }
        Inherit::reinit();
    }


    /// Uniform damping ratio (i.e. viscosity/stiffness) applied to all the constrained values.
    virtual SReal getDampingRatio()
    {
        return this->_viscosity.getValue()[0]/this->_youngModulusX.getValue()[0]; // somehow arbitrary. todo: check this.
    }


protected:
    HookeOrthotropicForceField(core::behavior::MechanicalState<_DataTypes> *mm = NULL)
        : Inherit(mm)
        , _youngModulusX(initData(&_youngModulusX,helper::vector<Real>((int)1,(Real)5000),"youngModulusX","Young Modulus along X"))
        , _youngModulusY(initData(&_youngModulusY,helper::vector<Real>((int)1,(Real)5000),"youngModulusY","Young Modulus along Y"))
        , _youngModulusZ(initData(&_youngModulusZ,helper::vector<Real>((int)1,(Real)5000),"youngModulusZ","Young Modulus along Z"))
        , _poissonRatioXY(initData(&_poissonRatioXY,helper::vector<Real>((int)1,(Real)0),"poissonRatioXY","Poisson Ratio about XY plane ]-1,0.5["))
        , _poissonRatioYZ(initData(&_poissonRatioYZ,helper::vector<Real>((int)1,(Real)0),"poissonRatioYZ","Poisson Ratio about YZ plane ]-1,0.5["))
        , _poissonRatioZX(initData(&_poissonRatioZX,helper::vector<Real>((int)1,(Real)0),"poissonRatioZX","Poisson Ratio about ZX plane ]-1,0.5["))
        , _shearModulusXY(initData(&_shearModulusXY,helper::vector<Real>((int)1,(Real)1500),"shearModulusXY","Shear Modulus about XY plane"))
        , _shearModulusYZ(initData(&_shearModulusYZ,helper::vector<Real>((int)1,(Real)1500),"shearModulusYZ","Shear Modulus about YZ plane"))
        , _shearModulusZX(initData(&_shearModulusZX,helper::vector<Real>((int)1,(Real)1500),"shearModulusZX","Shear Modulus about ZX plane"))
        , _viscosity(initData(&_viscosity,helper::vector<Real>((int)1,(Real)0),"viscosity","Viscosity (stress/strainRate)"))
    {
        // _poissonRatio.setWidget("poissonRatio");
    }

    virtual ~HookeOrthotropicForceField()     {    }
};



/** Compute stress from strain (=apply material law)
  * using Hooke's Law for transverse isotropic homogeneous materials (about X axis) :
*/

template <class _DataTypes>
class HookeTransverseForceField : public BaseMaterialForceFieldT<defaulttype::HookeMaterialBlock<_DataTypes, defaulttype::TransverseHookeLaw<typename _DataTypes::Real, _DataTypes::material_dimensions, _DataTypes::strain_size> > >
{
public:
    typedef defaulttype::TransverseHookeLaw<typename _DataTypes::Real, _DataTypes::material_dimensions, _DataTypes::strain_size> LawType;
    typedef defaulttype::HookeMaterialBlock<_DataTypes, LawType > BlockType;
    typedef BaseMaterialForceFieldT<BlockType> Inherit;

    SOFA_CLASS(SOFA_TEMPLATE(HookeTransverseForceField,_DataTypes),SOFA_TEMPLATE(BaseMaterialForceFieldT, BlockType));

    typedef typename Inherit::Real Real;

    /** @name  Material parameters */
    //@{
    Data<helper::vector<Real> > _youngModulusX;
    Data<helper::vector<Real> > _youngModulusY;
    Data<helper::vector<Real> > _poissonRatioXY;
    Data<helper::vector<Real> > _poissonRatioYZ;
    Data<helper::vector<Real> > _shearModulusXY;
    Data<helper::vector<Real> > _viscosity;
    //@}

    virtual void reinit()
    {
            Real youngModulusX=0,youngModulusY=0,poissonRatioXY=0,poissonRatioYZ=0,shearModulusXY=0,viscosity=0;
            for(unsigned int i=0; i<this->material.size(); i++)
            {
                if(i<_youngModulusX.getValue().size()) youngModulusX=_youngModulusX.getValue()[i]; else if(_youngModulusX.getValue().size()) youngModulusX=_youngModulusX.getValue()[0];
                if(i<_youngModulusY.getValue().size()) youngModulusY=_youngModulusY.getValue()[i]; else if(_youngModulusY.getValue().size()) youngModulusY=_youngModulusY.getValue()[0];
                if(i<_poissonRatioXY.getValue().size()) poissonRatioXY=_poissonRatioXY.getValue()[i]; else if(_poissonRatioXY.getValue().size()) poissonRatioXY=_poissonRatioXY.getValue()[0];
                if(i<_poissonRatioYZ.getValue().size()) poissonRatioYZ=_poissonRatioYZ.getValue()[i]; else if(_poissonRatioYZ.getValue().size()) poissonRatioYZ=_poissonRatioYZ.getValue()[0];
                if(i<_shearModulusXY.getValue().size()) shearModulusXY=_shearModulusXY.getValue()[i]; else if(_shearModulusXY.getValue().size()) shearModulusXY=_shearModulusXY.getValue()[0];
                if(i<_viscosity.getValue().size())    viscosity=_viscosity.getValue()[i];       else if(_viscosity.getValue().size())    viscosity=_viscosity.getValue()[0];

                assert( helper::isClamped<Real>( poissonRatioXY, -1+std::numeric_limits<Real>::epsilon(), 0.5-std::numeric_limits<Real>::epsilon() ) );
                assert( helper::isClamped<Real>( poissonRatioYZ, -1+std::numeric_limits<Real>::epsilon(), 0.5-std::numeric_limits<Real>::epsilon() ) );

                std::vector<Real> params;
                params.push_back(youngModulusX); params.push_back(youngModulusY);
                params.push_back(poissonRatioXY); params.push_back(poissonRatioYZ);
                params.push_back(shearModulusXY);
                this->material[i].init( params, viscosity );
            }
        Inherit::reinit();
    }


    /// Uniform damping ratio (i.e. viscosity/stiffness) applied to all the constrained values.
    virtual SReal getDampingRatio()
    {
        return this->_viscosity.getValue()[0]/this->_youngModulusX.getValue()[0]; // somehow arbitrary. todo: check this.
    }


protected:
    HookeTransverseForceField(core::behavior::MechanicalState<_DataTypes> *mm = NULL)
        : Inherit(mm)
        , _youngModulusX(initData(&_youngModulusX,helper::vector<Real>((int)1,(Real)5000),"youngModulusX","Young Modulus along X"))
        , _youngModulusY(initData(&_youngModulusY,helper::vector<Real>((int)1,(Real)5000),"youngModulusY","Young Modulus along Y"))
        , _poissonRatioXY(initData(&_poissonRatioXY,helper::vector<Real>((int)1,(Real)0),"poissonRatioXY","Poisson Ratio about XY plane ]-1,0.5["))
        , _poissonRatioYZ(initData(&_poissonRatioYZ,helper::vector<Real>((int)1,(Real)0),"poissonRatioYZ","Poisson Ratio about YZ plane ]-1,0.5["))
        , _shearModulusXY(initData(&_shearModulusXY,helper::vector<Real>((int)1,(Real)1500),"shearModulusXY","Shear Modulus about XY plane"))
        , _viscosity(initData(&_viscosity,helper::vector<Real>((int)1,(Real)0),"viscosity","Viscosity (stress/strainRate)"))
    {
        // _poissonRatio.setWidget("poissonRatio");
    }

    virtual ~HookeTransverseForceField()     {    }
};
}
}
}

#endif
