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
#ifndef FLEXIBLE_MooneyRivlinMaterialBlock_INL
#define FLEXIBLE_MooneyRivlinMaterialBlock_INL

#include "MooneyRivlinMaterialBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/StrainTypes.h"

namespace sofa
{

namespace defaulttype
{

//////////////////////////////////////////////////////////////////////////////////
////  macros
//////////////////////////////////////////////////////////////////////////////////
#define I331(type)  InvariantStrainTypes<3,3,0,type>
//#define I332(type)  InvariantStrainTypes<3,3,1,type>
//#define I333(type)  InvariantStrainTypes<3,3,2,type>
#define U331(type)  PrincipalStretchesStrainTypes<3,3,0,type>



//////////////////////////////////////////////////////////////////////////////////
////  I331
//////////////////////////////////////////////////////////////////////////////////

template<class _Real>
class MooneyRivlinMaterialBlock< I331(_Real) > :
    public  BaseMaterialBlock< I331(_Real) >
{
public:
    typedef I331(_Real) T;

    typedef BaseMaterialBlock<T> Inherit;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::Deriv Deriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::Real Real;

    /**
      * DOFs: sqrt(I1), sqrt(I2), J
      *
      * classic Mooney rivlin
      *     - W = vol* [ C1 ( I1 - 3)  + C2 ( I2 - 3) + bulk/2 (I3 -1)^2 ]
      *     - f = -vol [ 2*C1*sqrt(I1) , 2*C2*sqrt(I2) , bulk*(J-1) ]
      *     - df =  -vol [ 2*C1 , 2*C2 , bulk*dJ ]
      */

    static const bool constantK=true;

    Real C1Vol2;  ///<  first coef * volume * 2
    Real C2Vol2;  ///<  second coef * volume * 2
    Real bulkVol; ///< bulk modulus * volume

    void init(const Real &C1,const Real &C2,const Real &bulk)
    {
        Real vol=1.;
        if(this->volume) vol=(*this->volume)[0];
        C1Vol2=C1*vol*(Real)2.;
        C2Vol2=C2*vol*(Real)2.;
        bulkVol = bulk * vol;
    }

    Real getPotentialEnergy(const Coord& x) const
    {
        Real Jm1 = x.getStrain()[2]-(Real)1;
        return C1Vol2*(Real)0.5*(x.getStrain()[0]*x.getStrain()[0]-(Real)3.) +
               C2Vol2*(Real)0.5*(x.getStrain()[1]*x.getStrain()[1]-(Real)3.) +
               bulkVol*(Real)0.5*Jm1*Jm1;
    }

    void addForce( Deriv& f , const Coord& x , const Deriv& /*v*/) const
    {
        f.getStrain()[0]-=C1Vol2*x.getStrain()[0];
        f.getStrain()[1]-=C2Vol2*x.getStrain()[1];
        f.getStrain()[2]-=bulkVol*(x.getStrain()[2]-(Real)1.);
    }

    void addDForce( Deriv&   df, const Deriv&   dx, const double& kfactor, const double& /*bfactor*/ ) const
    {
        df.getStrain()[0]-=C1Vol2*dx.getStrain()[0]*kfactor;
        df.getStrain()[1]-=C2Vol2*dx.getStrain()[1]*kfactor;
        df.getStrain()[2]-=bulkVol*dx.getStrain()[2]*kfactor;
    }

    MatBlock getK() const
    {
        MatBlock K = MatBlock();
        K[0][0]=-C1Vol2;
        K[1][1]=-C2Vol2;
        K[2][2]=-bulkVol;
        return K;
    }

    MatBlock getC() const
    {
        MatBlock C = MatBlock();
        C[0][0]=1./C1Vol2;
        C[1][1]=1./C2Vol2;
        C[2][2]=1./bulkVol;
        return C;
    }

    MatBlock getB() const
    {
        return MatBlock();
    }
};




//////////////////////////////////////////////////////////////////////////////////
////  U331
//////////////////////////////////////////////////////////////////////////////////

template<class _Real>
class MooneyRivlinMaterialBlock< U331(_Real) > :
    public  BaseMaterialBlock< U331(_Real) >
{
public:
    typedef U331(_Real) T;

    typedef BaseMaterialBlock<T> Inherit;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::Deriv Deriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::Real Real;

    /**
      * DOFs: principal stretches U1,U2,U3   J=U1*U2*U3
      *
      * classic Mooney rivlin
      *     - W = vol * [ C1 ( (U1^2+U2^2+U3^2)/J^{2/3} - 3)  + C2 ( (U1^2.U2^2 + U2^2.U3^2 + U3^2.U1^2))/J^{4/3} - 3) + bulk/2 J^2]
      * see maple file ./doc/mooneyRivlin_principalStretches.mw for derivative
      */

    static const bool constantK=true;

    Real C1Vol;  ///<  first coef * volume
    Real C2Vol;  ///<  second coef * volume
    Real bulkVol;   ///<  volume coef * volume

    mutable MatBlock _K;

    static const Real MIN_DETERMINANT() {return 0.001;} ///< threshold to clamp J to avoid undefined deviatoric expressions

    void init(const Real &C1,const Real &C2,const Real &bulk)
    {
        Real vol=1.;
        if(this->volume) vol=(*this->volume)[0];
        C1Vol = C1*vol;
        C2Vol = C2*vol;
        bulkVol = bulk*vol;
    }

    Real getPotentialEnergy(const Coord& x) const
    {
        Real J = x.getStrain()[0]*x.getStrain()[1]*x.getStrain()[2];
        if( J<MIN_DETERMINANT() ) J = MIN_DETERMINANT();
        Real Jm1 = J-1;
        Real squareU[3] = { x.getStrain()[0]*x.getStrain()[0], x.getStrain()[1]*x.getStrain()[1], x.getStrain()[2]*x.getStrain()[2] };
        return C1Vol*((squareU[0]+squareU[1]+squareU[2])*pow(J,-2.0/3.0)-(Real)3.) +
                C2Vol*((squareU[0]*squareU[1]+squareU[1]*squareU[2]+squareU[2]*squareU[0])*pow(J,-4.0/3.0)-(Real)3.) +
                0.5*bulkVol*Jm1*Jm1;
    }

    void addForce( Deriv& f, const Coord& x, const Deriv& /*v*/) const
    {
        const Real& U1 = x.getStrain()[0];
        const Real& U2 = x.getStrain()[1];
        const Real& U3 = x.getStrain()[2];

//        Real squareU[3] = { x.getStrain()[0]*x.getStrain()[0], x.getStrain()[1]*x.getStrain()[1], x.getStrain()[2]*x.getStrain()[2] };

//        Real I1 = squareU[0]+squareU[1]+squareU[2];
//        Real I2 = squareU[0]*squareU[1] + squareU[1]*squareU[2] + squareU[2]*squareU[0];
//        Real J = x.getStrain()[0]*x.getStrain()[1]*x.getStrain()[2];

//        Real Jm23 = pow(J,-2.0/3.0);
//        Real Jm43 = pow(J,-4.0/3.0);


        // TODO optimize this crappy code generated by maple
        // there are a lot of redondencies between computation of f and K

        Real t1 =  U1 *  U2;

        Real J = t1 * U3; if( helper::rabs(J)<MIN_DETERMINANT() ) J=helper::sign(J)*MIN_DETERMINANT();
        Real Jm1 = J-1;

        Real Jm23 = pow(J,-2.0/3.0);
        Real Jm43 = pow(J,-4.0/3.0);
        Real Jm53 = pow(J,-5.0/3.0);
        Real Jm73 = pow(J,-7.0/3.0);
        Real Jm83 = pow(J,-8.0/3.0);
        Real Jm103 = pow(J,-10.0/3.0);

        Real t3 = Jm23;
        Real t4 =  U3 *  U3;
        Real t5 =  U2 *  U2;
        Real t6 =  U1 *  U1;
        Real t7 = -0.2e1 / 0.3e1;
        Real t8 = 2 * U1;
        t7 = t7 * (t6 + t5 + t4) * Jm53;
        Real t9 = Jm43;
        Real t10 = t6 + t5;
        Real t11 = -0.4e1 / 0.3e1;
        t11 = t11 * (t10 * t4 + t6 * t5) * Jm73;
        Real t2 = bulkVol * (Jm1);
        Real t12 = 2 * U2;
        Real t13 = 2 * U3;
        f.getStrain()[0] -= C1Vol * ( t8 * t3 + t7 *  U2 *  U3) + C2Vol * ( t8 * (t5 + t4) * t9 + t11 *  U2 *  U3) + t2 *  U2 *  U3;
        f.getStrain()[1] -= C1Vol * ( t12 * t3 + t7 *  U1 *  U3) + C2Vol * ( t12 * (t6 + t4) * t9 + t11 *  U1 *  U3) + t2 *  U1 *  U3;
        f.getStrain()[2] -= C1Vol * ( t13 * t3 + t7 * t1) + C2Vol * ( t13 * t10 * t9 + t11 * t1) + t2 * t1;

        t3 = Jm53;
        t7 = t6 + t5 + t4;
        t8 = Jm83;
        t9 =  U2 * U3;
        t10 = 0.2e1 * Jm23;
        t11 = t5 + t4;
        t12 = 2 * t11;
        t13 = Jm43;
        Real t14 = t12 * U1;
        Real t15 = Jm73;
        Real t16 = t6 + t5;
        Real t17 = t16 * t4 + t6 * t5;
        Real t18 = Jm103;
        Real t19 = bulkVol * t5;
        Real t20 = (0.10e2 / 0.9e1 * J * t8 - 0.2e1 / 0.3e1 * t3) * t7;
        Real t21 = t6 + t4;
        Real t22 = 2 * t21;
        Real t23 = t22 * U2;
        Real t24 = 0.4e1 * t13;
        Real t25 = 0.28e2 / 0.9e1 * t17 * t18;
        Real t26 = t14 * U1;
        Real t27 = t23 * U2;
        t2 = 0.2e1 * Jm1;
        Real t28 = bulkVol * U3 * t2 + C1Vol * U3 * (t20 - 0.4e1 / 0.3e1 * t3 * t16) + C2Vol * (-0.4e1 / 0.3e1 * ( t26 +  t27 + t17) * U3 * t15 + t1 * (t24 + t25 * t4));
        t16 = 0.2e1 * t16;
        Real t29 = t16 * U3;
        Real t30 = t29 * U3;
        Real t31 =  U1 * U3;
        t5 = bulkVol *  U2 * t2 + C1Vol *  U2 * (t20 - 0.4e1 / 0.3e1 * t3 *  t21) + C2Vol * (-0.4e1 / 0.3e1 * ( t26 + t30 + t17) *  U2 * t15 + t31 * (t24 + t25 * t5));
        t21 = -(0.8e1 / 0.3e1 * t3);
        t2 = bulkVol *  U1 * t2 + C1Vol *  U1 * (t20 - 0.4e1 / 0.3e1 * t3 *  t11) + C2Vol* (-0.4e1 / 0.3e1 * ( t27 + t30 + t17) *  U1 * t15 + t9 * (t24 + t25 * t6));
        _K[0][0] = C1Vol * (t9 * (-0.8e1 / 0.3e1 *  U1 * t3 + 0.10e2 / 0.9e1 * t9 * t7 * t8) + t10) + C2Vol * (t9 * (-0.8e1 / 0.3e1 *  t14 * t15 + 0.28e2 / 0.9e1 * t9 * t17 * t18) +  t12 * t13) + t19 * t4;
        _K[0][1] = t28;
        _K[0][2] = t5;
        _K[1][0] = t28;
        _K[1][1] = C1Vol * (t31 * ( (t21 * U2) + 0.10e2 / 0.9e1 * t31 * t7 * t8) + t10) + C2Vol * (t31 * (-0.8e1 / 0.3e1 *  t23 * t15 + t31 * t25) +  t22 * t13) + bulkVol * t6 * t4;
        _K[1][2] = t2;
        _K[2][0] = t5;
        _K[2][1] = t2;
        _K[2][2] = C1Vol * (t1 * ( t21 * U3 + 0.10e2 / 0.9e1 * t1 * t7 * t8) + t10) + C2Vol * (t1 * (-0.8e1 / 0.3e1 * t29 * t15 + t25 * t1) + t16 * t13) + t19 * t6;

    }

    void addDForce( Deriv& df, const Deriv& dx, const double& kfactor, const double& /*bfactor*/ ) const
    {
        df.getStrain() = -_K * dx.getStrain() * kfactor;
    }

    MatBlock getK() const
    {
        return _K;
    }

    MatBlock getC() const
    {
        MatBlock C = MatBlock();
        C.invert( _K );
        return C;
    }

    MatBlock getB() const
    {
        return MatBlock();
    }
};



} // namespace defaulttype
} // namespace sofa



#endif
