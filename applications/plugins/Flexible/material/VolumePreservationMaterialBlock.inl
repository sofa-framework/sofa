/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef FLEXIBLE_VolumePreservationMaterialBlock_INL
#define FLEXIBLE_VolumePreservationMaterialBlock_INL

#include "../material/VolumePreservationMaterialBlock.h"
#include "../BaseJacobian.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/StrainTypes.h"


namespace sofa
{

namespace defaulttype
{


//////////////////////////////////////////////////////////////////////////////////
////  I331
//////////////////////////////////////////////////////////////////////////////////

template<class _Real>
class VolumePreservationMaterialBlock< I331(_Real) > :
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
      * method 0:
      *     - W = vol* k/2 ln( J )^2
      *     - f =   [ 0 , 0 , - k*vol*ln(J)/J ]
      *     - df =  [ 0 , 0 , k*vol*(ln(J)-1)/J^2* dJ ]
      * method 1:
      *     - W = vol* k/2 (J-1)^2
      *     - f =   [ 0 , 0 , - k*vol*(J-1) ]
      *     - df =  [ 0 , 0 , - k*vol*dJ ]
      */

    static const bool constantK=false;

    Real KVol;  ///< bulk  * volume
    mutable Real dfdJ; ///< store stiffness

    void init( const Real &k )
    {
        Real vol=1.;
        if(this->volume) vol=(*this->volume)[0];
        KVol=k*vol;
    }

    Real getPotentialEnergy( const Coord& ) const { assert(false); return 0; }

    Real getPotentialEnergy_method0(const Coord& x) const
    {
        Real J=x.getStrain()[2];
        return KVol*log(J)*log(J)*(Real)0.5;
    }

    Real getPotentialEnergy_method1(const Coord& x) const
    {
        Real J=x.getStrain()[2];
        return KVol*(J-(Real)1.)*(J-(Real)1.)*(Real)0.5;
    }

    void addForce( Deriv& , const Coord& , const Deriv& ) const { assert(false); }

    void addForce_method0( Deriv& f , const Coord& x , const Deriv& /*v*/)  const
    {
        Real J=x.getStrain()[2];
        f.getStrain()[2]-=KVol*log(J)/J;
        dfdJ=KVol*(log(J)-(Real)1.)/(J*J);
    }

    void addForce_method1( Deriv& f , const Coord& x , const Deriv& /*v*/)  const
    {
        Real J=x.getStrain()[2];
        f.getStrain()[2]-=KVol*(J-(Real)1.);
        dfdJ=-KVol;
    }

    void addDForce( Deriv&   df , const Deriv&   dx, const SReal& kfactor, const SReal& /*bfactor*/ )  const
    {
        df.getStrain()[2]+=dfdJ*dx.getStrain()[2]*kfactor;
    }

    MatBlock getK() const
    {
        MatBlock K = MatBlock();
        K(2,2)=dfdJ;
        return K;
    }

    MatBlock getC() const
    {
        MatBlock C = MatBlock();
        C(2,2)=-1./dfdJ;
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
class VolumePreservationMaterialBlock< U331(_Real) > :
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
      * method 0:
      *     - W = vol* k/2 ln( U1*U2*U3 )^2
      *     - fi =   - k*vol*ln(U1*U2*U3)/Ui ]
      *     - dfi/dUi = -k*vol(1-ln(U1*U2*U3)) / Ui^2
      *       dfi/dUj = -k*vol/(UiUj)
      * method 1:
      *     - W = vol* k/2 (U1*U2*U3-1)^2
      *     - fi = -k*vol*(U1*U2*U3-1)UjUk
      *     - dfi/dUi = -k*vol*Uj^2*Uk^2
      *       dfi/dUj = -k*vol*Uk(U1*U2*U3+U1*U2*U3-1)
      */

    static const bool constantK=false;

    Real KVol;  ///< bulk  * volume
    mutable MatBlock _K; ///< store stiffness

    static const Real MIN_DETERMINANT() {return 0.001;} ///< J is clamped to avoid undefined deviatoric expressions

    void init( const Real &k )
    {
        Real vol=1.;
        if(this->volume) vol=(*this->volume)[0];
        KVol=k*vol;
    }

    Real getPotentialEnergy( const Coord& ) const { assert(false); return 0; }

    Real getPotentialEnergy_method0(const Coord& x) const
    {
        Real J = x.getStrain()[0]*x.getStrain()[1]*x.getStrain()[2];
        if( J<MIN_DETERMINANT() ) J=MIN_DETERMINANT();
        Real logJ = log(J);
        return KVol*logJ*logJ*(Real)0.5;
    }

    Real getPotentialEnergy_method1(const Coord& x) const
    {
        Real J = x.getStrain()[0]*x.getStrain()[1]*x.getStrain()[2];
        if( J<MIN_DETERMINANT() ) J=MIN_DETERMINANT();
        Real Jm1 = J-(Real)1;

        return KVol*Jm1*Jm1*(Real)0.5;
    }

    void addForce( Deriv&, const Coord&, const Deriv& ) const { assert(false); }

    void addForce_method0( Deriv& f , const Coord& x , const Deriv& /*v*/)  const
    {
        Real J = x.getStrain()[0]*x.getStrain()[1]*x.getStrain()[2];
        if( J<MIN_DETERMINANT() ) J=MIN_DETERMINANT();
        Real logJ = log(J);

        for( int i = 0 ; i<3 ; ++i )
            f.getStrain()[i] -= KVol*logJ/x.getStrain()[i];

        Real nom = KVol*(1-logJ);

        for( int i = 0 ; i<3 ; ++i )
        {
            _K[i][i] = nom / x.getStrain()[i]*x.getStrain()[i];
            for( int j = i+1 ; j<3 ; ++j )
            {
                _K[i][j] = _K[j][i] = KVol / ( x.getStrain()[i]*x.getStrain()[j] );
            }
        }
    }

    void addForce_method1( Deriv& f , const Coord& x , const Deriv& /*v*/)  const
    {
        Real J = x.getStrain()[0]*x.getStrain()[1]*x.getStrain()[2];
//        if( J<MIN_DETERMINANT() ) J=MIN_DETERMINANT();
        Real Jm1 = J-1;
        Real KVolJm1 = KVol * Jm1;

        f.getStrain()[0] -= KVolJm1 * x.getStrain()[1]*x.getStrain()[2];
        f.getStrain()[1] -= KVolJm1 * x.getStrain()[0]*x.getStrain()[2];
        f.getStrain()[2] -= KVolJm1 * x.getStrain()[1]*x.getStrain()[0];

        Real squareU[3] = { x.getStrain()[0]*x.getStrain()[0], x.getStrain()[1]*x.getStrain()[1], x.getStrain()[2]*x.getStrain()[2] };

        _K[0][0] = KVol * squareU[1] * squareU[2];
        _K[1][1] = KVol * squareU[0] * squareU[2];
        _K[2][2] = KVol * squareU[1] * squareU[0];

        for( int i = 0 ; i<3 ; ++i )
        {
            for( int j = i+1 ; j<3 ; ++j )
            {
                unsigned k = (j+1)%3;
                _K[i][j] = _K[j][i] = KVol * x.getStrain()[k] *( J + Jm1 );
            }
        }
    }

    void addDForce( Deriv& df, const Deriv& dx, const SReal& kfactor, const SReal& /*bfactor*/ )  const
    {
        df.getStrain() -= _K * dx.getStrain() * kfactor;
    }

    MatBlock getK() const
    {
        return _K;
    }

    MatBlock getC() const
    {
        MatBlock C;
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
