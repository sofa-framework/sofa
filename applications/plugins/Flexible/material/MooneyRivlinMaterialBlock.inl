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

#include "../material/MooneyRivlinMaterialBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/StrainTypes.h"

#include "../types/StrainTypes.h"

namespace sofa
{

namespace defaulttype
{

//////////////////////////////////////////////////////////////////////////////////
////  macros
//////////////////////////////////////////////////////////////////////////////////
#define I331(type)  BaseStrainTypes<3,3,1,type>
#define I332(type)  BaseStrainTypes<3,3,2,type>
#define I333(type)  BaseStrainTypes<3,3,3,type>

//////////////////////////////////////////////////////////////////////////////////
////  helpers
//////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////////////
////  F331
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
      * power =1 : classic Mooney rivlin (unstable..)
      *     - W = vol* [ C1 ( I1 - 3)  + C2 ( I2 - 3) ]
      *     - f = - [ vol*C1 , vol*C2 , 0 ]
      *     - df =  [ 0      , 0      , 0 ]
      * power !=1 : ewtended Mooney rivlin
      *     - W = vol* [ C1 ( I1 - 3)^p  + C2 ( I2 - 3)^p ]
      *     - f = - [ vol*C1* p*(I1-3)^(p-1) , vol*C2* p*(I2-3)^(p-1) , 0 ]
      *     - df = - [ vol*C1* p*(p-1)*(I1-3)^(p-2) , vol*C2* p*(p-1)*(I2-3)^(p-2) , 0 ]
      */

    Real C1Vol;  ///<  first coef * volume
    Real C2Vol;  ///<  second coef * volume
    Real Power;
    Real i1;
    Real i2;

    void init(const Real &C1,const Real &C2,const Real &power)
    {
        Real vol=1.;
        if(this->volume) vol=(*this->volume)[0];
        C1Vol=C1*vol;
        C2Vol=C2*vol;
        Power=power;
    }

    Real getPotentialEnergy(const Coord& x) const
    {
        if(Power==1) return C1Vol*(x.getStrain()[0]-(Real)3.) + C2Vol*(x.getStrain()[1]-(Real)3.);
        else return C1Vol*pow(x.getStrain()[0]-(Real)3.,Power) + C2Vol*pow(x.getStrain()[1]-(Real)3.,Power);
    }

    void addForce( Deriv& f , const Coord& x , const Deriv& /*v*/)
    {
        if(Power==1)
        {
            f.getStrain()[0]-=C1Vol;
            f.getStrain()[1]-=C2Vol;
        }
        else
        {
            i1=x.getStrain()[0]-(Real)3.;
            i2=x.getStrain()[1]-(Real)3.;

            f.getStrain()[0]-=C1Vol*Power*pow(i1,Power-(Real)1.);
            f.getStrain()[1]-=C2Vol*Power*pow(i2,Power-(Real)1.);
        }
    }

    void addDForce( Deriv&   df, const Deriv&   dx, const double& kfactor, const double& /*bfactor*/ )
    {
        if(Power==1) return;
        if(Power==2)
        {
            df.getStrain()[0]-=C1Vol*Power*dx.getStrain()[0]*kfactor;
            df.getStrain()[1]-=C2Vol*Power*dx.getStrain()[1]*kfactor;
        }
        else
        {
            df.getStrain()[0]-=C1Vol*Power*(Power-(Real)1.)*pow(i1,Power-(Real)2.)*dx.getStrain()[0]*kfactor;
            df.getStrain()[1]-=C2Vol*Power*(Power-(Real)1.)*pow(i2,Power-(Real)2.)*dx.getStrain()[1]*kfactor;
        }

    }


    MatBlock getK()
    {
        MatBlock K;
        return K;
    }

    MatBlock getC()
    {
        MatBlock C;
        return C;
    }

    MatBlock getB()
    {
        MatBlock B;
        return B;
    }
};




} // namespace defaulttype
} // namespace sofa



#endif
