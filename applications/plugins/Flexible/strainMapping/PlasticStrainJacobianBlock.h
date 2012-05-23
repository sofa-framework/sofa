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
#ifndef FLEXIBLE_PlasticStrainJacobianBlock_H
#define FLEXIBLE_PlasticStrainJacobianBlock_H

#include "../BaseJacobian.h"

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/StrainTypes.h"
#include "../helper.h"

namespace sofa
{

namespace defaulttype
{



#define E221(type)  StrainTypes<2,2,0,type>
#define E331(type)  StrainTypes<3,3,0,type>
#define E332(type)  StrainTypes<3,3,1,type>



/** Template class used to implement one jacobian block for PlasticStrainMapping */
template<class TStrain>
class PlasticStrainJacobianBlock : public BaseJacobianBlock<TStrain,TStrain>
{

public:

    typedef BaseJacobianBlock<TStrain,TStrain> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    typedef typename TStrain::StrainMat StrainMat;  ///< Matrix representing a strain
    typedef typename TStrain::StrainVec StrainVec;  ///< Vec representing a strain (Voigt notation)
    enum { strain_size = TStrain::strain_size };


    static const bool constantJ = true;


    /**
    Mapping:   ADDITION -> \f$ E_elastic = E_total - E_plastic \f ; MULTIPLICATION -> \f$ E_elastic = E_total * E_plastic^-1 \f$
    Jacobian:    \f$  dE = Id \f$
    */


    enum PlasticMethod { ADDITION, MULTIPLICATION }; ///< ADDITION -> Müller method (faster), MULTIPLICATION -> Fedkiw method
    PlasticMethod _method;

    Real _max;
    Real _yield; ///< squared _yield
    Real _creep;

    StrainVec _plasticStrain;

    void reset()
    {
        _plasticStrain.clear();
    }


    void addapply( OutCoord& result, const InCoord& data )
    {
        // eventually remove a part of the strain to simulate plasticity

        if( _method == MULTIPLICATION ) // totalStrain = elasticStrain * plasticStrain
        {
            // could be optimized by storing the computation of the previous time step
            StrainMat plasticStrainMat = StrainVoigtToMat( _plasticStrain ) + StrainMat::Identity();
            StrainMat plasticStrainMatInverse; plasticStrainMatInverse.invert( plasticStrainMat );

            // elasticStrain = totalStrain * plasticStrain^-1
            StrainMat elasticStrainMat = ( StrainVoigtToMat( data.getStrain() ) + StrainMat::Identity() ) * plasticStrainMatInverse;
            StrainVec elasticStrainVec = StrainMatToVoigt( elasticStrainMat - StrainMat::Identity() );

            // if( ||elasticStrain||  > c_yield ) plasticStrain += dt * c_creep * dt * elasticStrain
            if( elasticStrainVec.norm2() > _yield )
                _plasticStrain += _creep * elasticStrainVec;

            // if( ||plasticStrain|| > c_max ) plasticStrain *= c_max / ||plasticStrain||
            Real plasticStrainNorm2 = _plasticStrain.norm2();
            if( plasticStrainNorm2 > _max*_max )
                _plasticStrain *= _max / helper::rsqrt( plasticStrainNorm2 );

            plasticStrainMat = StrainVoigtToMat( _plasticStrain ) + StrainMat::Identity();

            // remaining elasticStrain = totalStrain * plasticStrain^-1
            plasticStrainMatInverse.invert( plasticStrainMat );
            elasticStrainMat = ( StrainVoigtToMat( data.getStrain() ) + StrainMat::Identity() ) * plasticStrainMatInverse;
            elasticStrainVec = StrainMatToVoigt( elasticStrainMat - StrainMat::Identity() );

            result.getStrain() += elasticStrainVec;
        }
        else //if( _method == ADDITION ) // totalStrain = elasticStrain + plasticStrain
        {
            // elasticStrain = totalStrain - plasticStrain
            StrainVec elasticStrain = data.getStrain() - _plasticStrain;

            // if( ||elasticStrain||  > c_yield ) plasticStrain += dt * c_creep * dt * elasticStrain
            if( elasticStrain.norm2() > _yield )
                _plasticStrain += _creep * elasticStrain;

            // if( ||plasticStrain|| > c_max ) plasticStrain *= c_max / ||plasticStrain||
            Real plasticStrainNorm2 = _plasticStrain.norm2();
            if( plasticStrainNorm2 > _max*_max )
                _plasticStrain *= _max / helper::rsqrt( plasticStrainNorm2 );

            // remaining elasticStrain = totatStrain - plasticStrain
            elasticStrain = data.getStrain() - _plasticStrain;

            result.getStrain() += elasticStrain;
        }
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getStrain() += data.getStrain();
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getStrain() += data.getStrain();
    }

    MatBlock getJ()
    {
        return MatBlock::Identity();
    }

    // Not Yet implemented..
    KBlock getK(const OutDeriv& /*childForce*/)
    {
        return KBlock();
    }

    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */)
    {
    }

}; // class PlasticStrainJacobianBlock

} // namespace defaulttype
} // namespace sofa



#endif // FLEXIBLE_PlasticStrainJacobianBlock_H
