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
    enum { order = TStrain::order };
    enum { spatial_dimensions = TStrain::spatial_dimensions };

    static const bool constant = true;


    /**
    Mapping:   ADDITION       -> \f$ E_elastic = E_total - E_plastic \f$
               MULTIPLICATION -> \f$ E_elastic = E_total * E_plastic^-1 \f$
    Jacobian:                    \f$  dE = Id \f$
    */


    InCoord _plasticStrain;

    void reset()
    {
        _plasticStrain.clear();
    }


    void addapply( OutCoord& /*result*/, const InCoord& /*data*/ ) {}

    void addapply_multiplication( OutCoord& result, const InCoord& data, Real max, Real squaredYield, Real creep )
    {
        // eventually remove a part of the strain to simulate plasticity

        // could be optimized by storing the computation of the previous time step
        StrainMat plasticStrainMat = StrainVoigtToMat( _plasticStrain.getStrain() ) + StrainMat::s_identity;
        StrainMat plasticStrainMatInverse; plasticStrainMatInverse.invert( plasticStrainMat );

        // elasticStrain = totalStrain * plasticStrain^-1
        StrainMat elasticStrainMat = ( StrainVoigtToMat( data.getStrain() ) + StrainMat::s_identity ) * plasticStrainMatInverse;
        StrainVec elasticStrainVec = StrainMatToVoigt( elasticStrainMat - StrainMat::s_identity );

        // if( ||elasticStrain||  > c_yield ) plasticStrain += dt * c_creep * dt * elasticStrain
        if( elasticStrainVec.norm2() > squaredYield )
            _plasticStrain.getStrain() += creep * elasticStrainVec;

        // if( ||plasticStrain|| > c_max ) plasticStrain *= c_max / ||plasticStrain||
        Real plasticStrainNorm2 = _plasticStrain.getStrain().norm2();
        if( plasticStrainNorm2 > max*max )
            _plasticStrain.getStrain() *= max / helper::rsqrt( plasticStrainNorm2 );

        plasticStrainMat = StrainVoigtToMat( _plasticStrain.getStrain() ) + StrainMat::s_identity;

        // remaining elasticStrain = totalStrain * plasticStrain^-1
        plasticStrainMatInverse.invert( plasticStrainMat );
        elasticStrainMat = ( StrainVoigtToMat( data.getStrain() ) + StrainMat::s_identity ) * plasticStrainMatInverse;
        elasticStrainVec = StrainMatToVoigt( elasticStrainMat - StrainMat::s_identity );

        result.getStrain() += elasticStrainVec;
    }

    void addapply_addition( OutCoord& result, const InCoord& data, Real max, Real squaredYield, Real creep )
    {
        // eventually remove a part of the strain to simulate plasticity

        // elasticStrain = totalStrain - plasticStrain
        InCoord elasticStrain = data - _plasticStrain;

        // if( ||elasticStrain||  > c_yield ) plasticStrain += dt * c_creep * dt * elasticStrain
        if( elasticStrain.getStrain().norm2() > squaredYield )
            _plasticStrain += elasticStrain * creep;

        // if( ||plasticStrain|| > c_max ) plasticStrain *= c_max / ||plasticStrain||
        Real plasticStrainNorm2 = _plasticStrain.getStrain().norm2();
        if( plasticStrainNorm2 > max*max )
            _plasticStrain.getStrain() *= max / helper::rsqrt( plasticStrainNorm2 );

        // remaining elasticStrain = totatStrain - plasticStrain
        elasticStrain = data - _plasticStrain;

        result += elasticStrain;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result += data;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result += data;
    }

    MatBlock getJ()
    {
        return MatBlock::s_identity;
    }

    KBlock getK(const OutDeriv& /*childForce*/, bool=false)
    {
        return KBlock();
    }

    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const SReal& /*kfactor */)
    {
    }

}; // class PlasticStrainJacobianBlock







///** Template class used to implement one jacobian block for PlasticStrainMapping for PrincipalStretchesStrainTypes*/
///// @warning @todo only implemented for principal stretches see as strain
///// @TODO it not working because the principal stretches are not ordered
//template<int _spatial_dimensions, int _material_dimensions, int _order, typename _Real>
//class PlasticStrainJacobianBlock_U : public BaseJacobianBlock< PrincipalStretchesStrainTypes<_spatial_dimensions,_material_dimensions,_order,_Real>,PrincipalStretchesStrainTypes<_spatial_dimensions,_material_dimensions,_order,_Real> >
//{

//public:

//    typedef PrincipalStretchesStrainTypes<_spatial_dimensions,_material_dimensions,_order,_Real> TStrain;
//    typedef BaseJacobianBlock<TStrain,TStrain> Inherit;
//    typedef typename Inherit::InCoord InCoord;
//    typedef typename Inherit::InDeriv InDeriv;
//    typedef typename Inherit::OutCoord OutCoord;
//    typedef typename Inherit::OutDeriv OutDeriv;
//    typedef typename Inherit::MatBlock MatBlock;
//    typedef typename Inherit::KBlock KBlock;
//    typedef typename Inherit::Real Real;

//    enum { strain_size = TStrain::strain_size };
//    enum { order = TStrain::order };
//    enum { spatial_dimensions = TStrain::spatial_dimensions };

//    typedef Mat<strain_size,strain_size,Real> StrainMat;  ///< Matrix representing a strain
//    typedef typename TStrain::StrainVec StrainVec;  ///< Vec representing a strain (Voigt notation)

//    static const bool constant = true;


//    /**
//    Mapping:   ADDITION       -> \f$ E_elastic = E_total - E_plastic \f$
//               MULTIPLICATION -> \f$ E_elastic = E_total * E_plastic^-1 \f$
//    Jacobian:                    \f$  dE = Id \f$
//    */


//    InCoord _plasticStrain;

//    void reset()
//    {
//        _plasticStrain.clear();
//    }


//    void addapply( OutCoord& /*result*/, const InCoord& /*data*/ ) {}

//    void addapply_multiplication( OutCoord& result, const InCoord& data, Real max, Real squaredYield, Real creep )
//    {
//        // eventually remove a part of the strain to simulate plasticity

//        // could be optimized by storing the computation of the previous time step
//        StrainMat plasticStrainMat = PrincipalStretchesToMat( _plasticStrain.getStrain() ) + StrainMat::s_identity;
//        StrainMat plasticStrainMatInverse; plasticStrainMatInverse.invert( plasticStrainMat );

//        // elasticStrain = totalStrain * plasticStrain^-1
//        StrainMat elasticStrainMat = ( PrincipalStretchesToMat( data.getStrain() ) + StrainMat::s_identity ) * plasticStrainMatInverse;
//        StrainVec elasticStrainVec = MatToPrincipalStretches( elasticStrainMat - StrainMat::s_identity );

//        // if( ||elasticStrain||  > c_yield ) plasticStrain += dt * c_creep * dt * elasticStrain
//        if( elasticStrainVec.norm2() > squaredYield )
//            _plasticStrain.getStrain() += creep * elasticStrainVec;

//        // if( ||plasticStrain|| > c_max ) plasticStrain *= c_max / ||plasticStrain||
//        Real plasticStrainNorm2 = _plasticStrain.getStrain().norm2();
//        if( plasticStrainNorm2 > max*max )
//            _plasticStrain.getStrain() *= max / helper::rsqrt( plasticStrainNorm2 );

//        plasticStrainMat = PrincipalStretchesToMat( _plasticStrain.getStrain() ) + StrainMat::s_identity;

//        // remaining elasticStrain = totalStrain * plasticStrain^-1
//        plasticStrainMatInverse.invert( plasticStrainMat );
//        elasticStrainMat = ( PrincipalStretchesToMat( data.getStrain() ) + StrainMat::s_identity ) * plasticStrainMatInverse;
//        elasticStrainVec = MatToPrincipalStretches( elasticStrainMat - StrainMat::s_identity );

//        result.getStrain() += elasticStrainVec;
//    }

//    void addapply_addition( OutCoord& result, const InCoord& data, Real max, Real squaredYield, Real creep )
//    {
//        // eventually remove a part of the strain to simulate plasticity

//        // elasticStrain = totalStrain - plasticStrain
//        InCoord elasticStrain = data - _plasticStrain;

//        // if( ||elasticStrain||  > c_yield ) plasticStrain += dt * c_creep * dt * elasticStrain
//        if( elasticStrain.getStrain().norm2() > squaredYield )
//            _plasticStrain += elasticStrain * creep;

//        // if( ||plasticStrain|| > c_max ) plasticStrain *= c_max / ||plasticStrain||
//        Real plasticStrainNorm2 = _plasticStrain.getStrain().norm2();
//        if( plasticStrainNorm2 > max*max )
//            _plasticStrain.getStrain() *= max / helper::rsqrt( plasticStrainNorm2 );

//        // remaining elasticStrain = totatStrain - plasticStrain
//        elasticStrain = data - _plasticStrain;

//        result += elasticStrain;
//    }

//    void addmult( OutDeriv& result,const InDeriv& data )
//    {
//        result += data;
//    }

//    void addMultTranspose( InDeriv& result, const OutDeriv& data )
//    {
//        result += data;
//    }

//    MatBlock getJ()
//    {
//        return MatBlock::Identity();
//    }

//    KBlock getK(const OutDeriv& /*childForce*/, bool=false)
//    {
//        return KBlock();
//    }

//    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const SReal& /*kfactor */)
//    {
//    }

//}; // class PlasticStrainJacobianBlock_U
//template<typename Real> class PlasticStrainJacobianBlock<U331(Real)>: public PlasticStrainJacobianBlock_U<3,3,0,Real> {};
//template<typename Real> class PlasticStrainJacobianBlock<U321(Real)>: public PlasticStrainJacobianBlock_U<3,2,0,Real> {};





} // namespace defaulttype
} // namespace sofa



#endif // FLEXIBLE_PlasticStrainJacobianBlock_H
