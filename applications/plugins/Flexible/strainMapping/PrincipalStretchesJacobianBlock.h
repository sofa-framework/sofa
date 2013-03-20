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
#ifndef FLEXIBLE_PrincipalStretchesJacobianBlock_H
#define FLEXIBLE_PrincipalStretchesJacobianBlock_H

#include "../BaseJacobian.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

#include <sofa/helper/decompose.h>

#include <sofa/helper/MatEigen.h>


namespace sofa
{

namespace defaulttype
{


//////////////////////////////////////////////////////////////////////////////////
//// default implementation for strain matrices  F -> D, F -> E (@warning for E, asStrain should be always true)
//////////////////////////////////////////////////////////////////////////////////


template<class TIn, class TOut>
class PrincipalStretchesJacobianBlock : public BaseJacobianBlock<TIn,TOut>
{
public:

    typedef TIn In;
    typedef TOut Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    typedef typename In::Frame Frame;  ///< Matrix representing a deformation gradient
    typedef typename Out::StrainVec StrainVec;  ///< Vec representing a strain
    enum { material_dimensions = In::material_dimensions };
    enum { spatial_dimensions = In::spatial_dimensions };
    enum { strain_size = Out::strain_size };
    enum { order = Out::order };
    enum { frame_size = spatial_dimensions*material_dimensions };

    typedef Mat<material_dimensions,material_dimensions,Real> MaterialMaterialMat;
    typedef Mat<spatial_dimensions,material_dimensions,Real> SpatialMaterialMat;

    /**
    Mapping:   \f$ E = Ut.F.V\f$
               \f$ E_k = Ut.F_k.V\f$
    where:  U/V are the spatial and material rotation parts of F and E is diagonal
    Jacobian:  \f$  dE = Ut.dF.V \f$ Note that dE is not diagonal
               \f$  dE_k = Ut.dF_k.V \f$
    */

    static const bool constantJ = false;

    SpatialMaterialMat _U;  ///< Spatial Rotation
    MaterialMaterialMat _V; ///< Material Rotation

    Mat<frame_size,frame_size,Real> _dUOverdF;
    Mat<material_dimensions*material_dimensions,frame_size,Real> _dVOverdF;

    bool _degenerated;

    bool _asStrain;


    void addapply( OutCoord& result, const InCoord& data )
    {
        Vec<material_dimensions,Real> S; // principal stretches
        _degenerated = helper::Decompose<Real>::SVD_stable( data.getF(), _U, S, _V );

        if( !_degenerated ) helper::Decompose<Real>::SVDGradient_dUdVOverdM( _U, S, _V, _dUOverdF, _dVOverdF );

        // order 0
        if( _asStrain )
        {
            for( int i=0 ; i<material_dimensions ; ++i )
                result.getStrain()[i] += S[i] - (Real)1; // principal stretches - 1 = diagonalized lagrangian strain
        }
        else
        {
            for( int i=0 ; i<material_dimensions ; ++i )
            {
                if( S[i]<0.6 ) S[i]=0.6; // common hack to ensure stability (J=detF=S[0]*S[1]*S[2] not too small) @todo maybe this should be a parameter?
                result.getStrain()[i] += S[i];
            }
        }

        if( order > 0 )
        {
            // order 1
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                result.getStrainGradient(k) += StrainMatToVoigt( cauchyStrainTensor( _U.multTranspose( data.getGradientF( k ) * _V ) ) );
            }
        }
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        //order 0
        result.getStrain() = StrainMatToVoigt( _U.multTranspose( data.getF() * _V ) );

        if( order > 0 )
        {
            // order 1
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                result.getStrainGradient(k) += StrainMatToVoigt( _U.multTranspose( data.getGradientF(k) * _V ) );
            }
        }
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        // order 0
        result.getF() += _U * StressVoigtToMat( data.getStrain() ).multTransposed( _V );

        if( order > 0 )
        {
            // order 1
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                result.getGradientF(k) += _U * StressVoigtToMat( data.getStrainGradient(k) ).multTransposed( _V );
            }
        }
    }

    // TODO requires to write (Ut.dp.V) as a matrix-vector product J.dp
    MatBlock getJ()
    {
        return MatBlock();
    }


    // TODO requires to write (dU/dp.dp.fc.V+U.fc.dV/dp.dp) as a matrix-vector product K.dp
    KBlock getK(const OutDeriv& /*childForce*/)
    {
        return KBlock();
    }

    void addDForce( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )
    {
        if( _degenerated ) return;

        SpatialMaterialMat dU;
        MaterialMaterialMat dV;

        // order 0
        //helper::Decompose<Real>::SVDGradient_dUdV( _U, _S, _V, dx.getF(), dU, dV );
        for( int k=0 ; k<spatial_dimensions ; ++k ) // line of df
            for( int l=0 ; l<material_dimensions ; ++l ) // col of df
                for( int j=0 ; j<material_dimensions ; ++j ) // col of dU & dV
                {
                    for( int i=0 ; i<spatial_dimensions ; ++i ) // line of dU
                        dU[i][j] += _dUOverdF[i*material_dimensions+j][k*material_dimensions+l] * dx.getF()[k][l];

                    for( int i=0 ; i<material_dimensions ; ++i ) // line of dV
                        dV[i][j] += _dVOverdF[i*material_dimensions+j][k*material_dimensions+l] * dx.getF()[k][l];
                }

        df.getF() += dU * StressVoigtToMat( childForce.getStrain() ) * _V * kfactor;
        df.getF() += _U * StressVoigtToMat( childForce.getStrain() ) * dV * kfactor;

        if( order > 0 )
        {
            // order 1
            // TODO
            /*for(unsigned int g=0;g<spatial_dimensions;g++)
            {
                for( int k=0 ; k<spatial_dimensions ; ++k ) // line of df
                for( int l=0 ; l<material_dimensions ; ++l ) // col of df
                for( int j=0 ; j<material_dimensions ; ++j ) // col of dU & dV
                {
                    for( int i=0 ; i<spatial_dimensions ; ++i ) // line of dU
                        dU[i][j] += _dUOverdF[i*material_dimensions+j][k*material_dimensions+l] * dx.getGradientF(g)[k][l];

                    for( int i=0 ; i<material_dimensions ; ++i ) // line of dV
                        dV[i][j] += _dVOverdF[i*material_dimensions+j][k*material_dimensions+l] * dx.getGradientF(g)[k][l];
                }

                df.getGradientF(g) += dU * StressVoigtToMat( childForce.getStrainGradient(g) ) * _V * kfactor;
                df.getGradientF(g) += _U * StressVoigtToMat( childForce.getStrainGradient(g) ) * dV * kfactor;
            }*/
        }
    }
};





//////////////////////////////////////////////////////////////////////////////////
////  F -> U
//////////////////////////////////////////////////////////////////////////////////

/** Template class used to implement one jacobian block for PrincipalStretchesMapping*/
template<class InReal, class OutReal, int MaterialDimension>
class PrincipalStretchesJacobianBlock< DefGradientTypes<3,MaterialDimension,0,InReal>, PrincipalStretchesStrainTypes<3,MaterialDimension,0,OutReal> >
           : public BaseJacobianBlock< DefGradientTypes<3,MaterialDimension,0,InReal>, PrincipalStretchesStrainTypes<3,MaterialDimension,0,OutReal> >
{
public:

    typedef DefGradientTypes<3,MaterialDimension,0,InReal> In;
    typedef PrincipalStretchesStrainTypes<3,MaterialDimension,0,OutReal> Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    typedef typename In::Frame Frame;  ///< Matrix representing a deformation gradient
    typedef typename Out::StrainVec StrainVec;  ///< Vec representing a strain
    enum { material_dimensions = In::material_dimensions };
    enum { spatial_dimensions = In::spatial_dimensions };
    enum { strain_size = Out::strain_size };
    enum { frame_size = spatial_dimensions*material_dimensions };

    typedef Mat<material_dimensions,material_dimensions,Real> MaterialMaterialMat;
    typedef Mat<spatial_dimensions,material_dimensions,Real> SpatialMaterialMat;

    /**
    Mapping:   \f$ E = Ut.F.V\f$
    where:  U/V are the spatial and material rotation parts of F and E is diagonal
    Jacobian:    \f$  dE = Ut.dF.V \f$ Note that dE is still diagonal (no anisotropy possible)
    Note that superior orders should not be possible because the gradients E_k are not necessary diagonal -> DiagonalStrain (including non-diagonal terms) can be used instead
    */

    static const bool constantJ = false;

    SpatialMaterialMat _U;  ///< Spatial Rotation
    MaterialMaterialMat _V; ///< Material Rotation


    MatBlock _J;

    Mat<frame_size,frame_size,Real> _dUOverdF;
    Mat<material_dimensions*material_dimensions,frame_size,Real> _dVOverdF;

    bool _degenerated;

    bool _asStrain;

    void addapply( OutCoord& result, const InCoord& data )
    {
        StrainVec S; // principal stretches
        _degenerated = helper::Decompose<Real>::SVD_stable( data.getF(), _U, S, _V );

        if( !_degenerated ) helper::Decompose<Real>::SVDGradient_dUdVOverdM( _U, S, _V, _dUOverdF, _dVOverdF );

        if( _asStrain )
        {
            for( int i=0 ; i<material_dimensions ; ++i )
                result.getStrain()[i] += S[i] - (Real)1; // principal stretches - 1 = diagonalized lagrangian strain
        }
        else
        {
            for( int i=0 ; i<material_dimensions ; ++i )
            {
                if( S[i]<0.6 ) S[i]=0.6; // common hack to ensure stability (J=detF=S[0]*S[1]*S[2] not too small) @todo maybe this should be a parameter?
                result.getStrain()[i] += S[i];
            }
        }

        computeJ();
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        for( int i=0 ; i<spatial_dimensions ; ++i )
            for( int j=0 ; j<material_dimensions ; ++j )
                for( int k=0 ; k<material_dimensions ; ++k )
                    result.getStrain()[k] += _J[k][i*material_dimensions+j] * data.getF()[i][j];
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        for( int i=0 ; i<spatial_dimensions ; ++i )
            for( int j=0 ; j<material_dimensions ; ++j )
                for( int k=0 ; k<material_dimensions ; ++k )
                    result.getF()[i][j] += _J[k][i*material_dimensions+j] * data.getStrain()[k];
    }

    void computeJ()
    {
        for( int i=0 ; i<spatial_dimensions ; ++i )
            for( int j=0 ; j<material_dimensions ; ++j )
                for( int k=0 ; k<material_dimensions ; ++k )
                    _J[k][i*material_dimensions+j] = _U[i][k]*_V[j][k];
    }

    MatBlock getJ()
    {
        return _J;
    }

    // TODO requires to write (dU/dp.dp.fc.V+U.fc.dV/dp.dp) as a matrix-vector product K.dp
    KBlock getK( const OutDeriv& /*childForce*/ )
    {
        return KBlock();
    }

    void addDForce( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )
    {
        if( _degenerated ) return;

        SpatialMaterialMat dU;
        MaterialMaterialMat dV;

        for( int k=0 ; k<spatial_dimensions ; ++k ) // line of df
            for( int l=0 ; l<material_dimensions ; ++l ) // col of df
                for( int j=0 ; j<material_dimensions ; ++j ) // col of dU & dV
                {
                    for( int i=0 ; i<spatial_dimensions ; ++i ) // line of dU
                        dU[i][j] += _dUOverdF[i*material_dimensions+j][k*material_dimensions+l] * dx.getF()[k][l];

                    for( int i=0 ; i<material_dimensions ; ++i ) // line of dV
                        dV[i][j] += _dVOverdF[i*material_dimensions+j][k*material_dimensions+l] * dx.getF()[k][l];
                }

        df.getF() += dU.multDiagonal( childForce.getStrain() ) * _V * kfactor;
        df.getF() += _U.multDiagonal( childForce.getStrain() ) * dV * kfactor;
    }
};






} // namespace defaulttype
} // namespace sofa



#endif // FLEXIBLE_PrincipalStretchesJacobianBlock_H
