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

template<class TIn, class TOut>
class PrincipalStretchesJacobianBlock : public BaseJacobianBlock<TIn,TOut>
{
};

//////////////////////////////////////////////////////////////////////////////////
//// default implementation for strain matrices  F -> D
//////////////////////////////////////////////////////////////////////////////////


//template<class TIn, class TOut>
//class PrincipalStretchesJacobianBlock : public BaseJacobianBlock<TIn,TOut>
//{
//public:

//    typedef TIn In;
//    typedef TOut Out;

//    typedef BaseJacobianBlock<In,Out> Inherit;
//    typedef typename Inherit::InCoord InCoord;
//    typedef typename Inherit::InDeriv InDeriv;
//    typedef typename Inherit::OutCoord OutCoord;
//    typedef typename Inherit::OutDeriv OutDeriv;
//    typedef typename Inherit::MatBlock MatBlock;
//    typedef typename Inherit::KBlock KBlock;
//    typedef typename Inherit::Real Real;

//    typedef typename In::Frame Frame;  ///< Matrix representing a deformation gradient
//    typedef typename Out::StrainVec StrainVec;  ///< Vec representing a strain
//    enum { material_dimensions = In::material_dimensions };
//    enum { spatial_dimensions = In::spatial_dimensions };
//    enum { strain_size = Out::strain_size };
//    enum { order = Out::order };
//    enum { frame_size = spatial_dimensions*material_dimensions };

//    typedef Mat<material_dimensions,material_dimensions,Real> MaterialMaterialMat;
//    typedef Mat<spatial_dimensions,material_dimensions,Real> SpatialMaterialMat;

//    /**
//    Mapping:   \f$ E = Ut.F.V\f$
//               \f$ E_k = Ut.F_k.V\f$
//    where:  U/V are the spatial and material rotation parts of F and E is diagonal
//    Jacobian:  \f$  dE = Ut.dF.V \f$ Note that dE is not diagonal
//               \f$  dE_k = Ut.dF_k.V \f$
//    */

//    static const bool constant = false;

//    SpatialMaterialMat _U;  ///< Spatial Rotation
//    MaterialMaterialMat _V; ///< Material Rotation

//    Mat<frame_size,frame_size,Real> _dUOverdF;
//    Mat<material_dimensions*material_dimensions,frame_size,Real> _dVOverdF;

//    bool _degenerated;

//    bool _asStrain;
//    Real _threshold;

//    PrincipalStretchesJacobianBlock() : _asStrain(false), _threshold(-std::numeric_limits<Real>::max()) {}

//    void init( bool asStrain, Real threshold/*, bool*/ )
//    {
//        _asStrain = asStrain;
//        _threshold = threshold;
//    }

//    void addapply( OutCoord& result, const InCoord& data )
//    {
//        Vec<material_dimensions,Real> S; // principal stretches
//        _degenerated = helper::Decompose<Real>::SVD_stable( data.getF(), _U, S, _V );

//        // order 0
//        if( _asStrain )
//        {
//            for( int i=0 ; i<material_dimensions ; ++i )
//                result.getStrain()[i] += S[i] - (Real)1; // principal stretches - 1 = diagonalized lagrangian strain
//        }
//        else
//        {
//            for( int i=0 ; i<material_dimensions ; ++i )
//            {
//                if( S[i]<_threshold ) S[i]=_threshold; // common hack to ensure stability (J=detF=S[0]*S[1]*S[2] not too small)
//                result.getStrain()[i] += S[i];
//            }
//        }

//        if( !_degenerated ) _degenerated = !helper::Decompose<Real>::SVDGradient_dUdVOverdM( _U, S, _V, _dUOverdF, _dVOverdF );

//        if( order > 0 )
//        {
//            // order 1
//            for(unsigned int k=0; k<spatial_dimensions; k++)
//            {
//                result.getStrainGradient(k) += StrainMatToVoigt( cauchyStrainTensor( _U.multTranspose( data.getGradientF( k ) * _V ) ) );
//            }
//        }
//    }

//    void addmult( OutDeriv& result,const InDeriv& data )
//    {
//        //order 0
//        result.getStrain() += StrainMatToVoigt( _U.multTranspose( data.getF() * _V ) );

//        if( order > 0 )
//        {
//            // order 1
//            for(unsigned int k=0; k<spatial_dimensions; k++)
//            {
//                result.getStrainGradient(k) += StrainMatToVoigt( _U.multTranspose( data.getGradientF(k) * _V ) );
//            }
//        }
//    }

//    void addMultTranspose( InDeriv& result, const OutDeriv& data )
//    {
//        // order 0
//        result.getF() += _U * StressVoigtToMat( data.getStrain() ).multTransposed( _V );

//        if( order > 0 )
//        {
//            // order 1
//            for(unsigned int k=0; k<spatial_dimensions; k++)
//            {
//                result.getGradientF(k) += _U * StressVoigtToMat( data.getStrainGradient(k) ).multTransposed( _V );
//            }
//        }
//    }

//    // TODO requires to write (Ut.dp.V) as a matrix-vector product J.dp
//    MatBlock getJ()
//    {
//        return MatBlock();
//    }


//    // TODO requires to write (dU/dp.dp.fc.V+U.fc.dV/dp.dp) as a matrix-vector product K.dp
//    KBlock getK(const OutDeriv& /*childForce*/, bool=false)
//    {
//        return KBlock();
//    }

//    void addDForce( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const SReal& kfactor )
//    {
//        if( _degenerated ) return;

//        SpatialMaterialMat dU;
//        MaterialMaterialMat dV;

//        // order 0
////        helper::Decompose<Real>::SVDGradient_dUdV( _U, _S, _V, dx.getF(), dU, dV );
//        for( int k=0 ; k<spatial_dimensions ; ++k ) // line of df
//            for( int l=0 ; l<material_dimensions ; ++l ) // col of df
//                for( int j=0 ; j<material_dimensions ; ++j ) // col of dU & dV
//                {
//                    for( int i=0 ; i<spatial_dimensions ; ++i ) // line of dU
//                        dU[i][j] += _dUOverdF[i*material_dimensions+j][k*material_dimensions+l] * dx.getF()[k][l];

//                    for( int i=0 ; i<material_dimensions ; ++i ) // line of dV
//                        dV[i][j] += _dVOverdF[i*material_dimensions+j][k*material_dimensions+l] * dx.getF()[k][l];
//                }

//        df.getF() += dU * StressVoigtToMat( childForce.getStrain() ) * _V * kfactor;
//        df.getF() += _U * StressVoigtToMat( childForce.getStrain() ) * dV * kfactor;


//        if( order > 0 )
//        {
//            // order 1
//            // TODO
//            /*for(unsigned int g=0;g<spatial_dimensions;g++)
//            {
//                for( int k=0 ; k<spatial_dimensions ; ++k ) // line of df
//                for( int l=0 ; l<material_dimensions ; ++l ) // col of df
//                for( int j=0 ; j<material_dimensions ; ++j ) // col of dU & dV
//                {
//                    for( int i=0 ; i<spatial_dimensions ; ++i ) // line of dU
//                        dU[i][j] += _dUOverdF[i*material_dimensions+j][k*material_dimensions+l] * dx.getGradientF(g)[k][l];

//                    for( int i=0 ; i<material_dimensions ; ++i ) // line of dV
//                        dV[i][j] += _dVOverdF[i*material_dimensions+j][k*material_dimensions+l] * dx.getGradientF(g)[k][l];
//                }

//                df.getGradientF(g) += dU * StressVoigtToMat( childForce.getStrainGradient(g) ) * _V * kfactor;
//                df.getGradientF(g) += _U * StressVoigtToMat( childForce.getStrainGradient(g) ) * dV * kfactor;
//            }*/
//        }
//    }
//};





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

    static const bool constant = false;

    SpatialMaterialMat _U;  ///< Spatial Rotation
    MaterialMaterialMat _V; ///< Material Rotation


    MatBlock _J;

    Mat<frame_size,frame_size,Real> _dUOverdF;
    Mat<material_dimensions*material_dimensions,frame_size,Real> _dVOverdF;

    bool _degenerated;

    bool _asStrain;
    Real _threshold;
    bool _PSDStabilization;

    PrincipalStretchesJacobianBlock() : _asStrain(false), _threshold(-std::numeric_limits<Real>::max()) {}

    void init( bool asStrain, Real threshold, bool PSDStabilization )
    {
        _asStrain = asStrain;
        _threshold = threshold;
        _PSDStabilization = PSDStabilization;
    }

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
                if( S[i]<_threshold) S[i]=_threshold; // common hack to ensure stability (J=detF=S[0]*S[1]*S[2] not too small)
                result.getStrain()[i] += S[i];
            }
        }

        computeJ();
    }

    void addmult( OutDeriv& result, const InDeriv& data )
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

    // write (dU/dp.dp.fc.V+U.fc.dV/dp.dp) as a matrix-vector product K.dp
    KBlock getK( const OutDeriv& childForce, bool=false )
    {
        KBlock K;

        if( _degenerated ) return K;

        compute_K( K, childForce );
        return K;
    }

    void addDForce( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const SReal& kfactor )
    {
        if( _degenerated ) return;

        if( _PSDStabilization )
        {
            // to be able to perform the PSD stabilization, the stiffness matrix needs to be built
            KBlock K;
            compute_K( K, childForce );
            df.getVec() += K * dx.getVec() * kfactor;
        }
        else
        {
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
    }



    /// @ todo find a general algorithm to compute K for any dimensions
    // see the maple file doc/principalStretches_geometricStiffnessMatrix.mw
    void compute_K( Mat<9,9,Real>& K, const OutDeriv& childForce ) // for spatial=3 material=3
    {
        K[0][0] = _dUOverdF[0][0] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[1][0] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[2][0] * childForce.getStrain()[2] * _V[2][0] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[0][0] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[3][0] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[6][0];
        K[0][1] = _dUOverdF[0][0] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[1][0] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[2][0] * childForce.getStrain()[2] * _V[2][1] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[1][0] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[4][0] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[7][0];
        K[0][2] = _dUOverdF[0][0] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[1][0] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[2][0] * childForce.getStrain()[2] * _V[2][2] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[2][0] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[5][0] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[8][0];
        K[0][3] = _dUOverdF[3][0] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[4][0] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[5][0] * childForce.getStrain()[2] * _V[2][0] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[0][0] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[3][0] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[6][0];
        K[0][4] = _dUOverdF[3][0] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[4][0] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[5][0] * childForce.getStrain()[2] * _V[2][1] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[1][0] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[4][0] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[7][0];
        K[0][5] = _dUOverdF[3][0] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[4][0] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[5][0] * childForce.getStrain()[2] * _V[2][2] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[2][0] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[5][0] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[8][0];
        K[0][6] = _dUOverdF[6][0] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[7][0] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[8][0] * childForce.getStrain()[2] * _V[2][0] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[0][0] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[3][0] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[6][0];
        K[0][7] = _dUOverdF[6][0] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[7][0] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[8][0] * childForce.getStrain()[2] * _V[2][1] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[1][0] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[4][0] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[7][0];
        K[0][8] = _dUOverdF[6][0] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[7][0] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[8][0] * childForce.getStrain()[2] * _V[2][2] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[2][0] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[5][0] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[8][0];
        K[1][0] = K[0][1]; //_dUOverdF[0][1] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[1][1] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[2][1] * childForce.getStrain()[2] * _V[2][0] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[0][1] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[3][1] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[6][1];
        K[1][1] = _dUOverdF[0][1] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[1][1] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[2][1] * childForce.getStrain()[2] * _V[2][1] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[1][1] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[4][1] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[7][1];
        K[1][2] = _dUOverdF[0][1] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[1][1] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[2][1] * childForce.getStrain()[2] * _V[2][2] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[2][1] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[5][1] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[8][1];
        K[1][3] = _dUOverdF[3][1] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[4][1] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[5][1] * childForce.getStrain()[2] * _V[2][0] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[0][1] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[3][1] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[6][1];
        K[1][4] = _dUOverdF[3][1] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[4][1] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[5][1] * childForce.getStrain()[2] * _V[2][1] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[1][1] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[4][1] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[7][1];
        K[1][5] = _dUOverdF[3][1] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[4][1] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[5][1] * childForce.getStrain()[2] * _V[2][2] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[2][1] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[5][1] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[8][1];
        K[1][6] = _dUOverdF[6][1] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[7][1] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[8][1] * childForce.getStrain()[2] * _V[2][0] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[0][1] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[3][1] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[6][1];
        K[1][7] = _dUOverdF[6][1] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[7][1] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[8][1] * childForce.getStrain()[2] * _V[2][1] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[1][1] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[4][1] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[7][1];
        K[1][8] = _dUOverdF[6][1] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[7][1] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[8][1] * childForce.getStrain()[2] * _V[2][2] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[2][1] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[5][1] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[8][1];
        K[2][0] = K[0][2]; //_dUOverdF[0][2] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[1][2] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[2][2] * childForce.getStrain()[2] * _V[2][0] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[0][2] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[3][2] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[6][2];
        K[2][1] = K[1][2]; //_dUOverdF[0][2] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[1][2] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[2][2] * childForce.getStrain()[2] * _V[2][1] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[1][2] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[4][2] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[7][2];
        K[2][2] = _dUOverdF[0][2] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[1][2] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[2][2] * childForce.getStrain()[2] * _V[2][2] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[2][2] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[5][2] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[8][2];
        K[2][3] = _dUOverdF[3][2] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[4][2] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[5][2] * childForce.getStrain()[2] * _V[2][0] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[0][2] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[3][2] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[6][2];
        K[2][4] = _dUOverdF[3][2] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[4][2] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[5][2] * childForce.getStrain()[2] * _V[2][1] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[1][2] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[4][2] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[7][2];
        K[2][5] = _dUOverdF[3][2] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[4][2] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[5][2] * childForce.getStrain()[2] * _V[2][2] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[2][2] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[5][2] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[8][2];
        K[2][6] = _dUOverdF[6][2] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[7][2] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[8][2] * childForce.getStrain()[2] * _V[2][0] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[0][2] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[3][2] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[6][2];
        K[2][7] = _dUOverdF[6][2] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[7][2] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[8][2] * childForce.getStrain()[2] * _V[2][1] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[1][2] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[4][2] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[7][2];
        K[2][8] = _dUOverdF[6][2] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[7][2] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[8][2] * childForce.getStrain()[2] * _V[2][2] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[2][2] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[5][2] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[8][2];
        K[3][0] = K[0][3]; //_dUOverdF[0][3] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[1][3] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[2][3] * childForce.getStrain()[2] * _V[2][0] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[0][3] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[3][3] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[6][3];
        K[3][1] = K[1][3]; //_dUOverdF[0][3] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[1][3] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[2][3] * childForce.getStrain()[2] * _V[2][1] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[1][3] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[4][3] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[7][3];
        K[3][2] = K[2][3]; //_dUOverdF[0][3] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[1][3] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[2][3] * childForce.getStrain()[2] * _V[2][2] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[2][3] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[5][3] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[8][3];
        K[3][3] = _dUOverdF[3][3] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[4][3] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[5][3] * childForce.getStrain()[2] * _V[2][0] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[0][3] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[3][3] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[6][3];
        K[3][4] = _dUOverdF[3][3] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[4][3] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[5][3] * childForce.getStrain()[2] * _V[2][1] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[1][3] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[4][3] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[7][3];
        K[3][5] = _dUOverdF[3][3] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[4][3] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[5][3] * childForce.getStrain()[2] * _V[2][2] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[2][3] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[5][3] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[8][3];
        K[3][6] = _dUOverdF[6][3] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[7][3] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[8][3] * childForce.getStrain()[2] * _V[2][0] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[0][3] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[3][3] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[6][3];
        K[3][7] = _dUOverdF[6][3] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[7][3] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[8][3] * childForce.getStrain()[2] * _V[2][1] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[1][3] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[4][3] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[7][3];
        K[3][8] = _dUOverdF[6][3] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[7][3] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[8][3] * childForce.getStrain()[2] * _V[2][2] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[2][3] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[5][3] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[8][3];
        K[4][0] = K[0][4]; //_dUOverdF[0][4] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[1][4] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[2][4] * childForce.getStrain()[2] * _V[2][0] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[0][4] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[3][4] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[6][4];
        K[4][1] = K[1][4]; //_dUOverdF[0][4] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[1][4] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[2][4] * childForce.getStrain()[2] * _V[2][1] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[1][4] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[4][4] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[7][4];
        K[4][2] = K[2][4]; //_dUOverdF[0][4] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[1][4] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[2][4] * childForce.getStrain()[2] * _V[2][2] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[2][4] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[5][4] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[8][4];
        K[4][3] = K[3][4]; //_dUOverdF[3][4] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[4][4] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[5][4] * childForce.getStrain()[2] * _V[2][0] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[0][4] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[3][4] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[6][4];
        K[4][4] = _dUOverdF[3][4] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[4][4] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[5][4] * childForce.getStrain()[2] * _V[2][1] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[1][4] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[4][4] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[7][4];
        K[4][5] = _dUOverdF[3][4] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[4][4] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[5][4] * childForce.getStrain()[2] * _V[2][2] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[2][4] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[5][4] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[8][4];
        K[4][6] = _dUOverdF[6][4] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[7][4] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[8][4] * childForce.getStrain()[2] * _V[2][0] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[0][4] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[3][4] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[6][4];
        K[4][7] = _dUOverdF[6][4] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[7][4] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[8][4] * childForce.getStrain()[2] * _V[2][1] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[1][4] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[4][4] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[7][4];
        K[4][8] = _dUOverdF[6][4] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[7][4] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[8][4] * childForce.getStrain()[2] * _V[2][2] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[2][4] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[5][4] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[8][4];
        K[5][0] = K[0][5]; //_dUOverdF[0][5] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[1][5] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[2][5] * childForce.getStrain()[2] * _V[2][0] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[0][5] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[3][5] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[6][5];
        K[5][1] = K[1][5]; //_dUOverdF[0][5] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[1][5] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[2][5] * childForce.getStrain()[2] * _V[2][1] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[1][5] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[4][5] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[7][5];
        K[5][2] = K[2][5]; //_dUOverdF[0][5] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[1][5] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[2][5] * childForce.getStrain()[2] * _V[2][2] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[2][5] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[5][5] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[8][5];
        K[5][3] = K[3][5]; //_dUOverdF[3][5] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[4][5] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[5][5] * childForce.getStrain()[2] * _V[2][0] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[0][5] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[3][5] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[6][5];
        K[5][4] = K[4][5]; //_dUOverdF[3][5] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[4][5] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[5][5] * childForce.getStrain()[2] * _V[2][1] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[1][5] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[4][5] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[7][5];
        K[5][5] = _dUOverdF[3][5] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[4][5] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[5][5] * childForce.getStrain()[2] * _V[2][2] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[2][5] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[5][5] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[8][5];
        K[5][6] = _dUOverdF[6][5] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[7][5] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[8][5] * childForce.getStrain()[2] * _V[2][0] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[0][5] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[3][5] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[6][5];
        K[5][7] = _dUOverdF[6][5] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[7][5] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[8][5] * childForce.getStrain()[2] * _V[2][1] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[1][5] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[4][5] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[7][5];
        K[5][8] = _dUOverdF[6][5] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[7][5] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[8][5] * childForce.getStrain()[2] * _V[2][2] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[2][5] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[5][5] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[8][5];
        K[6][0] = K[0][6]; //_dUOverdF[0][6] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[1][6] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[2][6] * childForce.getStrain()[2] * _V[2][0] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[0][6] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[3][6] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[6][6];
        K[6][1] = K[1][6]; //_dUOverdF[0][6] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[1][6] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[2][6] * childForce.getStrain()[2] * _V[2][1] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[1][6] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[4][6] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[7][6];
        K[6][2] = K[2][6]; //_dUOverdF[0][6] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[1][6] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[2][6] * childForce.getStrain()[2] * _V[2][2] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[2][6] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[5][6] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[8][6];
        K[6][3] = K[3][6]; //_dUOverdF[3][6] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[4][6] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[5][6] * childForce.getStrain()[2] * _V[2][0] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[0][6] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[3][6] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[6][6];
        K[6][4] = K[4][6]; //_dUOverdF[3][6] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[4][6] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[5][6] * childForce.getStrain()[2] * _V[2][1] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[1][6] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[4][6] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[7][6];
        K[6][5] = K[5][6]; //_dUOverdF[3][6] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[4][6] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[5][6] * childForce.getStrain()[2] * _V[2][2] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[2][6] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[5][6] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[8][6];
        K[6][6] = _dUOverdF[6][6] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[7][6] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[8][6] * childForce.getStrain()[2] * _V[2][0] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[0][6] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[3][6] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[6][6];
        K[6][7] = _dUOverdF[6][6] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[7][6] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[8][6] * childForce.getStrain()[2] * _V[2][1] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[1][6] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[4][6] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[7][6];
        K[6][8] = _dUOverdF[6][6] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[7][6] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[8][6] * childForce.getStrain()[2] * _V[2][2] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[2][6] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[5][6] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[8][6];
        K[7][0] = K[0][7]; //_dUOverdF[0][7] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[1][7] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[2][7] * childForce.getStrain()[2] * _V[2][0] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[0][7] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[3][7] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[6][7];
        K[7][1] = K[1][7]; //_dUOverdF[0][7] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[1][7] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[2][7] * childForce.getStrain()[2] * _V[2][1] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[1][7] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[4][7] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[7][7];
        K[7][2] = K[2][7]; //_dUOverdF[0][7] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[1][7] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[2][7] * childForce.getStrain()[2] * _V[2][2] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[2][7] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[5][7] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[8][7];
        K[7][3] = K[3][7]; //_dUOverdF[3][7] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[4][7] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[5][7] * childForce.getStrain()[2] * _V[2][0] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[0][7] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[3][7] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[6][7];
        K[7][4] = K[4][7]; //_dUOverdF[3][7] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[4][7] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[5][7] * childForce.getStrain()[2] * _V[2][1] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[1][7] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[4][7] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[7][7];
        K[7][5] = K[5][7]; //_dUOverdF[3][7] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[4][7] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[5][7] * childForce.getStrain()[2] * _V[2][2] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[2][7] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[5][7] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[8][7];
        K[7][6] = K[6][7]; //_dUOverdF[6][7] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[7][7] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[8][7] * childForce.getStrain()[2] * _V[2][0] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[0][7] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[3][7] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[6][7];
        K[7][7] = _dUOverdF[6][7] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[7][7] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[8][7] * childForce.getStrain()[2] * _V[2][1] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[1][7] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[4][7] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[7][7];
        K[7][8] = _dUOverdF[6][7] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[7][7] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[8][7] * childForce.getStrain()[2] * _V[2][2] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[2][7] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[5][7] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[8][7];
        K[8][0] = K[0][8]; //_dUOverdF[0][8] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[1][8] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[2][8] * childForce.getStrain()[2] * _V[2][0] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[0][8] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[3][8] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[6][8];
        K[8][1] = K[1][8]; //_dUOverdF[0][8] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[1][8] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[2][8] * childForce.getStrain()[2] * _V[2][1] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[1][8] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[4][8] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[7][8];
        K[8][2] = K[2][8]; //_dUOverdF[0][8] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[1][8] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[2][8] * childForce.getStrain()[2] * _V[2][2] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[2][8] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[5][8] + _U[0][2] * childForce.getStrain()[2] * _dVOverdF[8][8];
        K[8][3] = K[3][8]; //_dUOverdF[3][8] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[4][8] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[5][8] * childForce.getStrain()[2] * _V[2][0] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[0][8] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[3][8] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[6][8];
        K[8][4] = K[4][8]; //_dUOverdF[3][8] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[4][8] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[5][8] * childForce.getStrain()[2] * _V[2][1] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[1][8] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[4][8] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[7][8];
        K[8][5] = K[5][8]; //_dUOverdF[3][8] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[4][8] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[5][8] * childForce.getStrain()[2] * _V[2][2] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[2][8] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[5][8] + _U[1][2] * childForce.getStrain()[2] * _dVOverdF[8][8];
        K[8][6] = K[6][8]; //_dUOverdF[6][8] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[7][8] * childForce.getStrain()[1] * _V[1][0] + _dUOverdF[8][8] * childForce.getStrain()[2] * _V[2][0] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[0][8] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[3][8] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[6][8];
        K[8][7] = K[7][8]; //_dUOverdF[6][8] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[7][8] * childForce.getStrain()[1] * _V[1][1] + _dUOverdF[8][8] * childForce.getStrain()[2] * _V[2][1] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[1][8] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[4][8] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[7][8];
        K[8][8] = _dUOverdF[6][8] * childForce.getStrain()[0] * _V[0][2] + _dUOverdF[7][8] * childForce.getStrain()[1] * _V[1][2] + _dUOverdF[8][8] * childForce.getStrain()[2] * _V[2][2] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[2][8] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[5][8] + _U[2][2] * childForce.getStrain()[2] * _dVOverdF[8][8];

        if( _PSDStabilization )
        {
            // stabilized K by projecting sub-matrices to their closest PSD like in [Teran05]
            // warning the index order to build the sub-matrices is { 0,4,8,1,3,2,6,5,7 };
            helper::Decompose<Real>::NSDProjection( K[1][1], K[1][3], K[3][1], K[3][3] );
            helper::Decompose<Real>::NSDProjection( K[2][2], K[2][6], K[6][2], K[6][6] );
            helper::Decompose<Real>::NSDProjection( K[5][5], K[5][7], K[7][5], K[7][7] );
        }
    }


    void compute_K( Mat<6,6,Real>& K, const OutDeriv& childForce )  // for spatial=3 material=2
    {
        K[0][0] =  _dUOverdF[0][0] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[1][0] * childForce.getStrain()[1] * _V[1][0] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[0][0] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[2][0];
        K[0][1] =  _dUOverdF[0][0] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[1][0] * childForce.getStrain()[1] * _V[1][1] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[1][0] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[3][0];
        K[0][2] =  _dUOverdF[2][0] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[3][0] * childForce.getStrain()[1] * _V[1][0] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[0][0] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[2][0];
        K[0][3] =  _dUOverdF[2][0] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[3][0] * childForce.getStrain()[1] * _V[1][1] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[1][0] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[3][0];
        K[0][4] =  _dUOverdF[4][0] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[5][0] * childForce.getStrain()[1] * _V[1][0] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[0][0] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[2][0];
        K[0][5] =  _dUOverdF[4][0] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[5][0] * childForce.getStrain()[1] * _V[1][1] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[1][0] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[3][0];
        K[1][0] = K[0][1]; //_dUOverdF[0][1] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[1][1] * childForce.getStrain()[1] * _V[1][0] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[0][1] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[2][1];
        K[1][1] =  _dUOverdF[0][1] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[1][1] * childForce.getStrain()[1] * _V[1][1] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[1][1] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[3][1];
        K[1][2] =  _dUOverdF[2][1] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[3][1] * childForce.getStrain()[1] * _V[1][0] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[0][1] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[2][1];
        K[1][3] =  _dUOverdF[2][1] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[3][1] * childForce.getStrain()[1] * _V[1][1] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[1][1] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[3][1];
        K[1][4] =  _dUOverdF[4][1] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[5][1] * childForce.getStrain()[1] * _V[1][0] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[0][1] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[2][1];
        K[1][5] =  _dUOverdF[4][1] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[5][1] * childForce.getStrain()[1] * _V[1][1] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[1][1] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[3][1];
        K[2][0] = K[0][2]; //_dUOverdF[0][2] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[1][2] * childForce.getStrain()[1] * _V[1][0] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[0][2] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[2][2];
        K[2][1] = K[1][2]; //_dUOverdF[0][2] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[1][2] * childForce.getStrain()[1] * _V[1][1] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[1][2] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[3][2];
        K[2][2] =  _dUOverdF[2][2] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[3][2] * childForce.getStrain()[1] * _V[1][0] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[0][2] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[2][2];
        K[2][3] =  _dUOverdF[2][2] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[3][2] * childForce.getStrain()[1] * _V[1][1] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[1][2] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[3][2];
        K[2][4] =  _dUOverdF[4][2] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[5][2] * childForce.getStrain()[1] * _V[1][0] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[0][2] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[2][2];
        K[2][5] =  _dUOverdF[4][2] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[5][2] * childForce.getStrain()[1] * _V[1][1] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[1][2] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[3][2];
        K[3][0] = K[0][3]; //_dUOverdF[0][3] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[1][3] * childForce.getStrain()[1] * _V[1][0] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[0][3] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[2][3];
        K[3][1] = K[1][3]; //_dUOverdF[0][3] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[1][3] * childForce.getStrain()[1] * _V[1][1] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[1][3] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[3][3];
        K[3][2] = K[2][3]; //_dUOverdF[2][3] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[3][3] * childForce.getStrain()[1] * _V[1][0] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[0][3] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[2][3];
        K[3][3] =  _dUOverdF[2][3] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[3][3] * childForce.getStrain()[1] * _V[1][1] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[1][3] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[3][3];
        K[3][4] =  _dUOverdF[4][3] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[5][3] * childForce.getStrain()[1] * _V[1][0] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[0][3] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[2][3];
        K[3][5] =  _dUOverdF[4][3] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[5][3] * childForce.getStrain()[1] * _V[1][1] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[1][3] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[3][3];
        K[4][0] = K[0][4]; //_dUOverdF[0][4] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[1][4] * childForce.getStrain()[1] * _V[1][0] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[0][4] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[2][4];
        K[4][1] = K[1][4]; //_dUOverdF[0][4] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[1][4] * childForce.getStrain()[1] * _V[1][1] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[1][4] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[3][4];
        K[4][2] = K[2][4]; //_dUOverdF[2][4] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[3][4] * childForce.getStrain()[1] * _V[1][0] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[0][4] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[2][4];
        K[4][3] = K[3][4]; //_dUOverdF[2][4] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[3][4] * childForce.getStrain()[1] * _V[1][1] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[1][4] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[3][4];
        K[4][4] =  _dUOverdF[4][4] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[5][4] * childForce.getStrain()[1] * _V[1][0] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[0][4] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[2][4];
        K[4][5] =  _dUOverdF[4][4] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[5][4] * childForce.getStrain()[1] * _V[1][1] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[1][4] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[3][4];
        K[5][0] = K[0][5]; //_dUOverdF[0][5] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[1][5] * childForce.getStrain()[1] * _V[1][0] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[0][5] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[2][5];
        K[5][1] = K[1][5]; //_dUOverdF[0][5] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[1][5] * childForce.getStrain()[1] * _V[1][1] + _U[0][0] * childForce.getStrain()[0] * _dVOverdF[1][5] + _U[0][1] * childForce.getStrain()[1] * _dVOverdF[3][5];
        K[5][2] = K[2][5]; //_dUOverdF[2][5] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[3][5] * childForce.getStrain()[1] * _V[1][0] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[0][5] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[2][5];
        K[5][3] = K[3][5]; //_dUOverdF[2][5] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[3][5] * childForce.getStrain()[1] * _V[1][1] + _U[1][0] * childForce.getStrain()[0] * _dVOverdF[1][5] + _U[1][1] * childForce.getStrain()[1] * _dVOverdF[3][5];
        K[5][4] = K[4][5]; //_dUOverdF[4][5] * childForce.getStrain()[0] * _V[0][0] + _dUOverdF[5][5] * childForce.getStrain()[1] * _V[1][0] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[0][5] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[2][5];
        K[5][5] =  _dUOverdF[4][5] * childForce.getStrain()[0] * _V[0][1] + _dUOverdF[5][5] * childForce.getStrain()[1] * _V[1][1] + _U[2][0] * childForce.getStrain()[0] * _dVOverdF[1][5] + _U[2][1] * childForce.getStrain()[1] * _dVOverdF[3][5];
    }


};




} // namespace defaulttype
} // namespace sofa



#endif // FLEXIBLE_PrincipalStretchesJacobianBlock_H

