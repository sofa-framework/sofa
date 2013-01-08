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
#ifndef FLEXIBLE_CorotationalStrainJacobianBlock_INL
#define FLEXIBLE_CorotationalStrainJacobianBlock_INL


#include "CorotationalStrainJacobianBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

#include <sofa/helper/decompose.inl>

#include <sofa/helper/MatEigen.h>


namespace sofa
{

namespace defaulttype
{


//////////////////////////////////////////////////////////////////////////////////
////  macros
//////////////////////////////////////////////////////////////////////////////////
#define F321(type)  DefGradientTypes<3,2,0,type>
#define F311(type)  DefGradientTypes<3,1,0,type>
#define E321(type)  StrainTypes<3,2,0,type>
#define E311(type)  StrainTypes<3,1,0,type>

//////////////////////////////////////////////////////////////////////////////////
////  helpers
//////////////////////////////////////////////////////////////////////////////////


/// 3D->3D
template<typename Real>
bool computeQR( const Mat<3,3,Real> &f, Mat<3,3,Real> &r, Mat<3,3,Real> &s )
{
    bool degenerated = helper::Decompose<Real>::QRDecomposition_stable( f, r );

    Mat<3,3,Real> T = r.multTranspose( f ); // T = rt * f
    s = cauchyStrainTensor( T ); // s = ( T + Tt ) * 0.5

    return degenerated;
}

/// 3D->2D
template<typename Real>
void computeQR( const Mat<3,2,Real> &f, Mat<3,2,Real> &r, Mat<2,2,Real> &s )
{
    helper::Decompose<Real>::QRDecomposition_stable( f, r );

    Mat<2,2,Real> T = r.multTranspose( f ); // T = rt * f
    s = cauchyStrainTensor( T ); // s = ( T + Tt ) * 0.5
}


/// 3D->1D
template<typename Real>
void computeQR( const Mat<3,1,Real> &f, Mat<3,1,Real> &r, Mat<1,1,Real> &s )
{
    Real nrm = sqrt(f(0,0)*f(0,0)+f(1,0)*f(1,0)+f(2,0)*f(2,0));
    r(0,0)=f(0,0)/nrm;
    r(1,0)=f(1,0)/nrm;
    r(2,0)=f(2,0)/nrm;
    s(0,0) = nrm;
}

/// 3D->3D
template<typename Real>
bool computeSVD( const Mat<3,3,Real> &F, Mat<3,3,Real> &r, Mat<3,3,Real> &s, Mat<3,3,Real> &U, Vec<3,Real>& F_diag, Mat<3,3,Real> &V )
{
    bool degenerated = helper::Decompose<Real>::SVD_stable( F, U, F_diag, V );

    // the world rotation of the element based on the two rotations computed by the SVD (world and material space)
    r = U.multTransposed( V ); // r = U * Vt
    s = r.multTranspose( F ); // s = rt * F
    //s = V.multDiagonal( F_diag ).multTransposed( V ); // s = V * F_diag * Vt

    return degenerated;
}


/// 3D->2D
template<typename Real>
void computeSVD( const Mat<3,2,Real> &F, Mat<3,2,Real> &r, Mat<2,2,Real> &s, Mat<3,2,Real> &U, Vec<2,Real>& F_diag, Mat<2,2,Real> &V )
{
    helper::Decompose<Real>::SVD_stable( F, U, F_diag, V );

    r = U.multTransposed( V ); // r = U * Vt
    s = r.multTranspose( F ); // s = rt * F
}


////////////////////////////////////////////////////////


/// stuff needed to compute the geometric stiffness and that is different for each decomposition method -> use pointers and create variables only when needed
template<int material_dimension,int frame_size, class Real>
class CorotationalStrainJacobianBlockGeometricStiffnessData
{
    bool *_degenerated; ///< is the deformation gradient too flat or inverted?
    Mat<material_dimension,material_dimension,Real> *_matrix; ///< R^-1 for QR, G^-1 for polar
    Mat<frame_size,frame_size,Real> *_dROverdF; ///< dR/dF for SVD

public:
    CorotationalStrainJacobianBlockGeometricStiffnessData()
        : _degenerated(NULL)
        , _matrix(NULL)
        , _dROverdF(NULL)
    {}

    void init_small()
    {
        if( _matrix ) { delete _matrix; _matrix=NULL; }
        if( _dROverdF ) { delete _dROverdF; _dROverdF=NULL; }
        if( _degenerated ) { delete _degenerated; _degenerated=NULL; }
    }
    void init_qr( bool geometricStiffness )
    {
        if( _dROverdF ) { delete _dROverdF; _dROverdF=NULL; }

        if( geometricStiffness )
        {
            if( !_matrix ) _matrix = new Mat<material_dimension,material_dimension,Real>();
            if( !_degenerated ) _degenerated = new bool;
        }
        else
        {
            if( _matrix ) { delete _matrix; _matrix=NULL; }
            if( _degenerated ) { delete _degenerated; _degenerated=NULL; }
        }
    }
    void init_polar( bool geometricStiffness )
    {
        if( _dROverdF ) { delete _dROverdF; _dROverdF=NULL; }

        if( geometricStiffness )
        {
            if( !_matrix ) _matrix = new Mat<material_dimension,material_dimension,Real>();
            if( !_degenerated ) _degenerated = new bool;
        }
        else
        {
            if( _matrix ) { delete _matrix; _matrix=NULL; }
            if( _degenerated ) { delete _degenerated; _degenerated=NULL; }
        }
    }
    void init_svd( bool geometricStiffness )
    {
        if( _matrix ) { delete _matrix; _matrix=NULL; }

        if( geometricStiffness )
        {
            if( !_dROverdF ) _dROverdF = new Mat<frame_size,frame_size,Real>();
            if( !_degenerated ) _degenerated = new bool;
        }
        else
        {
            if( _dROverdF ) { delete _dROverdF; _dROverdF=NULL; }
            if( _degenerated ) { delete _degenerated; _degenerated=NULL; }
        }
    }


    inline bool& degenerated() const { return *_degenerated; }
    inline Mat<material_dimension,material_dimension,Real>* invT() const { return _matrix; } ///< for qr method
    inline Mat<material_dimension,material_dimension,Real>* invG() const { return _matrix; } ///< for polar method
    inline Mat<frame_size,frame_size,Real>* dROverdF() const { return _dROverdF; } ///< for SVD method

};



//////////////////////////////////////////////////////////////////////////////////
////  CorotationalStrainJacobianBlock
//////////////////////////////////////////////////////////////////////////////////

/** Template class used to implement one jacobian block for CorotationalStrainMapping */
template<class TIn, class TOut>
class CorotationalStrainJacobianBlock :
    public  BaseJacobianBlock< TIn, TOut >
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
    typedef typename Out::StrainMat StrainMat;  ///< Matrix representing a strain
    enum { material_dimensions = In::material_dimensions };
    enum { spatial_dimensions = In::spatial_dimensions };
    enum { strain_size = Out::strain_size };
    enum { order = Out::order };
    enum { frame_size = spatial_dimensions*material_dimensions };

    typedef Mat<spatial_dimensions,spatial_dimensions,Real> Affine;  ///< Matrix representing a linear spatial transformation

    /**
    Mapping:
        - \f$ E = [grad(R^T u)+grad(R^T u)^T ]/2 = [R^T F + F^T R ]/2 - I = D - I \f$
        - \f$ E_k = [R^T F_k + F_k^T R ]/2  \f$
    where:
        - _R/D are the rotational/skew symmetric parts of F=RD
        - _k denotes derivative with respect to spatial dimension k
    Jacobian:
        - \f$  dE = [R^T dF + dF^T R ]/2 \f$
        - \f$  dE_k = [R^T dF_k + dF_k^T R ]/2 \f$
      */


    static const bool constantJ = false;

    Affine _R;   ///< =  store rotational part of deformation gradient to compute J



    CorotationalStrainJacobianBlockGeometricStiffnessData<material_dimensions,frame_size,Real> _geometricStiffnessData; ///< store stuff dedicated to geometric stiffness


    CorotationalStrainJacobianBlock()
        : Inherit()
    {
    }

    void init_small() { _geometricStiffnessData.init_small(); }
    void init_qr( bool geometricStiffness ) { _geometricStiffnessData.init_qr( geometricStiffness ); }
    void init_polar( bool geometricStiffness ) { _geometricStiffnessData.init_polar( geometricStiffness ); }
    void init_svd( bool geometricStiffness ) { _geometricStiffnessData.init_svd( geometricStiffness ); }


    void addapply( OutCoord& /*result*/, const InCoord& /*data*/ ) {}
    void addapply_small( OutCoord& result, const InCoord& data )
    {
        StrainMat strainmat = cauchyStrainTensor( data.getF() ); // strainmat = ( F + Ft ) * 0.5
        _R.identity();

        addapply_common( result, data, strainmat );
    }
    void addapply_qr( OutCoord& result, const InCoord& data )
    {
        StrainMat strainmat;

        if( _geometricStiffnessData.invT() )
        {
            _geometricStiffnessData.degenerated() = helper::Decompose<Real>::QRDecomposition_stable( data.getF(), _R ) || determinant( data.getF() ) < helper::Decompose<Real>::zeroTolerance();
            Mat<3,3,Real> T = _R.multTranspose( data.getF() ); // T = rt * f
            _geometricStiffnessData.invT()->invert( T );
            strainmat = cauchyStrainTensor( T ); // s = ( T + Tt ) * 0.5
        }
        else
        {
            computeQR( data.getF(), _R, strainmat );
        }

        addapply_common( result, data, strainmat );
    }
    void addapply_polar( OutCoord& result, const InCoord& data )
    {
        StrainMat strainmat;

        helper::Decompose<Real>::polarDecomposition( data.getF(), _R, strainmat );

        if( _geometricStiffnessData.invG() )
        {
            helper::Decompose<Real>::polarDecompositionGradient_G( _R, strainmat, *_geometricStiffnessData.invG() );
            _geometricStiffnessData.degenerated() = determinant( data.getF() ) < helper::Decompose<Real>::zeroTolerance();
        }

        addapply_common( result, data, strainmat );
    }
    void addapply_svd( OutCoord& result, const InCoord& data )
    {
        StrainMat strainmat;


        //_geometricStiffnessData.degenerated() = computeSVD( data.getF(), _R, strainmat, U, S, V ) || determinant( data.getF() ) < helper::Decompose<Real>::zeroTolerance();

        if( _geometricStiffnessData.dROverdF() )
        {
            Affine U, V; Vec<material_dimensions,Real> S;
            _geometricStiffnessData.degenerated() = computeSVD( data.getF(), _R, strainmat, U, S, V )
                    || !helper::Decompose<Real>::polarDecomposition_stable_Gradient_dQOverdM( U, S, V, *_geometricStiffnessData.dROverdF() )
                    || determinant( data.getF() ) < helper::Decompose<Real>::zeroTolerance();
        }
        else helper::Decompose<Real>::polarDecomposition_stable( data.getF(), _R, strainmat );

        addapply_common( result, data, strainmat );
    }

    void addapply_common( OutCoord& result, const InCoord& data, StrainMat& strainmat )
    {
        // order 0
        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=(Real)1.;
        result.getStrain() += StrainMatToVoigt( strainmat );

        if( order > 0 )
        {
            // order 1
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                StrainMat T = _R.multTranspose( data.getGradientF( k ) ); // T = Rt * g
                result.getStrainGradient(k) += StrainMatToVoigt( cauchyStrainTensor( T ) ); // (T+Tt)*0.5
            }
        }
    }


    void addmult( OutDeriv& result,const InDeriv& data )
    {
        // order 0
        result.getStrain() += StrainMatToVoigt( _R.multTranspose( data.getF() ) );

        if( order > 0 )
        {
            // order 1
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                result.getStrainGradient(k) += StrainMatToVoigt( _R.multTranspose( data.getGradientF(k) ) );
            }
        }
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        // order 0
        result.getF() += _R*StressVoigtToMat( data.getStrain() );

        if( order > 0 )
        {
            // order 1
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                result.getGradientF(k) += _R*StressVoigtToMat( data.getStrainGradient(k) );
            }
        }
    }

    MatBlock getJ()
    {
        MatBlock B = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,Out::deriv_total_size,In::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eB(&B[0][0]);

        // order 0
        typedef Eigen::Matrix<Real,strain_size,frame_size,Eigen::RowMajor> JBlock;
        JBlock J = this->assembleJ( _R );
        eB.block(0,0,strain_size,frame_size) = J;

        if( order > 0 )
        {
            // order 1
            unsigned int offsetE=strain_size;
            for(unsigned int k=0; k<spatial_dimensions; k++)
            {
                eB.block(offsetE,(k+1)*frame_size,strain_size,frame_size) = J;
                offsetE+=strain_size;
            }
        }

        return B;
    }


    // requires dJ/dp. Not Yet implemented..
    KBlock getK(const OutDeriv& /*childForce*/)
    {
        return KBlock();
    }

    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */) {}
    void addDForce_qr( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )
    {
        // VERY UNSTABLE
        if( _geometricStiffnessData.degenerated() ) return;

        Affine dR;
        helper::Decompose<Real>::QRDecompositionGradient_dQ( _R, *_geometricStiffnessData.invT(), dx.getF(), dR );

        // order 0
        df.getF() += dR * StressVoigtToMat( childForce.getStrain() ) * kfactor;

        if( order > 0 )
        {
            // order 1
            /*for(unsigned int k=0;k<spatial_dimensions;k++)
            {
                df.getGradientF(k) += dR * StressVoigtToMat( childForce.getStrainGradient(k) ) * kfactor;
                helper::Decompose<Real>::QRDecompositionGradient_dQ( _R, *_dJ_Mat1, dx.getF(), dR );
                df.getF() += dR * StressVoigtToMat( childForce.getStrainGradient(k) ) * kfactor;
            }*/
        }
    }
    void addDForce_polar( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )
    {
        if( _geometricStiffnessData.degenerated() ) return;

        Affine dR;
        helper::Decompose<Real>::polarDecompositionGradient_dQ( *_geometricStiffnessData.invG(), _R, dx.getF(), dR );

        // order 0
        df.getF() += dR * StressVoigtToMat( childForce.getStrain() ) * kfactor;

        if( order > 0 )
        {
            // order 1
            /*for(unsigned int k=0;k<spatial_dimensions;k++)
            {
                df.getGradientF(k) += dR * StressVoigtToMat( childForce.getStrainGradient(k) ) * kfactor;
                helper::Decompose<Real>::polarDecompositionGradient_dQ( *_dJ_Mat1, _R, dx.getF(), dR );
                df.getF() += dR * StressVoigtToMat( childForce.getStrainGradient(k) ) * kfactor;
            }*/
        }
    }
    void addDForce_svd( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )
    {
        if( _geometricStiffnessData.degenerated() ) return;  // inverted or too flat -> no geometric stiffness for robustness

        Affine dR;
        //if( !helper::Decompose<Real>::polarDecomposition_stable_Gradient_dQ( *_dJ_Mat1, *_dJ_Vec, *_dJ_Mat2, dx.getF(), dR ) ) return;

        const Mat<frame_size,frame_size,Real> &dROverdF = *_geometricStiffnessData.dROverdF();
        for( int k=0 ; k<spatial_dimensions ; ++k ) // line of df
            for( int l=0 ; l<material_dimensions ; ++l ) // col of df
                for( int j=0 ; j<material_dimensions ; ++j ) // col of dR
                    for( int i=0 ; i<spatial_dimensions ; ++i ) // line of dR
                        dR[i][j] += dROverdF[i*material_dimensions+j][k*material_dimensions+l] * dx.getF()[k][l];


        // order 0
        df.getF() += dR * StressVoigtToMat( childForce.getStrain() ) * kfactor;

        if( order > 0 )
        {
            // order 1
            /*for( unsigned int g=0 ; g<spatial_dimensions ; g++ )
            {
                for( int k=0 ; k<spatial_dimensions ; ++k ) // line of df
                for( int l=0 ; l<material_dimensions ; ++l ) // col of df
                for( int j=0 ; j<material_dimensions ; ++j ) // col of dR
                for( int i=0 ; i<spatial_dimensions ; ++i ) // line of dR
                    dR[i][j] += dROverdF[i*material_dimensions+j][k*material_dimensions+l] * dx.getGradientF(g)[k][l];
                df.getGradientF(g) += dR * StressVoigtToMat( childForce.getStrainGradient(g) ) * kfactor;
            }*/
        }
    }
};



////////////////////////////////////////////////////////////////////////////////////
//////  F321 -> E321
////////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class CorotationalStrainJacobianBlock< F321(InReal) , E321(OutReal) > :
    public  BaseJacobianBlock< F321(InReal) , E321(OutReal) >
{
public:
    typedef F321(InReal) In;
    typedef E321(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    typedef typename In::Frame Frame;  ///< Matrix representing a deformation gradient
    typedef typename Out::StrainMat StrainMat;  ///< Matrix representing a strain
    enum { material_dimensions = In::material_dimensions };
    enum { spatial_dimensions = In::spatial_dimensions };
    enum { strain_size = Out::strain_size };
    enum { frame_size = spatial_dimensions*material_dimensions };

    typedef Mat<spatial_dimensions,material_dimensions,Real> Affine;  ///< Matrix representing a linear spatial transformation

    /**
    Mapping:   \f$ E = [grad(R^T u)+grad(R^T u)^T ]/2 = [R^T F + F^T R ]/2 - I = D - I \f$
    where:  _R/D are the rotational/skew symmetric parts of F=RD
    Jacobian:    \f$  dE = [R^T dF + dF^T R ]/2 \f$
    */

    static const bool constantJ=false;

    Affine _R;   ///< =  store rotational part of deformation gradient to compute J

    CorotationalStrainJacobianBlockGeometricStiffnessData<material_dimensions,frame_size,Real> _geometricStiffnessData; ///< store stuff dedicated to geometric stiffness

    CorotationalStrainJacobianBlock()
        : Inherit()
    {
    }

    void init_small() { _geometricStiffnessData.init_small(); }
    void init_qr( bool geometricStiffness ) { _geometricStiffnessData.init_qr( geometricStiffness ); }
    void init_polar( bool geometricStiffness ) { _geometricStiffnessData.init_svd( geometricStiffness ); }
    void init_svd( bool geometricStiffness ) { _geometricStiffnessData.init_svd( geometricStiffness ); }


    void addapply( OutCoord& /*result*/, const InCoord& /*data*/ ) {}

    void addapply_small( OutCoord& result, const InCoord& data )
    {
        // is a pure Cauchy tensor possible for a 2D element in a 3D world?
        // TODO

        addapply_qr( result, data );
    }

    void addapply_qr( OutCoord& result, const InCoord& data )
    {
        StrainMat strainmat;

        //computeQR( data.getF(), _R, strainmat );


        if( _geometricStiffnessData.invT() )
        {
            _geometricStiffnessData.degenerated() = helper::Decompose<Real>::QRDecomposition_stable( data.getF(), _R )
                    || determinant( data.getF() ) < helper::Decompose<Real>::zeroTolerance();
            Mat<2,2,Real> T = _R.multTranspose( data.getF() ); // T = rt * f
            _geometricStiffnessData.invT()->invert( T );
            strainmat = cauchyStrainTensor( T ); // s = ( T + Tt ) * 0.5
        }
        else
        {
            computeQR( data.getF(), _R, strainmat );
        }


        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=(Real)1.;
        result.getStrain() += StrainMatToVoigt( strainmat );
    }

    void addapply_polar( OutCoord& result, const InCoord& data )
    {
        // polar & svd are identical since inversion is not defined for 2d elements in a 3d world
        addapply_svd( result, data );
    }



    void addapply_svd( OutCoord& result, const InCoord& data )
    {
        StrainMat strainmat;

        if( _geometricStiffnessData.dROverdF() )
        {
            Affine U; Mat<material_dimensions,material_dimensions,Real> V; Vec<2,Real> Fdiag;
            computeSVD( data.getF(), _R, strainmat, U, Fdiag, V );
            _geometricStiffnessData.degenerated() = !helper::Decompose<Real>::polarDecompositionGradient_dQOverdM( U, Fdiag, V, *_geometricStiffnessData.dROverdF() )
                    || determinant( data.getF() ) < helper::Decompose<Real>::zeroTolerance();
        }
        else
        {
            helper::Decompose<Real>::polarDecomposition( data.getF(), _R, strainmat );
        }

        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=(Real)1.;
        result.getStrain() += StrainMatToVoigt( strainmat );
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getStrain() += StrainMatToVoigt( _R.multTranspose( data.getF() ) );
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getF() += _R*StressVoigtToMat( data.getStrain() );
    }

    MatBlock getJ()
    {
        MatBlock B = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,Out::deriv_total_size,In::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eB(&B[0][0]);
        eB = this->assembleJ(_R);
        return B;
    }


    // requires dJ/dp. Not Yet implemented..
    KBlock getK(const OutDeriv& /*childForce*/)
    {
        return KBlock();
    }
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */) {}
    void addDForce_qr( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )
    {
        if( _geometricStiffnessData.degenerated() ) return;

        Affine dR;
        helper::Decompose<Real>::QRDecompositionGradient_dQ( _R, *_geometricStiffnessData.invT(), dx.getF(), dR );
        df.getF() += dR * StressVoigtToMat( childForce.getStrain() ) * kfactor;
    }
    void addDForce_polar( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )
    {
        addDForce_svd( df, dx, childForce, kfactor );
    }
    void addDForce_svd( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )
    {
        if( _geometricStiffnessData.degenerated() ) return;

        Affine dR;
        //if( !helper::Decompose<Real>::polarDecompositionGradient_dQ( U, Fdiag, V, dx.getF(), dR ) ) return;

        const Mat<frame_size,frame_size,Real> &dROverdF = *_geometricStiffnessData.dROverdF();
        for( int k=0 ; k<spatial_dimensions ; ++k ) // line of df
            for( int l=0 ; l<material_dimensions ; ++l ) // col of df
                for( int j=0 ; j<material_dimensions ; ++j ) // col of dR
                    for( int i=0 ; i<spatial_dimensions ; ++i ) // line of dR
                        dR[i][j] += dROverdF[i*material_dimensions+j][k*material_dimensions+l] * dx.getF()[k][l];

        df.getF() += dR * StressVoigtToMat( childForce.getStrain() ) * kfactor;
    }
};



////////////////////////////////////////////////////////////////////////////////////
//////  F311 -> E311
////////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class CorotationalStrainJacobianBlock< F311(InReal) , E311(OutReal) > :
    public  BaseJacobianBlock< F311(InReal) , E311(OutReal) >
{
public:
    typedef F311(InReal) In;
    typedef E311(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    typedef typename In::Frame Frame;  ///< Matrix representing a deformation gradient
    typedef typename Out::StrainMat StrainMat;  ///< Matrix representing a strain
    enum { material_dimensions = In::material_dimensions };
    enum { spatial_dimensions = In::spatial_dimensions };
    enum { strain_size = Out::strain_size };
    enum { frame_size = spatial_dimensions*material_dimensions };

    typedef Mat<spatial_dimensions,material_dimensions,Real> Affine;  ///< Matrix representing a linear spatial transformation

    /**
    Mapping:   \f$ E = [grad(R^T u)+grad(R^T u)^T ]/2 = [R^T F + F^T R ]/2 - I = D - I \f$
    where:  _R/D are the rotational/skew symmetric parts of F=RD
    Jacobian:    \f$  dE = [R^T dF + dF^T R ]/2 \f$
    */

    static const bool constantJ=false;

    Affine _R;   ///< =  store unit vector to compute J
    Real nrm;   ///< =  store norm of deformation gradient to compute dJ

    CorotationalStrainJacobianBlockGeometricStiffnessData<material_dimensions,frame_size,Real> _geometricStiffnessData; ///< store stuff dedicated to geometric stiffness

    CorotationalStrainJacobianBlock()
        : Inherit()
    {
    }

    void init_small() { _geometricStiffnessData.init_small(); }
    void init_qr( bool geometricStiffness ) { _geometricStiffnessData.init_qr( geometricStiffness ); }
    void init_polar( bool geometricStiffness ) { _geometricStiffnessData.init_svd( geometricStiffness ); }
    void init_svd( bool geometricStiffness ) { _geometricStiffnessData.init_svd( geometricStiffness ); }


    void addapply( OutCoord& /*result*/, const InCoord& /*data*/ ) {}

    void addapply_small( OutCoord& result, const InCoord& data )
    {
        // is a pure Cauchy tensor possible for a 1D element in a 3D world?
        // TODO

        addapply_qr( result, data );
    }

    void addapply_qr( OutCoord& result, const InCoord& data )
    {
        StrainMat strainmat;

        computeQR( data.getF(), _R, strainmat );
        this->nrm=strainmat(0,0);

        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=(Real)1.;
        result.getStrain() += StrainMatToVoigt( strainmat );
    }

    void addapply_polar( OutCoord& result, const InCoord& data )
    {
        addapply_qr( result, data );
    }

    void addapply_svd( OutCoord& result, const InCoord& data )
    {
        addapply_qr( result, data );
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getStrain() += StrainMatToVoigt( _R.multTranspose( data.getF() ) );
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getF() += _R*StressVoigtToMat( data.getStrain() );
    }

    // J = u^T where u is the unit vector stored in R
    MatBlock getJ()
    {
        MatBlock B = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,Out::deriv_total_size,In::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eB(&B[0][0]);
        eB = this->assembleJ(_R);
        return B;
    }


    // Geometric stiffness : dJ/dF=(I-uu^T)/nrm where nrm is the norm of F
    KBlock getK(const OutDeriv& childForce)
    {
        KBlock K = _R*_R.transposed()*(-1.);
        for(unsigned int j=0; j<spatial_dimensions; j++) K(j,j)+=(Real)1.;
        K*=childForce.getStrain()[0]/this->nrm;
        return K;
    }
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */) {}
    void addDForce_qr( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )
    {
        df.getF()+=(dx.getF() - _R*_R.transposed()*dx.getF()) *kfactor*childForce.getStrain()[0]/this->nrm;
    }

    void addDForce_polar( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )
    {
        addDForce_qr( df, dx, childForce, kfactor );
    }
    void addDForce_svd( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )
    {
        addDForce_qr( df, dx, childForce, kfactor );
    }
};


} // namespace defaulttype
} // namespace sofa



#endif
