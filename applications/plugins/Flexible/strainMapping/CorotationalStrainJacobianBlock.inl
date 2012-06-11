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

#include <sofa/helper/decompose.h>

#include <sofa/helper/MatEigen.h>


// TODO
// - use code from E331 in E332
// - find a way to store data needed to compute dJ


namespace sofa
{

namespace defaulttype
{


//////////////////////////////////////////////////////////////////////////////////
////  macros
//////////////////////////////////////////////////////////////////////////////////
#define F221(type)  DefGradientTypes<2,2,0,type>
#define F321(type)  DefGradientTypes<3,2,0,type>
#define F331(type)  DefGradientTypes<3,3,0,type>
#define F332(type)  DefGradientTypes<3,3,1,type>
#define E221(type)  StrainTypes<2,2,0,type>
#define E331(type)  StrainTypes<3,3,0,type>
#define E332(type)  StrainTypes<3,3,1,type>

//////////////////////////////////////////////////////////////////////////////////
////  helpers
//////////////////////////////////////////////////////////////////////////////////


/// \return 0.5 * ( A + At )
template<int N, class Real>
static defaulttype::Mat<N,N,Real> cauchyStrainTensor( const defaulttype::Mat<N,N,Real>& A )
{
    defaulttype::Mat<N,N,Real> B;
    for( int i=0 ; i<N ; i++ )
    {
        B[i][i] = A[i][i];
        for( int j=i+1 ; j<N ; j++ )
            B[i][j] = B[j][i] = (Real)0.5 * ( A[i][j] + A[j][i] );
    }
    return B;
}


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
    Vec<3,Real> edgex( f[0][0], f[1][0], f[2][0] );
    Vec<3,Real> edgey( f[0][1], f[1][1], f[2][1] );

    edgex.normalize();
    Vec<3,Real> edgez = cross( edgex, edgey );
    edgez.normalize();
    edgey = cross( edgez, edgex );

    r[0][0] = edgex[0]; r[0][1] = edgey[0];
    r[1][0] = edgex[1]; r[1][1] = edgey[1];
    r[2][0] = edgex[2]; r[2][1] = edgey[2];

    Mat<2,2,Real> T = r.multTranspose( f ); // T = rt * f
    s = cauchyStrainTensor( T ); // s = ( T + Tt ) * 0.5
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



//////////////////////////////////////////////////////////////////////////////////
////  F331 -> E331
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class CorotationalStrainJacobianBlock< F331(InReal) , E331(OutReal) > :
    public  BaseJacobianBlock< F331(InReal) , E331(OutReal) >
{
public:

    typedef F331(InReal) In;
    typedef E331(OutReal) Out;

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

    typedef Mat<spatial_dimensions,spatial_dimensions,Real> Affine;  ///< Matrix representing a linear spatial transformation

    /**
    Mapping:   \f$ E = [grad(R^T u)+grad(R^T u)^T ]/2 = [R^T F + F^T R ]/2 - I = D - I \f$
    where:  R/D are the rotational/skew symmetric parts of F=RD
    Jacobian:    \f$  dE = [R^T dF + dF^T R ]/2 \f$
    */

    static const bool constantJ=false;

    Affine _R;   ///< =  store rotational part of deformation gradient to compute J

    // stuff only needed to compute the geometric stiffness and different for each decomposition method -> use pointers and create variables only when needed
    // TODO deal with it by using only one struct pointer?
    bool *_degenerated; // for qr & svd dJ
    Affine *_invTriS; // for qr dJ
    Affine *_G; // for polar dJ
    Affine *_U, *_V; Vec<3,Real> *_Fdiag; // for svd dJ

    CorotationalStrainJacobianBlock()
        : Inherit()
        , _degenerated( NULL )
        , _invTriS( NULL )
        , _G( NULL )
        , _U( NULL )
        , _V( NULL )
        , _Fdiag( NULL )
    {
    }


    void init_small()
    {
        if( _degenerated ) { delete _degenerated; _degenerated=NULL; }
        if( _invTriS ) { delete _invTriS; _invTriS=NULL; }
        if( _G ) { delete _G; _G=NULL; }
        if( _U ) { delete _U; _U=NULL; }
        if( _V ) { delete _V; _V=NULL; }
        if( _Fdiag ) { delete _Fdiag; _Fdiag=NULL; }
    }
    void init_qr( bool geometricStiffness )
    {
        if( _G ) { delete _G; _G=NULL; }
        if( _U ) { delete _U; _U=NULL; }
        if( _V ) { delete _V; _V=NULL; }
        if( _Fdiag ) { delete _Fdiag; _Fdiag=NULL; }

        if( geometricStiffness )
        {
            if( !_degenerated ) _degenerated = new bool();
            if( !_invTriS ) _invTriS = new Affine();
        }
        else
        {
            if( _degenerated ) { delete _degenerated; _degenerated=NULL; }
            if( _invTriS ) { delete _invTriS; _invTriS=NULL; }
        }
    }
    void init_polar( bool geometricStiffness )
    {
        if( _degenerated ) { delete _degenerated; _degenerated=NULL; }
        if( _invTriS ) { delete _invTriS; _invTriS=NULL; }
        if( _U ) { delete _U; _U=NULL; }
        if( _V ) { delete _V; _V=NULL; }
        if( _Fdiag ) { delete _Fdiag; _Fdiag=NULL; }

        if( geometricStiffness )
        {
            if( !_G )_G = new Affine();
        }
        else
        {
            if( _G ) { delete _G; _G=NULL; }
        }
    }
    void init_svd()
    {
        if( !_degenerated ) _degenerated = new bool();
        if( _invTriS ) { delete _invTriS; _invTriS=NULL; }
        if( _G ) { delete _G; _G=NULL; }
        if( !_U ) _U = new Affine();
        if( !_V ) _V = new Affine();
        if( !_Fdiag ) _Fdiag = new Vec<3,Real>();
    }



    void addapply( OutCoord& /*result*/, const InCoord& /*data*/ ) {}
    void addapply_small( OutCoord& result, const InCoord& data )
    {
        StrainMat strainmat = cauchyStrainTensor( data.getF() ); // strainmat = ( F + Ft ) * 0.5
        _R.identity();

        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=(Real)1.;
        result.getStrain() += StrainMatToVoigt( strainmat );
    }
    void addapply_qr( OutCoord& result, const InCoord& data )
    {
        StrainMat strainmat;

        *_degenerated = helper::Decompose<Real>::QRDecomposition_stable( data.getF(), _R );
        Mat<3,3,Real> T = _R.multTranspose( data.getF() ); // T = rt * f
        _invTriS->invert(T);
        strainmat = cauchyStrainTensor( T ); // s = ( T + Tt ) * 0.5

        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=(Real)1.;
        result.getStrain() += StrainMatToVoigt( strainmat );
    }
    void addapply_polar( OutCoord& result, const InCoord& data )
    {
        StrainMat strainmat;

        helper::Decompose<Real>::polarDecomposition( data.getF(), _R, strainmat );

        if( _G ) helper::Decompose<Real>::polarDecompositionGradient_G( _R, strainmat, *_G );

        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=(Real)1.;
        result.getStrain() += StrainMatToVoigt( strainmat );

    }
    void addapply_svd( OutCoord& result, const InCoord& data )
    {
        StrainMat strainmat;

        *_degenerated = computeSVD( data.getF(), _R, strainmat, *_U, *_Fdiag, *_V );

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
        eB = assembleJ(_R);
        return B;
    }


    // requires derivative of _R. Not Yet implemented..
    KBlock getK(const OutDeriv& /*childForce*/)
    {
        return KBlock();
    }

    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */) {}
    void addDForce_qr( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )
    {
        // VERY UNSTABLE
        if( *_degenerated ) return;

        Affine dR;
        helper::Decompose<Real>::QRDecompositionGradient_dQ( _R, *_invTriS, dx.getF(), dR );
        df.getF() += dR * StressVoigtToMat( childForce.getStrain() ) * kfactor;
    }
    void addDForce_polar( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )
    {
        //QUITE UNSTABLE
        Affine dR;
        helper::Decompose<Real>::polarDecompositionGradient_dQ( *_G, _R, dx.getF(), dR );
        df.getF() += dR * StressVoigtToMat( childForce.getStrain() ) * kfactor;
    }
    void addDForce_svd( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )
    {
        // QUITE STABLE
        if( *_degenerated ) return;  // inverted or too flat -> no geometric stiffness for robustness

        Affine dR;
        helper::Decompose<Real>::polarDecomposition_stable_Gradient_dQ( *_U, *_Fdiag, *_V, dx.getF(), dR ); // using SVD decomposition method
        df.getF() += dR * StressVoigtToMat( childForce.getStrain() ) * kfactor;
    }
};



////////////////////////////////////////////////////////////////////////////////////
//////  F321 -> E221
////////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class CorotationalStrainJacobianBlock< F321(InReal) , E221(OutReal) > :
    public  BaseJacobianBlock< F321(InReal) , E221(OutReal) >
{
public:
    typedef F321(InReal) In;
    typedef E221(OutReal) Out;

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

    Affine *_U; Mat<material_dimensions,material_dimensions,Real> *_V; Vec<2,Real> *_Fdiag; // for svd dJ


    CorotationalStrainJacobianBlock()
        : Inherit()
        , _U( NULL )
        , _V( NULL )
        , _Fdiag( NULL )
    {
    }

    void init_small()
    {
        if( _U ) { delete _U; _U=NULL; }
        if( _V ) { delete _V; _V=NULL; }
        if( _Fdiag ) { delete _Fdiag; _Fdiag=NULL; }
    }
    void init_qr( bool /*geometricStiffness*/ ) { init_small(); }
    void init_polar( bool /*geometricStiffness*/ ) { init_svd(); }
    void init_svd()
    {
        if( !_U ) _U = new Affine();
        if( !_V ) _V = new Mat<material_dimensions,material_dimensions,Real>();
        if( !_Fdiag ) _Fdiag = new Vec<2,Real>();
    }


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

        computeQR( data.getF(), _R, strainmat );

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

        computeSVD( data.getF(), _R, strainmat, *_U, *_Fdiag, *_V );

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
        eB = assembleJ(_R);
        return B;
    }


    // requires derivative of _R. Not Yet implemented..
    KBlock getK(const OutDeriv& /*childForce*/)
    {
        return KBlock();
    }
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */) {}
    void addDForce_qr( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */)
    {
    }
    void addDForce_polar( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )
    {
        addDForce_svd( df, dx, childForce, kfactor );
    }
    void addDForce_svd( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )
    {
        // QUITE STABLE
        Affine dR;
        helper::Decompose<Real>::polarDecompositionGradient_dQ( *_U, *_Fdiag, *_V, dx.getF(), dR );
        df.getF() += dR * StressVoigtToMat( childForce.getStrain() ) * kfactor;
    }
};



//////////////////////////////////////////////////////////////////////////////////
////  F332 -> E332
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class CorotationalStrainJacobianBlock< F332(InReal) , E332(OutReal) > :
    public  BaseJacobianBlock< F332(InReal) , E332(OutReal) >
{
public:

    typedef F332(InReal) In;
    typedef E332(OutReal) Out;

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

    static const bool constantJ=false;

    Affine _R;   ///< =  store rotational part of deformation gradient to compute J



    void init_small() {}
    void init_qr( bool /*geometricStiffness*/ ) {}
    void init_polar( bool /*geometricStiffness*/ ) {}
    void init_svd() {}


    void addapply( OutCoord& /*result*/, const InCoord& /*data*/ ) {}

    void addapply_small( OutCoord& result, const InCoord& data )
    {
        // order 0
        StrainMat strainmat = cauchyStrainTensor( data.getF() ); // strainmat = ( F + Ft ) * 0.5
        _R.identity();

        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=(Real)1.;
        result.getStrain() += StrainMatToVoigt( strainmat );

        // order 1
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            StrainMat T = _R.multTranspose( data.getGradientF( k ) ); // T = Rt * g
            result.getStrainGradient(k) += StrainMatToVoigt( cauchyStrainTensor( T ) ); // (T+Tt)*0.5
        }
    }

    void addapply_qr( OutCoord& result, const InCoord& data )
    {
        // order 0
        StrainMat strainmat;
        computeQR( data.getF(), _R, strainmat );

        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=(Real)1.;
        result.getStrain() += StrainMatToVoigt( strainmat );

        // order 1
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            StrainMat T = _R.multTranspose( data.getGradientF( k ) ); // T = Rt * g
            result.getStrainGradient(k) += StrainMatToVoigt( cauchyStrainTensor( T ) ); // (T+Tt)*0.5
        }
    }

    void addapply_polar( OutCoord& result, const InCoord& data )
    {
        // order 0
        StrainMat strainmat;
        helper::Decompose<Real>::polarDecomposition(data.getF(), _R, strainmat);

        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=(Real)1.;
        result.getStrain() += StrainMatToVoigt( strainmat );

        // order 1
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            StrainMat T = _R.multTranspose( data.getGradientF( k ) ); // T = Rt * g
            result.getStrainGradient(k) += StrainMatToVoigt( cauchyStrainTensor( T ) ); // (T+Tt)*0.5
        }
    }

    void addapply_svd( OutCoord& result, const InCoord& data )
    {
        // order 0
        StrainMat strainmat; Affine U, V; Vec<material_dimensions,Real> Fdiag;
        computeSVD( data.getF(), _R, strainmat, U, Fdiag, V );

        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=(Real)1.;
        result.getStrain() += StrainMatToVoigt( strainmat );

        // order 1
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            StrainMat T = _R.multTranspose( data.getGradientF( k ) ); // T = Rt * g
            result.getStrainGradient(k) += StrainMatToVoigt( cauchyStrainTensor( T ) ); // (T+Tt)*0.5
        }
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        // order 0
        result.getStrain() += StrainMatToVoigt( _R.multTranspose( data.getF() ) );
        // order 1
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            result.getStrainGradient(k) += StrainMatToVoigt( _R.multTranspose( data.getGradientF(k) ) );
        }
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        // order 0
        result.getF() += _R*StressVoigtToMat( data.getStrain() );
        // order 1
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            result.getGradientF(k) += _R*StressVoigtToMat( data.getStrainGradient(k) );
        }
    }

    MatBlock getJ()
    {
        MatBlock B = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,Out::deriv_total_size,In::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eB(&B[0][0]);
        // order 0
        typedef Eigen::Matrix<Real,strain_size,frame_size,Eigen::RowMajor> JBlock;
        JBlock J = assembleJ(_R);
        eB.template block(0,0,strain_size,frame_size) = J;
        // order 1
        unsigned int offsetE=strain_size;
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            eB.template block(offsetE,(k+1)*frame_size,strain_size,frame_size) = J;
            offsetE+=strain_size;
        }
        return B;
    }

    // requires derivative of _R. Not Yet implemented..
    KBlock getK(const OutDeriv& /*childForce*/)
    {
        return KBlock();
    }
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */) {}
    void addDForce_qr( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */)
    {
    }
    void addDForce_polar( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */)
    {
    }
    void addDForce_svd( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */)
    {
    }
};


} // namespace defaulttype
} // namespace sofa



#endif
