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
#include "../helper.h"

#include <sofa/helper/decompose.h>

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


/// 3D->3D
template<typename Real>
void computeQR( const Mat<3,3,Real> &f, Mat<3,3,Real> &r, Mat<3,3,Real> &s )
{
    helper::QRDecomposition_stable( f, r );

    Mat<3,3,Real> T = r.multTranspose( f ); // T = rt * f
    s = helper::symetrize( T ); // s = ( T + Tt ) * 0.5
}

/// 3D->3D
template<typename Real>
void computeSVD( const Mat<3,3,Real> &F, Mat<3,3,Real> &r, Mat<3,3,Real> &s )
{
    if( determinant(F) < 0 ) // inverted element -> SVD decomposition + handle degenerate cases
    {

        // using "invertible FEM" article notations

        Mat<3,3,Real> U, V; // the two rotations
        Vec<3,Real> F_diagonal; // diagonalized strain

        Mat<3,3,Real> FtF = F.multTranspose( F ); // transformation from actual pos to rest pos

        eigenDecomposition_noniterative( FtF, V, F_diagonal ); // eigen problem to obtain an orthogonal matrix V and diagonalized F


        // if V is a reflexion -> made it a rotation by negating a column
        if( determinant(V) < 0 )
            for( int i=0 ; i<3; ++i )
                V[i][0] = -V[i][0];

        // compute the diagonalized strain
        for( int i = 0 ; i<3; ++i )
        {
            if( F_diagonal[i] < 1e-6 ) // numerical issues
                F_diagonal[i] = 0;
            else
                F_diagonal[i] = helper::rsqrt( F_diagonal[i] );
        }

        // sort F_diagonal from small to large
        Vec<3,unsigned> Forder;
        if( F_diagonal[0]<F_diagonal[1] )
        {
            if( F_diagonal[0]<F_diagonal[2] )
            {
                Forder[0] = 0;
                if( F_diagonal[1]<F_diagonal[2] )
                {
                    Forder[1] = 1;
                    Forder[2] = 2;
                }
                else
                {
                    Forder[1] = 2;
                    Forder[2] = 1;
                }
            }
            else
            {
                Forder[0] = 2;
                Forder[1] = 0;
                Forder[2] = 1;
            }
        }
        else
        {
            if( F_diagonal[1]<F_diagonal[2] )
            {
                Forder[0] = 1;
                if( F_diagonal[0]<F_diagonal[2] )
                {
                    Forder[1] = 0;
                    Forder[2] = 2;
                }
                else
                {
                    Forder[1] = 2;
                    Forder[2] = 0;
                }
            }
            else
            {
                Forder[0] = 2;
                Forder[1] = 1;
                Forder[2] = 0;
            }
        }


        // the numbers of strain values too close to 0 indicates the kind of degenerescence
        int degeneratedF;
        for( degeneratedF=0 ; degeneratedF<3 && F_diagonal[ Forder[degeneratedF] ] < (Real)1e-6 ; ++degeneratedF ) ;


        // Warning: after the switch F_diagonal is no longer valid (it can be is its own inverse)
        switch( degeneratedF )
        {
        case 0: // no null value -> inverted but not degenerate
        {
            F_diagonal[0] = (Real)1.0/F_diagonal[0];
            F_diagonal[1] = (Real)1.0/F_diagonal[1];
            F_diagonal[2] = (Real)1.0/F_diagonal[2];
            U = F * V.multDiagonal( F_diagonal );
            break;
        }
        case 1: // 1 null value -> collapsed to a plane -> keeps the 2 valid edges and construct the third
        {
            F_diagonal[Forder[0]] = (Real)1.0;
            F_diagonal[Forder[1]] = (Real)1.0/F_diagonal[Forder[1]];
            F_diagonal[Forder[2]] = (Real)1.0/F_diagonal[Forder[2]];
            U = F * V.multDiagonal( F_diagonal );

            Vec<3,Real> c = cross( Vec<3,Real>(U[0][Forder[1]],U[1][Forder[1]],U[2][Forder[1]]), Vec<3,Real>(U[0][Forder[2]],U[1][Forder[2]],U[2][Forder[2]]) );
            U[0][Forder[0]] = c[0];
            U[1][Forder[0]] = c[1];
            U[2][Forder[0]] = c[2];
            break;
        }
        case 2: // 2 null values -> collapsed to an edge -> keeps the valid edge and build 2 orthogonal vectors
        {
            F_diagonal[Forder[0]] = (Real)1.0;
            F_diagonal[Forder[1]] = (Real)1.0;
            F_diagonal[Forder[2]] = (Real)1.0/F_diagonal[Forder[2]];
            U = F * V.multDiagonal( F_diagonal );

            // TODO: check if there is a more efficient way to do this

            Vec<3,Real> edge0, edge1, edge2( U[0][Forder[2]], U[1][Forder[2]], U[2][Forder[2]] );

            // check the main direction of edge2 to try to take a not too close arbritary vector
            Real abs0 = helper::rabs( edge2[0] );
            Real abs1 = helper::rabs( edge2[1] );
            Real abs2 = helper::rabs( edge2[2] );
            if( abs0 > abs1 )
            {
                if( abs0 > abs2 )
                {
                    edge0[0] = 0; edge0[1] = 1; edge0[2] = 0;
                }
                else
                {
                    edge0[0] = 1; edge0[1] = 0; edge0[2] = 0;
                }
            }
            else
            {
                if( abs1 > abs2 )
                {
                    edge0[0] = 0; edge0[1] = 0; edge0[2] = 1;
                }
                else
                {
                    edge0[0] = 1; edge0[1] = 0; edge0[2] = 0;
                }
            }

            edge1 = cross( edge2, edge0 );
            edge1.normalize();
            edge0 = cross( edge1, edge2 );

            U[0][Forder[0]] = edge0[0];
            U[1][Forder[0]] = edge0[1];
            U[2][Forder[0]] = edge0[2];

            U[0][Forder[1]] = edge1[0];
            U[1][Forder[1]] = edge1[1];
            U[2][Forder[1]] = edge1[2];

            break;
        }
        case 3: // 3 null values -> collapsed to a point -> build any orthogonal frame
            U.identity();
            break;
        }

        // un-inverting the element -> made U a rotation by negating a column
        if( determinant(U) < 0 ) // should always be true since we are handling the case det(F)<0, but it is not (due to numerical considerations?)
        {
            U[0][Forder[0]] *= -1;
            U[1][Forder[0]] *= -1;
            U[2][Forder[0]] *= -1;
        }

        // the world rotation of the element based on the two rotations computed by the SVD (world and material space)
        r = U.multTransposed( V ); // r = U * Vt
        s = r.multTranspose( F ); // s = rt * F
    }
    else // not inverted -> classical polar
    {
        polarDecomposition( F, r, s );
    }
}


/// 3D->2D
template<typename Real>
void computeSVD( const Mat<3,2,Real> &F, Mat<3,2,Real> &r, Mat<2,2,Real> &s )
{
    Mat<3,2,Real> U;
    Mat<2,2,Real> V;
    Vec<2,Real> F_diagonal; // diagonalized strain

    Mat<2,2,Real> FtF = F.multTranspose( F ); // transformation from actual pos to rest pos

    eigenDecomposition( FtF, V, F_diagonal ); // eigen problem to obtain an orthogonal matrix V and diagonalized F


    // if V is a reflexion -> made it a rotation by negating a column
    if( determinant(V) < 0 )
        for( int i=0 ; i<2; ++i )
            V[i][0] = -V[i][0];

    // compute the diagonalized strain and take the inverse
    for( int i = 0 ; i<2; ++i )
    {
        if( F_diagonal[i] < 1e-6 ) // numerical issues
            F_diagonal[i] = 1;
        else
            F_diagonal[i] = (Real)1.0 / helper::rsqrt( F_diagonal[i] );
    }

    // TODO check for degenerate cases (collapsed to a point, to an edge)
    // note that inversion is not defined for a 2d element in a 3d world

    U = F * V.multDiagonal( F_diagonal );

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

    Affine R;   ///< =  store rotational part of deformation gradient to compute J
    unsigned int decompositionMethod;

    void addapply( OutCoord& result, const InCoord& data )
    {
        StrainMat strainmat;

        switch( decompositionMethod )
        {
        case POLAR:
            helper::polarDecomposition( data.getF(), R, strainmat );
            break;
        case QR:
            computeQR( data.getF(), R, strainmat );
            break;
        case SMALL:
            strainmat = helper::symetrize( data.getF() ); // strainmat = ( F + Ft ) * 0.5
            R.identity();
            break;
        case SVD:
            computeSVD( data.getF(), R, strainmat );
            break;
        }

        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=(Real)1.;
        result.getStrain() += MatToVoigt( strainmat );
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getStrain() += MatToVoigt( R.multTranspose( data.getF() ) );
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getF() += R*VoigtToMat( data.getStrain() );
    }

    MatBlock getJ()
    {
        MatBlock B = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,Out::deriv_total_size,In::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eB(&B[0][0]);
        // order 0
        eB.template block(0,spatial_dimensions,strain_size,frame_size) = assembleJ(R);
        return B;
    }


    // requires derivative of R. Not Yet implemented..
    KBlock getK(const OutDeriv& /*childForce*/)
    {
        KBlock K = KBlock();
        return K;
    }
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */)
    {
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
    where:  R/D are the rotational/skew symmetric parts of F=RD
    Jacobian:    \f$  dE = [R^T dF + dF^T R ]/2 \f$
    */

    static const bool constantJ=false;

    Affine R;   ///< =  store rotational part of deformation gradient to compute J
    unsigned int decompositionMethod;

    void addapply( OutCoord& result, const InCoord& data )
    {

        // compute a 2*2 matrix based on the 3*2 matrix, in the plane of the  3-vectors.
        /*F221(InReal) data;
        Vec<3,Real> e0, e1, e2;
        for(unsigned i=0; i<3; i++){
            e0[i]=data32[i];
            e1[i]=data32[i+3];
        }
        Vec<2,Real> f0(e0.norm(),0);
        e0.normalize();
        e2=cross(e0,e1);
        e2.normalize();
        Vec<3,Real> ee1=cross(e2,e0);
        Vec<2,Real> f1(e1*e0,e1*ee1);
        for(unsigned i=0; i<2; i++){
            data[i][0]=f0[i];
            data[i][1]=f1[i];
        }*/



        StrainMat strainmat;

        switch( decompositionMethod )
        {
        case QR: // TODO
            //break;
        case SMALL: // TODO
            //break;

        case POLAR: // polar & svd are identical since inversion is not defined for 2d elements in a 3d world
        case SVD:
            computeSVD( data.getF(), R, strainmat );
            break;
        }
        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=(Real)1.;
        result.getStrain() += MatToVoigt( strainmat );
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getStrain() += MatToVoigt( R.multTranspose( data.getF() ) );
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getF() += R*VoigtToMat( data.getStrain() );
    }

    MatBlock getJ()
    {
        MatBlock B = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,Out::deriv_total_size,In::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eB(&B[0][0]);
        // order 0
        eB.template block(0,spatial_dimensions,strain_size,frame_size) = assembleJ(R);
        return B;
    }


    // requires derivative of R. Not Yet implemented..
    KBlock getK(const OutDeriv& /*childForce*/)
    {
        KBlock K = KBlock();
        return K;
    }
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */)
    {
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
        - R/D are the rotational/skew symmetric parts of F=RD
        - _k denotes derivative with respect to spatial dimension k
    Jacobian:
        - \f$  dE = [R^T dF + dF^T R ]/2 \f$
        - \f$  dE_k = [R^T dF_k + dF_k^T R ]/2 \f$
      */

    static const bool constantJ=false;

    Affine R;   ///< =  store rotational part of deformation gradient to compute J
    unsigned int decompositionMethod;

    void addapply( OutCoord& result, const InCoord& data )
    {
        // order 0
        StrainMat strainmat;
        switch( decompositionMethod )
        {
        case POLAR:
            helper::polarDecomposition(data.getF(), R, strainmat);
            break;
        case QR:
            computeQR( data.getF(), R, strainmat );
            break;
        case SMALL:
            strainmat = helper::symetrize( data.getF() ); // strainmat = ( F + Ft ) * 0.5
            R.identity();
            break;
        case SVD:
            computeSVD( data.getF(), R, strainmat );
            break;
        }

        for(unsigned int j=0; j<material_dimensions; j++) strainmat[j][j]-=(Real)1.;
        result.getStrain() += MatToVoigt( strainmat );

        // order 1
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            StrainMat T=R.transposed()*data.getGradientF(k);
            result.getStrainGradient(k) += MatToVoigt( T.plusTransposed( T ) * (Real)0.5 );
        }
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        // order 0
        result.getStrain() += MatToVoigt( R.multTranspose( data.getF() ) );
        // order 1
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            result.getStrainGradient(k) += MatToVoigt( R.multTranspose( data.getGradientF(k) ) );
        }
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        // order 0
        result.getF() += R*VoigtToMat( data.getStrain() );
        // order 1
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            result.getGradientF(k) += R*VoigtToMat( data.getStrainGradient(k) );
        }
    }

    MatBlock getJ()
    {
        MatBlock B = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,Out::deriv_total_size,In::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eB(&B[0][0]);
        // order 0
        typedef Eigen::Matrix<Real,strain_size,frame_size,Eigen::RowMajor> JBlock;
        JBlock J = assembleJ(R);
        eB.template block(0,spatial_dimensions,strain_size,frame_size) = J;
        // order 1
        unsigned int offsetE=strain_size;
        for(unsigned int k=0; k<spatial_dimensions; k++)
        {
            eB.template block(offsetE,spatial_dimensions+(k+1)*frame_size,strain_size,frame_size) = J;
            offsetE+=strain_size;
        }
        return B;
    }

    // requires derivative of R. Not Yet implemented..
    KBlock getK(const OutDeriv& /*childForce*/)
    {
        KBlock K;
        return K;
    }
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */)
    {
    }
};


} // namespace defaulttype
} // namespace sofa



#endif
