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
#ifndef FLEXIBLE_DiagonalStrainJacobianBlock_H
#define FLEXIBLE_DiagonalStrainJacobianBlock_H

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
////  macros
//////////////////////////////////////////////////////////////////////////////////
#define F221(type)  DefGradientTypes<2,2,0,type>
#define F321(type)  DefGradientTypes<3,2,0,type>
#define F331(type)  DefGradientTypes<3,3,0,type>
#define F332(type)  DefGradientTypes<3,3,1,type>
#define D221(type)  DiagonalizedStrainTypes<2,2,0,type>
#define D331(type)  DiagonalizedStrainTypes<3,3,0,type>
#define D332(type)  DiagonalizedStrainTypes<3,3,1,type>

//////////////////////////////////////////////////////////////////////////////////
////  helpers
//////////////////////////////////////////////////////////////////////////////////



/// 3D->3D
template<typename Real>
Vec<3,Real> computeSVD( const Mat<3,3,Real> &F, Mat<3,3,Real> &U, Mat<3,3,Real> &V )
{
    // using "invertible FEM" article notations

    Vec<3,Real> F_diagonal; // diagonalized strain

    Mat<3,3,Real> FtF = F.multTranspose( F ); // transformation from actual pos to rest pos

    helper::Decompose<Real>::eigenDecomposition_iterative( FtF, V, F_diagonal ); // eigen problem to obtain an orthogonal matrix V and diagonalized F


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


    switch( degeneratedF )
    {
    case 0: // no null value -> inverted but not degenerate
    {
        Vec<3,Real> F_diagonal_1;
        F_diagonal_1[0] = (Real)1.0/F_diagonal[0];
        F_diagonal_1[1] = (Real)1.0/F_diagonal[1];
        F_diagonal_1[2] = (Real)1.0/F_diagonal[2];
        U = F * V.multDiagonal( F_diagonal_1 );
        break;
    }
    case 1: // 1 null value -> collapsed to a plane -> keeps the 2 valid edges and construct the third
    {
        Vec<3,Real> F_diagonal_1;
        F_diagonal_1[Forder[0]] = (Real)1.0;
        F_diagonal_1[Forder[1]] = (Real)1.0/F_diagonal[Forder[1]];
        F_diagonal_1[Forder[2]] = (Real)1.0/F_diagonal[Forder[2]];
        U = F * V.multDiagonal( F_diagonal_1 );

        Vec<3,Real> c = cross( Vec<3,Real>(U[0][Forder[1]],U[1][Forder[1]],U[2][Forder[1]]), Vec<3,Real>(U[0][Forder[2]],U[1][Forder[2]],U[2][Forder[2]]) );
        U[0][Forder[0]] = c[0];
        U[1][Forder[0]] = c[1];
        U[2][Forder[0]] = c[2];
        break;
    }
    case 2: // 2 null values -> collapsed to an edge -> keeps the valid edge and build 2 orthogonal vectors
    {
        Vec<3,Real> F_diagonal_1;
        F_diagonal_1[Forder[0]] = (Real)1.0;
        F_diagonal_1[Forder[1]] = (Real)1.0;
        F_diagonal_1[Forder[2]] = (Real)1.0/F_diagonal[Forder[2]];
        U = F * V.multDiagonal( F_diagonal_1 );

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


    F_diagonal[0] -= 1;
    F_diagonal[1] -= 1;
    F_diagonal[2] -= 1;

    //std::cerr<<"SVD "<<F_diagonal<<std::endl;

    return F_diagonal;
}


/// 3D->2D
template<typename Real>
Vec<2,Real> computeSVD( const Mat<3,2,Real> &F, Mat<3,2,Real> &U, Mat<2,2,Real> &V )
{
    Vec<2,Real> F_diagonal; // diagonalized strain

    Mat<2,2,Real> FtF = F.multTranspose( F ); // transformation from actual pos to rest pos

    helper::Decompose<Real>::eigenDecomposition_iterative( FtF, V, F_diagonal );

    // if V is a reflexion -> made it a rotation by negating a column
    if( determinant(V) < 0 )
        for( int i=0 ; i<2; ++i )
            V[i][0] = -V[i][0];

    // compute the diagonalized strain and take the inverse
    Vec<2,Real> F_diagonal_1;
    for( int i = 0 ; i<2; ++i )
    {
        if( F_diagonal[i] < 1e-6 ) // numerical issues
        {
            F_diagonal[i] = 0;
            F_diagonal_1[i] = 1;
        }
        else
        {
            F_diagonal[i] = helper::rsqrt( F_diagonal[i] );
            F_diagonal_1[i] = (Real)1.0 / F_diagonal[i];
        }
    }

    // TODO check for degenerate cases (collapsed to a point, to an edge)
    // note that inversion is not defined for a 2d element in a 3d world

    U = F * V.multDiagonal( F_diagonal_1 );


    F_diagonal[0] -= 1;
    F_diagonal[1] -= 1;


    return F_diagonal;
}



//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////


/** Template class used to implement one jacobian block for DiagonalStrainMapping */
template<class TIn, class TOut>
class DiagonalStrainJacobianBlock : public BaseJacobianBlock<TIn,TOut> {};



//////////////////////////////////////////////////////////////////////////////////
////  F331 -> D331
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class DiagonalStrainJacobianBlock< F331(InReal) , D331(OutReal) > :
    public  BaseJacobianBlock< F331(InReal) , D331(OutReal) >
{
public:

    typedef F331(InReal) In;
    typedef D331(OutReal) Out;

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
    Mapping:   \f$ E = U.F.Vt\f$
    where:  U/V are the spatial and material rotation parts of F and E is diagonal
    Jacobian:    \f$  dE = U.dF.Vt \f$ Note that dE is not diagonal
    */

    static const bool constantJ = false;

    SpatialMaterialMat U;  ///< Spatial Rotation
    MaterialMaterialMat V; ///< Material Rotation


    void addapply( OutCoord& result, const InCoord& data )
    {
        Vec<3,Real> F_diag = computeSVD( data.getF(), U, V );

        result.getStrain()[0] += F_diag[0];
        result.getStrain()[1] += F_diag[1];
        result.getStrain()[2] += F_diag[2];
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getStrain() += StrainMatToVoigt( U.multTranspose( data.getF() * V ) );
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getF() += U * StressVoigtToMat( data.getStrain() ).multTransposed( V );
    }

    // TODO how?????
    MatBlock getJ()
    {
        MatBlock B = MatBlock();
        //typedef Eigen::Map<Eigen::Matrix<Real,Out::deriv_total_size,In::deriv_total_size,Eigen::RowMajor> > EigenMap;
        //EigenMap eB(&B[0][0]);
        //eB = assembleJ( U.multTransposed( V ) );
        return B;
    }


    // requires derivative of R. Not Yet implemented..
    KBlock getK(const OutDeriv& /*childForce*/)
    {
        return KBlock();
    }
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */)
    {
    }
};



//////////////////////////////////////////////////////////////////////////////////
////  F321 -> D221
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class DiagonalStrainJacobianBlock< F321(InReal) , D221(OutReal) > :
    public  BaseJacobianBlock< F321(InReal) , D221(OutReal) >
{
public:

    typedef F321(InReal) In;
    typedef D221(OutReal) Out;

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
    Mapping:   \f$ E = U.F.Vt\f$
    where:  U/V are the spatial and material rotation parts of F and E is diagonal
    Jacobian:    \f$  dE = U.dF.Vt \f$ Note that dE is not diagonal
    */

    static const bool constantJ = false;

    SpatialMaterialMat U;  ///< Spatial Rotation
    MaterialMaterialMat V; ///< Material Rotation


    void addapply( OutCoord& result, const InCoord& data )
    {
        Vec<2,Real> F_diag = computeSVD( data.getF(), U, V );

        result.getStrain()[0] += F_diag[0];
        result.getStrain()[1] += F_diag[1];
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getStrain() += StrainMatToVoigt( U.multTranspose( data.getF() * V ) );
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getF() += U * StressVoigtToMat( data.getStrain() ).multTransposed( V );
    }

    // TODO how?????
    MatBlock getJ()
    {
        MatBlock B = MatBlock();
        //typedef Eigen::Map<Eigen::Matrix<Real,Out::deriv_total_size,In::deriv_total_size,Eigen::RowMajor> > EigenMap;
        //EigenMap eB(&B[0][0]);
        //eB = assembleJ( U.multTransposed( V ) );
        return B;
    }


    // requires derivative of R. Not Yet implemented..
    KBlock getK(const OutDeriv& /*childForce*/)
    {
        return KBlock();
    }
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/, const OutDeriv& /*childForce*/, const double& /*kfactor */)
    {
    }
};


} // namespace defaulttype
} // namespace sofa



#endif // FLEXIBLE_DiagonalStrainJacobianBlock_H
