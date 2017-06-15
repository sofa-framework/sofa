/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_DEFAULTTYPE_LINEARBLENTYPES_INL
#define SOFA_DEFAULTTYPE_LINEARBLENTYPES_INL

#include "Blending.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/decompose.h>

#if defined(WIN32)
#define finite(x) (_finite(x))
#endif

namespace sofa
{
namespace defaulttype
{

template<class Real, int Dim>
inline Mat<Dim, Dim, Real> covNN(const Vec<Dim,Real>& v1, const Vec<Dim,Real>& v2)
{
    Mat<Dim, Dim, Real> res;
    for ( unsigned int i = 0; i < Dim; ++i)
        for ( unsigned int j = 0; j < Dim; ++j)
        {
            res[i][j] = v1[i] * v2[j];
        }
    return res;
}

template<class Real, int Dim1, int Dim2>
inline Mat<Dim1, Dim2, Real> covMN(const Vec<Dim1,Real>& v1, const Vec<Dim2,Real>& v2)
{
    Mat<Dim1, Dim2, Real> res;
    for ( unsigned int i = 0; i < Dim1; ++i)
        for ( unsigned int j = 0; j < Dim2; ++j)
        {
            res[i][j] = v1[i] * v2[j];
        }
    return res;
}

template<class _Real>
inline Mat<3, 3, _Real> crossProductMatrix(const Vec<3,_Real>& v)
{
    Mat<3, 3, _Real> res;
    res[0][0]=0;
    res[0][1]=-v[2];
    res[0][2]=v[1];
    res[1][0]=v[2];
    res[1][1]=0;
    res[1][2]=-v[0];
    res[2][0]=-v[1];
    res[2][1]=v[0];
    res[2][2]=0;
    return res;
}



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct LinearBlending<
        StdAffineTypes<3,typename _Material::Real>,
        Out, _Material, nbRef, 0
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdAffineTypes<3,InReal> In;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    /** Linear blend skinning.
    \f$ p = \sum_i w_i M_i \bar M_i p_0 \f$ where \f$ \bar M_i\f$ is the inverse of \f$ M_i \f$ in the reference configuration, and \f$ p_0 \f$ is the position of p in the reference configuration.
      The variation of p when a change \f$ dM_i \f$ is applied is thus \f$ w_i dM_i \bar M_i p_0 \f$, which we can compute as:\f$ dM_i * ( w_i \bar M_i p_0 ) \f$ in homogeneous coordinates.
      */
    struct JacobianBlock
    {
        OutCoord Pa;    ///< = \f$ w_i \bar M_i p_0 \f$  : affine part
        Real Pt;      ///< = \f$ w_i  \f$  : translation part
    };

    Vec<nbRef,unsigned int> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned int i=0;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            Jb[i].Pa= In::inverse(InitialTransform[index[i]]).pointToParent(InitialPos) *w[i];
            Jb[i].Pt= w[i];
        }
        if ( i<nbRef ) Jb[i].Pt=(Real)0; // used for loop terminations
    }

    OutCoord apply( const VecInCoord& d )  // Called in Apply
    {
        OutCoord result;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            result += d[index[i]].getCenter() * Jb[i].Pt + d[index[i]].getAffine() * Jb[i].Pa;
        }
        return result;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutCoord result;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            result += d[index[i]].getVCenter() * Jb[i].Pt + d[index[i]].getVAffine() * Jb[i].Pa;
        }
        return result;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */

        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {

            res[index[i]].getVCenter() += d * Jb[i].Pt;
            for (unsigned int j = 0; j < 3; ++j) res[index[i]].getVAffine()[j] += Jb[i].Pa * d[j];
        }
    }

    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {
            InDeriv parentJacobianVec;
            parentJacobianVec.getVCenter() += childJacobianVec * Jb[i].Pt;
            for (unsigned int j = 0; j < 3; ++j) parentJacobianVec.getVAffine()[j] += Jb[i].Pa * childJacobianVec[j];
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }

    inline friend std::ostream& operator<< ( std::ostream& o, const LinearBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, LinearBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }

};






//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine -> Affine =  -> DefGradient1 with dw=0
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct LinearBlending<
        StdAffineTypes<3,typename _Material::Real>,
        Out, _Material, nbRef, 3
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdAffineTypes<3,InReal> In;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    struct JacobianBlock
    {
        /** Linear blend skinning: A = \sum_i w_i M_i \bar M_i A_0  where \bar M_i is the inverse of M_i in the reference configuration, and A_0 is the position of A in the reference configuration.
          The variation of A when a change dM_i is applied is thus w_i dM_i \bar M_i A_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        OutCoord Pa;    ///< = dA = dMa_i (w_i \bar M_i A_0)  : affine part
        Real Pt;      ///< = dA = dMt_i (w_i)  : translation part
    };

    Vec<nbRef,unsigned int> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned int i=0;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            InCoord inverseInitialTransform = In::inverse(InitialTransform[index[i]]);
            Jb[i].Pa.getCenter()= inverseInitialTransform.pointToParent(InitialPos.getCenter()) *w[i];
            Jb[i].Pa.getAffine()= inverseInitialTransform.getAffine()*InitialPos.getAffine() *w[i];
            Jb[i].Pt= w[i];
        }
        if ( i<nbRef ) Jb[i].Pt=(Real)0; // used for loop terminations
    }

    OutCoord apply( const VecInCoord& d )  // Called in Apply
    {
        OutCoord result;
        result.getAffine().fill(0);
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            result.getCenter() += d[index[i]].getCenter() * Jb[i].Pt + d[index[i]].getAffine() * Jb[i].Pa.getCenter();
            result.getAffine() += d[index[i]].getAffine() * Jb[i].Pa.getAffine();
        }
        return result;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv result;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            result.getVCenter() += d[index[i]].getVCenter() * Jb[i].Pt + d[index[i]].getVAffine() * Jb[i].Pa.getCenter();
            result.getVAffine() += d[index[i]].getVAffine() * Jb[i].Pa.getAffine();
        }
        return result;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */

        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {

            res[index[i]].getVCenter() += d.getVCenter() * Jb[i].Pt;
            for (unsigned int j = 0; j < 3; ++j)
            {
                res[index[i]].getVAffine()[j] += Jb[i].Pa.getCenter() * d.getVCenter()[j];
                res[index[i]].getVAffine()[j] += Jb[i].Pa.getAffine() * d.getVAffine()[j];
            }
            //res[index[i]].getVAffine() += d.getVAffine() * Jb[i].Pa.getAffine().transposed();
        }
    }

    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {
            InDeriv parentJacobianVec;
            parentJacobianVec.getVCenter() += childJacobianVec.getVCenter() * Jb[i].Pt;
            for (unsigned int j = 0; j < 3; ++j)
            {
                parentJacobianVec.getVAffine()[j] += Jb[i].Pa.getCenter() * childJacobianVec.getVCenter()[j];
                parentJacobianVec.getVAffine()[j] += Jb[i].Pa.getAffine() * childJacobianVec.getVAffine()[j];
            }
            //parentJacobianVec.getVAffine() += childJacobianVec.getVAffine() * Jb[i].Pa.getAffine().transposed();
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }

    inline friend std::ostream& operator<< ( std::ostream& o, const LinearBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, LinearBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};




//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine -> Rigid =  -> Affine with polarDecomposition
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct LinearBlending<
        StdAffineTypes<3,typename _Material::Real>,
        Out, _Material, nbRef, 4
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdAffineTypes<3,InReal> In;
    typedef typename StdAffineTypes<3,OutReal>::Coord Affine;
    typedef Mat<3,3,OutReal> Mat33;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;
    typedef helper::Quater<OutReal> Quat;

    struct JacobianBlock
    {
        /** Linear blend skinning: A = \sum_i w_i M_i \bar M_i A_0  where \bar M_i is the inverse of M_i in the reference configuration, and A_0 is the position of A in the reference configuration.
          The variation of A when a change dM_i is applied is thus w_i dM_i \bar M_i A_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        Affine Pa;    ///< = dA = dMa_i (w_i \bar M_i A_0)  : affine part
        Real Pt;      ///< = dA = dMt_i (w_i)  : translation part
    };

    Vec<nbRef,unsigned int> index;
    Vec<nbRef,JacobianBlock> Jb;
    Mat33 A,R,S,Ainv;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned int i=0;
        Mat33 R;
        InitialPos.getOrientation().toMatrix(R);
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            InCoord inverseInitialTransform = In::inverse(InitialTransform[index[i]]);
            Jb[i].Pa.getCenter()= inverseInitialTransform.pointToParent(InitialPos.getCenter()) *w[i];
            Jb[i].Pa.getAffine()= inverseInitialTransform.getAffine()*R *w[i];
            Jb[i].Pt= w[i];
        }
        if ( i<nbRef ) Jb[i].Pt=(Real)0; // used for loop terminations
    }

    OutCoord apply( const VecInCoord& d )  // Called in Apply
    {
        OutCoord result;
        A.fill(0);
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            result.getCenter() += d[index[i]].getCenter() * Jb[i].Pt + d[index[i]].getAffine() * Jb[i].Pa.getCenter();
            A += d[index[i]].getAffine() * Jb[i].Pa.getAffine();
        }
        helper::Decompose<Real>::polarDecomposition(A, R, S);
        Ainv.invert(A);
        result.getOrientation().fromMatrix(R);
        return result;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        Mat33 Adot;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res.getVCenter() += d[index[i]].getVCenter() * Jb[i].Pt + d[index[i]].getVAffine() * Jb[i].Pa.getCenter();
            Adot += d[index[i]].getVAffine() * Jb[i].Pa.getAffine();
        }
        //w_x =R2.R^-1 ~ A2.A^-1  ->  Adot~ (w_x-I)A
        Mat33 w=Adot*Ainv;
        w[0][0]+=(Real)1.;	w[1][1]+=(Real)1.; w[2][2]+=(Real)1.;
        res.getAngular()[0]=(w[2][1]-w[1][2])*0.5;
        res.getAngular()[1]=(w[0][2]-w[2][0])*0.5;
        res.getAngular()[2]=(w[1][0]-w[0][1])*0.5;

        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */
        //Adot~ (w_x-I)A
        Mat33 Adot=crossProductMatrix(d.getAngular());
        Adot[0][0]-=(Real)1.;	Adot[1][1]-=(Real)1.; Adot[2][2]-=(Real)1.;
        Adot=Adot*A;

        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {

            res[index[i]].getVCenter() += d.getVCenter() * Jb[i].Pt;
            for (unsigned int j = 0; j < 3; ++j)
            {
                res[index[i]].getVAffine()[j] += Jb[i].Pa.getCenter() * d.getVCenter()[j];
                res[index[i]].getVAffine()[j] += Jb[i].Pa.getAffine() * Adot[j];
            }
            //	res[index[i]].getVAffine() += d.getVAffine() * Jb[i].Pa.getAffine().transposed();
        }
    }

    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
        //Adot~ (w_x-I)A
        Mat33 Adot=crossProductMatrix(childJacobianVec.getAngular());
        Adot[0][0]-=(Real)1.;	Adot[1][1]-=(Real)1.; Adot[2][2]-=(Real)1.;
        Adot=Adot*A;

        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {
            InDeriv parentJacobianVec;
            parentJacobianVec.getVCenter() += childJacobianVec.getVCenter() * Jb[i].Pt;
            for (unsigned int j = 0; j < 3; ++j)
            {
                parentJacobianVec.getVAffine()[j] += Jb[i].Pa.getCenter() * childJacobianVec.getVCenter()[j];
                parentJacobianVec.getVAffine()[j] += Jb[i].Pa.getAffine() * Adot[j];
            }
            //parentJacobianVec.getVAffine() += childJacobianVec.getVAffine() * Jb[i].Pa.getAffine().transposed();
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }

    inline friend std::ostream& operator<< ( std::ostream& o, const LinearBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 4>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, LinearBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 4>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};




//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine->DeformationGradient first order
//////////////////////////////////////////////////////////////////////////////////
template<class Out, class _Material, int nbRef>
struct LinearBlending<
        StdAffineTypes<3,typename _Material::Real>,
        Out, _Material, nbRef, 1
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdAffineTypes<3,InReal> In;
    typedef Vec<Out::spatial_dimensions, InReal> SpatialCoord; // = Vec3
    typedef Mat<Out::material_dimensions,Out::material_dimensions, InReal> MaterialFrame;
    typedef Vec<Out::material_dimensions, MaterialFrame> MaterialFrameGradient;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        SpatialCoord Pa;  ///< = dp = dMa_i (w_i \bar M_i p_0)  : affine part
        Real Pt;      ///< = dp = dMt_i (w_i)  : translation part
        MaterialFrame Fa;  ///< = dF = dMa_i (w_i \bar M_i + \bar M_i p_0 dw_i)
        MaterialDeriv Ft; ///< = dF = dMt_i (dw_i)
    };

    Vec<nbRef,unsigned int> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& dw, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned int i=0;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            InCoord inverseInitialTransform = In::inverse(InitialTransform[index[i]]);
            const SpatialCoord& vectorInLocalCoordinates = (inverseInitialTransform.getAffine()*InitialPos.getCenter() + inverseInitialTransform.getCenter());
            Jb[i].Pa=vectorInLocalCoordinates*w[i];
            Jb[i].Pt=w[i];
            Jb[i].Fa=inverseInitialTransform.getAffine() * w[i] + covNN( vectorInLocalCoordinates, dw[i]);
            Jb[i].Ft=dw[i];  // cerr << "InitialTransform[index[i]]= "<< InitialTransform[index[i]] << " dw[i] = " << dw[i] << endl;
        }
        if ( i<nbRef ) Jb[i].Pt=(Real)0; // used for loop terminations
    }



    OutCoord apply( const VecInCoord& d )  // Called in Apply
    {
        OutCoord res;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res.getCenter() += d[index[i]].getCenter( ) * Jb[i].Pt + d[index[i]].getAffine( ) * Jb[i].Pa;
            res.getMaterialFrame() += covNN( d[index[i]].getCenter(), Jb[i].Ft) + d[index[i]].getAffine() * Jb[i].Fa;
        }
        return res;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res.getCenter() += d[index[i]].getVCenter( ) * Jb[i].Pt + d[index[i]].getVAffine( ) * Jb[i].Pa;
            res.getMaterialFrame() += covNN( d[index[i]].getVCenter(), Jb[i].Ft) + d[index[i]].getVAffine( ) * Jb[i].Fa;
        }
        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {

            res[index[i]].getVCenter() +=  d.getCenter() * Jb[i].Pt;
            res[index[i]].getVCenter() += d.getMaterialFrame() * Jb[i].Ft;

            for (unsigned int j = 0; j < 3; ++j)
            {
                res[index[i]].getVAffine()[j] += Jb[i].Pa * d.getCenter()[j];
                res[index[i]].getVAffine()[j] += Jb[i].Fa * (d.getMaterialFrame()[j]);
            }

            //std::cout<<"Jt["<<i<<"]="<<Jb[i].Ft<<std::endl;
            //std::cout<<"Ja["<<i<<"]="<<Jb[i].Fa<<std::endl;
            //std::cout<<"dt="<<d.getCenter()<<std::endl;
            //std::cout<<"dF="<<d.getMaterialFrame()<<std::endl;
            //std::cout<<"ft["<<i<<"]="<<res[index[i]].getVCenter()<<std::endl;
            //std::cout<<"fa["<<i<<"]="<<res[index[i]].getVAffine()<<std::endl;

        }
    }
    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {

            InDeriv parentJacobianVec;
            parentJacobianVec.getVCenter() +=  childJacobianVec.getCenter() * Jb[i].Pt;
            parentJacobianVec.getVCenter() += childJacobianVec.getMaterialFrame() * Jb[i].Ft;

            for (unsigned int j = 0; j < 3; ++j)
            {
                parentJacobianVec.getVAffine()[j] += Jb[i].Pa * childJacobianVec.getCenter()[j];
                parentJacobianVec.getVAffine()[j] += Jb[i].Fa * (childJacobianVec.getMaterialFrame()[j]);
            }
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }
    inline friend std::ostream& operator<< ( std::ostream& o, const LinearBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, LinearBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine->DeformationGradient second order
//////////////////////////////////////////////////////////////////////////////////


template<class Out, class  _Material, int nbRef>
struct LinearBlending<
        StdAffineTypes<3,typename _Material::Real>,
        Out,
        _Material, nbRef,2
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdAffineTypes<3,InReal> In;
    typedef Vec<Out::spatial_dimensions, InReal> SpatialCoord; // = Vec3
    typedef Mat<Out::material_dimensions,Out::material_dimensions, InReal> MaterialFrame;
    typedef Vec<Out::material_dimensions, MaterialFrame> MaterialFrameGradient;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;


    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        SpatialCoord Pa;  ///< = dp = dMa_i (w_i \bar M_i p_0)  : affine part
        Real Pt;      ///< = dp = dMt_i (w_i)  : translation part
        MaterialFrame Fa;  ///< = dF = dMa_i (w_i \bar M_i + \bar M_i p_0 dw_i)
        MaterialDeriv Ft; ///< = dF = dMt_i (dw_i)
        MaterialFrameGradient dFa;  ///< = d gradF_k = dMa_i ( grad(w_i)_k \bar M_i + \bar M_i p_0 grad(dw_i)_k + grad(\bar M_i p_0)_k dw_i)
        MaterialMat dFt;  ///< = d gradF_k = dMt_i (grad(dw_i)_k)
    };

    Vec<nbRef,unsigned int> index;
    Vec<nbRef,JacobianBlock> Jb;

    InDeriv parentJacobianVec;
    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& dw, const Vec<nbRef,MaterialMat>&  ddw)
    {
        index = Index;
        unsigned int i=0;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            InCoord inverseInitialTransform = In::inverse(InitialTransform[index[i]]);
            SpatialCoord vectorInLocalCoordinates = (inverseInitialTransform.getAffine()*InitialPos.getCenter() + inverseInitialTransform.getCenter());
            Jb[i].Pa=vectorInLocalCoordinates*w[i];
            Jb[i].Pt=w[i];
            Jb[i].Fa=inverseInitialTransform.getAffine() * w[i] + covNN( vectorInLocalCoordinates, dw[i]);
            Jb[i].Ft=dw[i];
            Jb[i].dFt=ddw[i].transposed();
            MaterialFrame inverseInitialTransformT=inverseInitialTransform.getAffine().transposed();
            for (unsigned int k = 0; k < 3; ++k)
                Jb[i].dFa[k] = inverseInitialTransform.getAffine() * dw[i][k] + covNN( vectorInLocalCoordinates, Jb[i].dFt[k]) + covNN(inverseInitialTransformT[k],dw[i]); // dFa
        }
        if ( i<nbRef ) Jb[i].Pt=(Real)0; // used for loop terminations
    }

    OutCoord apply( const VecInCoord& d )  // Called in Apply
    {
        OutCoord res;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res.getCenter() += d[index[i]].getCenter( ) * Jb[i].Pt + d[index[i]].getAffine( ) * Jb[i].Pa;
            InDeriv parentJacobianVec;
            res.getMaterialFrame() += covNN( d[index[i]].getCenter( ), Jb[i].Ft) + d[index[i]].getAffine( ) * Jb[i].Fa;
            for (unsigned int k = 0; k < 3; ++k)
                res.getMaterialFrameGradient()[k] += covNN( d[index[i]].getCenter( ), Jb[i].dFt[k]) + d[index[i]].getAffine( ) * Jb[i].dFa[k];
        }
        return res;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res.getCenter() += d[index[i]].getVCenter( ) * Jb[i].Pt + d[index[i]].getVAffine( ) * Jb[i].Pa;
            res.getMaterialFrame() += covNN( d[index[i]].getVCenter( ), Jb[i].Ft) + d[index[i]].getVAffine( ) * Jb[i].Fa;
            for (unsigned int k = 0; k < 3; ++k)
                res.getMaterialFrameGradient()[k] += covNN( d[index[i]].getVCenter( ), Jb[i].dFt[k]) + d[index[i]].getVAffine( ) * Jb[i].dFa[k];
        }
        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            res[index[i]].getVCenter() +=  d.getCenter() * Jb[i].Pt;
            res[index[i]].getVCenter() += d.getMaterialFrame() * Jb[i].Ft;
            for (unsigned int k = 0; k < 3; ++k) res[index[i]].getVCenter() += d.getMaterialFrameGradient()[k] * Jb[i].dFt[k];

            for (unsigned int m = 0; m < 3; ++m)
            {
                res[index[i]].getVAffine()[m] += Jb[i].Pa * d.getCenter()[m];
                res[index[i]].getVAffine()[m] += Jb[i].Fa * d.getMaterialFrame()[m];
                for (unsigned int k = 0; k < 3; ++k) res[index[i]].getVAffine()[m] += Jb[i].dFa[k] * d.getMaterialFrameGradient()[k][m];
            }
        }
    }

    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec  ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            InDeriv parentJacobianVec;
            parentJacobianVec.getVCenter() +=  childJacobianVec.getCenter() * Jb[i].Pt;
            parentJacobianVec.getVCenter() += childJacobianVec.getMaterialFrame() * Jb[i].Ft;
            for (unsigned int k = 0; k < 3; ++k) parentJacobianVec.getVCenter() += childJacobianVec.getMaterialFrameGradient()[k] * Jb[i].dFt[k];

            for (unsigned int m = 0; m < 3; ++m)
            {
                parentJacobianVec.getVAffine()[m] += Jb[i].Pa * childJacobianVec.getCenter()[m];
                parentJacobianVec.getVAffine()[m] += Jb[i].Fa * childJacobianVec.getMaterialFrame()[m];
                for (unsigned int k = 0; k < 3; ++k) parentJacobianVec.getVAffine()[m] += Jb[i].dFa[k] * childJacobianVec.getMaterialFrameGradient()[k][m];
            }
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }

    inline friend std::ostream& operator<< ( std::ostream& o, const LinearBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, LinearBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic
//////////////////////////////////////////////////////////////////////////////////
template<class Out, class _Material, int nbRef>
struct LinearBlending<
        StdQuadraticTypes<3,typename _Material::Real>,
        Out, _Material, nbRef,0
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdQuadraticTypes<3,InReal> In;
    typedef typename In::QuadraticCoord QuadraticCoord; // vec9
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        QuadraticCoord Pa;    ///< = dp = dMa_i (w_i \bar M_i p_0)  : affine part
        Real Pt;      ///< = dp = dMt_i (w_i)  : translation part
    };

    Vec<nbRef,unsigned int> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned int i=0;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            InCoord inverseInitialTransform = In::inverse(InitialTransform[index[i]]);
            QuadraticCoord vectorInLocalCoordinates = In::convertToQuadraticCoord( (inverseInitialTransform.getAffine()*InitialPos + inverseInitialTransform.getCenter()) );
            Jb[i].Pa=vectorInLocalCoordinates*w[i];
            Jb[i].Pt=w[i];
        }
        if ( i<nbRef ) Jb[i].Pt=(Real)0; // used for loop terminations
    }


    OutCoord apply( const VecInCoord& d ) // Called in Apply
    {
        OutCoord res;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res +=  d[index[i]].getCenter() * Jb[i].Pt +  d[index[i]].getQuadratic() * Jb[i].Pa;
        }
        return res;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res +=  d[index[i]].getVCenter() * Jb[i].Pt +  d[index[i]].getVQuadratic() * Jb[i].Pa;
        }
        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            res[index[i]].getVCenter() += d * Jb[i].Pt;
            for (unsigned int j = 0; j < 3; ++j) res[index[i]].getVQuadratic()[j] += Jb[i].Pa * d[j];
        }
    }
    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec  ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            InDeriv parentJacobianVec;

            parentJacobianVec.getVCenter() += childJacobianVec * Jb[i].Pt;
            for (unsigned int j = 0; j < 3; ++j) parentJacobianVec.getVQuadratic()[j] += Jb[i].Pa * childJacobianVec[j];
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }

    inline friend std::ostream& operator<< ( std::ostream& o, const LinearBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, LinearBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};




//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic -> Affine  =  -> DefGradient1 with dw=0
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct LinearBlending<
        StdQuadraticTypes<3,typename _Material::Real>,
        Out, _Material, nbRef,3
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdQuadraticTypes<3,InReal> In;
    typedef typename In::QuadraticCoord QuadraticCoord; // vec9
    typedef typename In::Affine Affine;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef Mat<In::spatial_dimensions*In::spatial_dimensions,3,Real> QuadraticMat; // mat 9x3
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    struct JacobianBlock
    {
        /** Linear blend skinning: A = \sum_i w_i M_i \bar M_i A_0  where \bar M_i is the inverse of M_i in the reference configuration, and A_0 is the position of A in the reference configuration.
          The variation of A when a change dM_i is applied is thus w_i dM_i \bar M_i A_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        QuadraticCoord PaT;    ///< = dA = dMa_i (w_i \bar M_i A_0)  : affine part
        QuadraticMat PaA;
        Real Pt;      ///< = dA = dMt_i (w_i)  : translation part
    };

    Vec<nbRef,unsigned int> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned int i=0;
        for ( ; i<nbRef && w[i]>0; i++ )
        {

            InCoord inverseInitialTransform = In::inverse(InitialTransform[index[i]]);

            // use only the inverse of the linear part (affine+translation), squared and crossterms part undefined
            Affine invaff = inverseInitialTransform.getAffine();
            QuadraticCoord vectorInLocalCoordinates = In::convertToQuadraticCoord( (invaff*InitialPos.getCenter() + inverseInitialTransform.getCenter()) );
            QuadraticMat dinverseInitialTransform;
            for (unsigned int ii=0; ii<3; ++ii) for (unsigned int j=0; j<3; ++j) dinverseInitialTransform[ii][j]=invaff[ii][j];
            for (unsigned int j=0; j<3; ++j) dinverseInitialTransform[3+j][j]+=2.*vectorInLocalCoordinates[j];
            dinverseInitialTransform[6][0]+=vectorInLocalCoordinates[1]; dinverseInitialTransform[6][1]+=vectorInLocalCoordinates[0];
            dinverseInitialTransform[7][1]+=vectorInLocalCoordinates[2]; dinverseInitialTransform[7][2]+=vectorInLocalCoordinates[1];
            dinverseInitialTransform[8][0]+=vectorInLocalCoordinates[2]; dinverseInitialTransform[8][2]+=vectorInLocalCoordinates[0];

            Jb[i].PaT=vectorInLocalCoordinates*w[i];
            Jb[i].Pt=w[i];
            Jb[i].PaA=dinverseInitialTransform * w[i];
        }
        if ( i<nbRef ) Jb[i].Pt=(Real)0; // used for loop terminations
    }

    OutCoord apply( const VecInCoord& d )  // Called in Apply
    {
        OutCoord result;
        result.getAffine().fill(0);
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            result.getCenter() += d[index[i]].getCenter() * Jb[i].Pt + d[index[i]].getQuadratic() * Jb[i].PaT;
            result.getAffine() += d[index[i]].getQuadratic() * Jb[i].PaA;
        }
        return result;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv result;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            result.getVCenter() += d[index[i]].getVCenter() * Jb[i].Pt + d[index[i]].getVQuadratic() * Jb[i].PaT;
            result.getVAffine() += d[index[i]].getVQuadratic() * Jb[i].PaA;
        }
        return result;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */

        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {

            res[index[i]].getVCenter() += d.getVCenter() * Jb[i].Pt;
            for (unsigned int j = 0; j < 3; ++j)
            {
                res[index[i]].getVQuadratic()[j] += Jb[i].PaT * d.getVCenter()[j];
                res[index[i]].getVQuadratic()[j] += Jb[i].PaA * d.getVAffine()[j];
            }
            //res[index[i]].getVQuadratic() += d.getVAffine() * Jb[i].PaA.transposed();
        }
    }

    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {
            InDeriv parentJacobianVec;
            parentJacobianVec.getVCenter() += childJacobianVec.getVCenter() * Jb[i].Pt;
            for (unsigned int j = 0; j < 3; ++j)
            {
                parentJacobianVec.getVQuadratic()[j] += Jb[i].PaT * childJacobianVec.getVCenter()[j];
                parentJacobianVec.getVQuadratic()[j] += Jb[i].PaA * childJacobianVec.getVAffine()[j];
            }
            //parentJacobianVec.getVQuadratic() += childJacobianVec.getVAffine() * Jb[i].PaA.transposed();
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }

    inline friend std::ostream& operator<< ( std::ostream& o, const LinearBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, LinearBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic->DeformationGradient first order
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct LinearBlending<
        StdQuadraticTypes<3,typename _Material::Real>,
        Out, _Material, nbRef,1
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdQuadraticTypes<3,InReal> In;
    typedef typename In::QuadraticCoord QuadraticCoord; // vec9
    typedef typename In::Affine Affine;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef Mat<9,3,Real> MaterialFrame2;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        QuadraticCoord Pa;  ///< = dp = dMa_i (w_i \bar M_i p_0)  : affine part
        Real Pt;      ///< = dp = dMt_i (w_i)  : translation part
        MaterialFrame2 Fa;  ///< = dF = dMa_i (w_i \bar M_i + \bar M_i p_0 dw_i)
        MaterialDeriv Ft; ///< = dF = dMt_i (dw_i)
    };

    Vec<nbRef,unsigned int> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& dw, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned int i=0;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            InCoord inverseInitialTransform = In::inverse(InitialTransform[index[i]]);

            // use only the inverse of the linear part (affine+translation), squared and crossterms part undefined
            Affine invaff = inverseInitialTransform.getAffine();
            QuadraticCoord vectorInLocalCoordinates = In::convertToQuadraticCoord( (invaff*InitialPos.getCenter() + inverseInitialTransform.getCenter()) );
            MaterialFrame2 dinverseInitialTransform;
            for (unsigned int ii=0; ii<3; ++ii) for (unsigned int j=0; j<3; ++j) dinverseInitialTransform[ii][j]=invaff[ii][j];
            for (unsigned int j=0; j<3; ++j) dinverseInitialTransform[3+j][j]+=2.*vectorInLocalCoordinates[j];
            dinverseInitialTransform[6][0]+=vectorInLocalCoordinates[1]; dinverseInitialTransform[6][1]+=vectorInLocalCoordinates[0];
            dinverseInitialTransform[7][1]+=vectorInLocalCoordinates[2]; dinverseInitialTransform[7][2]+=vectorInLocalCoordinates[1];
            dinverseInitialTransform[8][0]+=vectorInLocalCoordinates[2]; dinverseInitialTransform[8][2]+=vectorInLocalCoordinates[0];

            Jb[i].Pa=vectorInLocalCoordinates*w[i];
            Jb[i].Pt=w[i];
            Jb[i].Fa=covMN(vectorInLocalCoordinates, dw[i]);
            Jb[i].Fa+=dinverseInitialTransform * w[i];
            Jb[i].Ft=dw[i];

            //std::cout<<"vectorInLocalCoordinates["<<i<<"]="<<vectorInLocalCoordinates<<std::endl;
        }
        if ( i<nbRef ) Jb[i].Pt=(Real)0; // used for loop terminations
    }


    OutCoord apply( const VecInCoord& d ) // Called in Apply
    {
        OutCoord res;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res.getCenter() += d[index[i]].getCenter( ) * Jb[i].Pt + d[index[i]].getQuadratic ( ) * Jb[i].Pa;
            res.getMaterialFrame() += covNN( d[index[i]].getCenter( ), Jb[i].Ft) + d[index[i]].getQuadratic( ) * Jb[i].Fa;
        }
        return res;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res.getCenter() += d[index[i]].getVCenter( ) * Jb[i].Pt + d[index[i]].getVQuadratic ( ) * Jb[i].Pa;
            res.getMaterialFrame() += covNN( d[index[i]].getVCenter( ), Jb[i].Ft) + d[index[i]].getVQuadratic( ) * Jb[i].Fa;
        }
        return res;
    }


    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {

        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {

            res[index[i]].getVCenter() += d.getCenter() * Jb[i].Pt;
            res[index[i]].getVCenter() += d.getMaterialFrame() * Jb[i].Ft;

            for (unsigned int j = 0; j < 3; ++j)
            {
                res[index[i]].getVQuadratic()[j] += Jb[i].Pa * d.getCenter()[j];
                res[index[i]].getVQuadratic()[j] += Jb[i].Fa * (d.getMaterialFrame()[j]);
            }

            //std::cout<<"JPt["<<i<<"]="<<Jb[i].Pt<<std::endl;
            //std::cout<<"JFt["<<i<<"]="<<Jb[i].Ft<<std::endl;
            //std::cout<<"JPa["<<i<<"]="<<Jb[i].Pa<<std::endl;
            //std::cout<<"JFa["<<i<<"]="<<Jb[i].Fa<<std::endl;
            //std::cout<<"dt="<<d.getCenter()<<std::endl;
            //std::cout<<"dF="<<d.getMaterialFrame()<<std::endl;
            //std::cout<<"ft["<<i<<"]="<<res[index[i]].getVCenter()<<std::endl;
            //std::cout<<"fa["<<i<<"]="<<res[index[i]].getVQuadratic()<<std::endl;

        }
    }
    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT
    {

        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {

            InDeriv parentJacobianVec;
            parentJacobianVec.getVCenter() += childJacobianVec.getCenter() * Jb[i].Pt;
            parentJacobianVec.getVCenter() += childJacobianVec.getMaterialFrame() * Jb[i].Ft;

            for (unsigned int j = 0; j < 3; ++j)
            {
                parentJacobianVec.getVQuadratic()[j] += Jb[i].Pa * childJacobianVec.getCenter()[j];
                parentJacobianVec.getVQuadratic()[j] += Jb[i].Fa * (childJacobianVec.getMaterialFrame()[j]);
            }
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }
    inline friend std::ostream& operator<< ( std::ostream& o, const LinearBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, LinearBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic->DeformationGradient second order
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct LinearBlending<
        StdQuadraticTypes<3,typename _Material::Real>,
        Out, _Material, nbRef,2
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdQuadraticTypes<3,InReal> In;
    //typedef Vec<Out::spatial_dimensions, InReal> SpatialCoord; // = Vec3
    typedef Mat<Out::material_dimensions,Out::material_dimensions, InReal> MaterialFrame;
    typedef Vec<Out::material_dimensions, MaterialFrame> MaterialFrameGradient;
    typedef Mat<9,3,Real> MaterialFrame2;
    typedef Vec<3,MaterialFrame2> MaterialFrameGradient2;
    typedef typename In::QuadraticCoord QuadraticCoord; // vec9
    typedef typename In::Affine Affine;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        //                JacobianBlock(){}
        //                JacobianBlock(const SpatialCoord2& p2, const Real& w,  const MaterialFrame2& f2,  const MaterialDeriv& dw,  const MaterialFrameGradient2& df2,  const MaterialMat& ddw):Pa(p2),Pt(w),Fa(f2),Ft(dw),dFa(df2),dFt(ddw){}
        QuadraticCoord Pa;  ///< = dp = dMa_i (w_i \bar M_i p_0)  : affine part
        Real Pt;      ///< = dp = dMt_i (w_i)  : translation part
        MaterialFrame2 Fa;  ///< = dF = dMa_i (w_i \bar M_i + \bar M_i p_0 dw_i)
        MaterialDeriv Ft; ///< = dF = dMt_i (dw_i)
        MaterialFrameGradient2 dFa;  ///< = d gradF_k = dMa_i ( grad(w_i)_k \bar M_i + \bar M_i p_0 grad(dw_i)_k + grad(\bar M_i p_0)_k dw_i)
        MaterialMat dFt;  ///< = d gradF_k = dMt_i (grad(dw_i)_k)
    };

    Vec<nbRef,unsigned int> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& dw, const Vec<nbRef,MaterialMat>&  ddw)
    {
        index = Index;
        unsigned int i=0;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            InCoord inverseInitialTransform = In::inverse(InitialTransform[index[i]]);

            // use only the inverse of the linear part (affine+translation), squared and crossterms part undefined
            Affine invaff = inverseInitialTransform.getAffine();
            QuadraticCoord vectorInLocalCoordinates = In::convertToQuadraticCoord( (invaff*InitialPos.getCenter() + inverseInitialTransform.getCenter()) );
            MaterialFrame2 dinverseInitialTransform;
            for (unsigned int ii=0; ii<3; ++ii) for (unsigned int j=0; j<3; ++j) dinverseInitialTransform[ii][j]=invaff[ii][j];
            for (unsigned int j=0; j<3; ++j) dinverseInitialTransform[3+j][j]+=2.*vectorInLocalCoordinates[j];
            dinverseInitialTransform[6][0]+=vectorInLocalCoordinates[1]; dinverseInitialTransform[6][1]+=vectorInLocalCoordinates[0];
            dinverseInitialTransform[7][1]+=vectorInLocalCoordinates[2]; dinverseInitialTransform[7][2]+=vectorInLocalCoordinates[1];
            dinverseInitialTransform[8][0]+=vectorInLocalCoordinates[2]; dinverseInitialTransform[8][2]+=vectorInLocalCoordinates[0];

            Jb[i].Pa=vectorInLocalCoordinates*w[i];
            Jb[i].Pt=w[i];
            Jb[i].Fa=covMN(vectorInLocalCoordinates, dw[i]);
            Jb[i].Fa+=dinverseInitialTransform * w[i];
            Jb[i].Ft=dw[i];

            Jb[i].dFt=ddw[i].transposed();
            Mat<3,9,Real> dinverseInitialTransformT=dinverseInitialTransform.transposed();
            for (unsigned int k = 0; k < 3; ++k)
            {
                Jb[i].dFa[k] = covMN( vectorInLocalCoordinates, Jb[i].dFt[k]);
                MaterialFrame2 m=covMN(dinverseInitialTransformT[k],dw[i] ); // dFa
                Jb[i].dFa[k]+=m+ dinverseInitialTransform * dw[i][k];
            }
        }
        if ( i<nbRef ) Jb[i].Pt=(Real)0; // used for loop terminations
    }

    OutCoord apply( const VecInCoord& d ) // Called in Apply
    {
        OutCoord res;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res.getCenter() += d[index[i]].getCenter( ) * Jb[i].Pt + d[index[i]].getQuadratic ( ) * Jb[i].Pa;
            res.getMaterialFrame() += covNN( d[index[i]].getCenter( ), Jb[i].Ft) + d[index[i]].getQuadratic( ) * Jb[i].Fa;
            for (unsigned int k = 0; k < 3; ++k)
            {
                res.getMaterialFrameGradient()[k] += covNN( d[index[i]].getCenter(), Jb[i].dFt[k]) + d[index[i]].getQuadratic() * Jb[i].dFa[k];
            }
        }
        return res;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res.getCenter() += d[index[i]].getVCenter( ) * Jb[i].Pt + d[index[i]].getVQuadratic ( ) * Jb[i].Pa;
            res.getMaterialFrame() += covNN( d[index[i]].getVCenter( ), Jb[i].Ft) + d[index[i]].getVQuadratic( ) * Jb[i].Fa;
            for (unsigned int k = 0; k < 3; ++k)
            {
                res.getMaterialFrameGradient()[k] += covNN( d[index[i]].getVCenter(), Jb[i].dFt[k]) + d[index[i]].getVQuadratic() * Jb[i].dFa[k];
            }
        }
        return res;
    }


    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            res[index[i]].getVCenter() +=  d.getCenter() * Jb[i].Pt;
            res[index[i]].getVCenter() += d.getMaterialFrame() * Jb[i].Ft;
            for (unsigned int k = 0; k < 3; ++k) res[index[i]].getVCenter() += d.getMaterialFrameGradient()[k] * Jb[i].dFt[k];

            for (unsigned int m = 0; m < 3; ++m)
            {
                res[index[i]].getVQuadratic()[m] += Jb[i].Pa * d.getCenter()[m];
                res[index[i]].getVQuadratic()[m] += Jb[i].Fa * d.getMaterialFrame()[m];
                for (unsigned int k = 0; k < 3; ++k) res[index[i]].getVQuadratic()[m] += Jb[i].dFa[k] * d.getMaterialFrameGradient()[k][m];
            }
        }
    }
    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            InDeriv parentJacobianVec;
            parentJacobianVec.getVCenter() +=  childJacobianVec.getCenter() * Jb[i].Pt;
            parentJacobianVec.getVCenter() += childJacobianVec.getMaterialFrame() * Jb[i].Ft;
            for (unsigned int k = 0; k < 3; ++k) parentJacobianVec.getVCenter() += childJacobianVec.getMaterialFrameGradient()[k] * Jb[i].dFt[k];

            for (unsigned int m = 0; m < 3; ++m)
            {
                parentJacobianVec.getVQuadratic()[m] += Jb[i].Pa * childJacobianVec.getCenter()[m];
                parentJacobianVec.getVQuadratic()[m] += Jb[i].Fa * childJacobianVec.getMaterialFrame()[m];
                for (unsigned int k = 0; k < 3; ++k) parentJacobianVec.getVQuadratic()[m] += Jb[i].dFa[k] * childJacobianVec.getMaterialFrameGradient()[k][m];
            }
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }
    inline friend std::ostream& operator<< ( std::ostream& o, const LinearBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, LinearBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic -> Quadratic
////   Warning !! Just declared but not implemented
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct LinearBlending<
        StdQuadraticTypes<3,typename _Material::Real>,
        Out, _Material, nbRef,5
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdQuadraticTypes<3,InReal> In;
    typedef typename In::QuadraticCoord QuadraticCoord; // vec9
    typedef typename In::Affine Affine;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef Mat<In::spatial_dimensions*In::spatial_dimensions,3,Real> QuadraticMat; // mat 9x3
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    struct JacobianBlock
    {
        /** Linear blend skinning: A = \sum_i w_i M_i \bar M_i A_0  where \bar M_i is the inverse of M_i in the reference configuration, and A_0 is the position of A in the reference configuration.
          The variation of A when a change dM_i is applied is thus w_i dM_i \bar M_i A_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        QuadraticCoord PaT;    ///< = dA = dMa_i (w_i \bar M_i A_0)  : affine part
        QuadraticMat PaA;
        Real Pt;      ///< = dA = dMt_i (w_i)  : translation part
    };

    Vec<nbRef,unsigned int> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned int i=0;
        for ( ; i<nbRef && w[i]>0; i++ )
        {

            InCoord inverseInitialTransform = In::inverse(InitialTransform[index[i]]);

            // use only the inverse of the linear part (affine+translation), squared and crossterms part undefined
            Affine invaff = inverseInitialTransform.getAffine();
            QuadraticCoord vectorInLocalCoordinates = In::convertToQuadraticCoord( (invaff*InitialPos.getCenter() + inverseInitialTransform.getCenter()) );
            QuadraticMat dinverseInitialTransform;
            for (unsigned int ii=0; ii<3; ++ii) for (unsigned int j=0; j<3; ++j) dinverseInitialTransform[ii][j]=invaff[ii][j];
            for (unsigned int j=0; j<3; ++j) dinverseInitialTransform[3+j][j]+=2.*vectorInLocalCoordinates[j];
            dinverseInitialTransform[6][0]+=vectorInLocalCoordinates[1]; dinverseInitialTransform[6][1]+=vectorInLocalCoordinates[0];
            dinverseInitialTransform[7][1]+=vectorInLocalCoordinates[2]; dinverseInitialTransform[7][2]+=vectorInLocalCoordinates[1];
            dinverseInitialTransform[8][0]+=vectorInLocalCoordinates[2]; dinverseInitialTransform[8][2]+=vectorInLocalCoordinates[0];

            Jb[i].PaT=vectorInLocalCoordinates*w[i];
            Jb[i].Pt=w[i];
            Jb[i].PaA=dinverseInitialTransform * w[i];
        }
        if ( i<nbRef ) Jb[i].Pt=(Real)0; // used for loop terminations
    }

    OutCoord apply( const VecInCoord& d )  // Called in Apply
    {
        OutCoord result;
        result.getAffine().fill(0);
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            result.getCenter() += d[index[i]].getCenter() * Jb[i].Pt + d[index[i]].getQuadratic() * Jb[i].PaT;
            result.getAffine() += d[index[i]].getQuadratic() * Jb[i].PaA;
        }
        return result;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv result;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            result.getVCenter() += d[index[i]].getVCenter() * Jb[i].Pt + d[index[i]].getVQuadratic() * Jb[i].PaT;
            result.getVAffine() += d[index[i]].getVQuadratic() * Jb[i].PaA;
        }
        return result;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */

        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {

            res[index[i]].getVCenter() += d.getVCenter() * Jb[i].Pt;
            for (unsigned int j = 0; j < 3; ++j)
            {
                res[index[i]].getVQuadratic()[j] += Jb[i].PaT * d.getVCenter()[j];
                res[index[i]].getVQuadratic()[j] += Jb[i].PaA * d.getVAffine()[j];
            }
            //res[index[i]].getVQuadratic() += d.getVAffine() * Jb[i].PaA.transposed();
        }
    }

    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {
            InDeriv parentJacobianVec;
            parentJacobianVec.getVCenter() += childJacobianVec.getVCenter() * Jb[i].Pt;
            for (unsigned int j = 0; j < 3; ++j)
            {
                parentJacobianVec.getVQuadratic()[j] += Jb[i].PaT * childJacobianVec.getVCenter()[j];
                parentJacobianVec.getVQuadratic()[j] += Jb[i].PaA * childJacobianVec.getVAffine()[j];
            }
            //parentJacobianVec.getVQuadratic() += childJacobianVec.getVAffine() * Jb[i].PaA.transposed();
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }

    inline friend std::ostream& operator<< ( std::ostream& o, const LinearBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 5>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, LinearBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 5>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Rigid->Vec
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct LinearBlending<
        StdRigidTypes<3,typename _Material::Real>,
        Out, _Material, nbRef,0
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdRigidTypes<3,InReal> In;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        OutCoord Pa0; ///< weighted point in local frame:   dp = dMa_i (w_i \bar M_i p_0)  : affine part
        OutCoord Pa;  ///< rotated point :  dp = Omega_i x [ Ma_i (w_i \bar M_i p_0) ]  : affine part
        Real Pt;      ///< = dp = dMt_i (w_i)  : translation part
    };

    Vec<nbRef,unsigned int> index;
    Vec<nbRef,JacobianBlock> Jb;


    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned int i=0;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            Jb[i].Pa0= InitialTransform[index[i]].pointToChild(InitialPos) * w[i] ;
            Jb[i].Pa= (InitialPos - InitialTransform[index[i]].getCenter() ) * w[i] ;
            Jb[i].Pt= w[i];
        }
        if ( i<nbRef ) Jb[i].Pt=(Real)0; // used for loop terminations
    }


    OutCoord apply( const VecInCoord& in )  // Called in Apply
    {
        OutCoord result;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            Jb[i].Pa=in[index[i]].rotate(Jb[i].Pa0); // = update of J according to current transform
            result += in[index[i]].getCenter() * Jb[i].Pt + Jb[i].Pa;
        }
        return result;
    }

    OutDeriv mult( const VecInDeriv& in ) // Called in ApplyJ
    {
        OutDeriv result;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {
            result += getLinear( in[index[i]] ) * Jb[i].Pt + cross(getAngular(in[index[i]]), Jb[i].Pa);
        }
        return result;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec6 product, and apply the transpose of this matrix
          */
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {
            getLinear(res[index[i]])  += d * Jb[i].Pt;
            getAngular(res[index[i]]) += cross(Jb[i].Pa, d);
        }
    }
    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec6 product, and apply the transpose of this matrix
          */
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {
            InDeriv parentJacobianVec;
            getLinear(parentJacobianVec)  += childJacobianVec * Jb[i].Pt;
            getAngular(parentJacobianVec) += cross(Jb[i].Pa, childJacobianVec);
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }
    inline friend std::ostream& operator<< ( std::ostream& o, const LinearBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, LinearBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }

};





//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Rigid -> Affine =  -> DefGradient1 with dw=0
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct LinearBlending<
        StdRigidTypes<3,typename _Material::Real>,
        Out, _Material, nbRef, 3
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdRigidTypes<3,InReal> In;
    typedef Vec<Out::spatial_dimensions, InReal> SpatialCoord; // = Vec3
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef Mat<3,3,Real> Mat33;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    struct JacobianBlock
    {
        /** Linear blend skinning: A = \sum_i w_i M_i \bar M_i A_0  where \bar M_i is the inverse of M_i in the reference configuration, and A_0 is the position of A in the reference configuration.
          The variation of A when a change dM_i is applied is thus w_i dM_i \bar M_i A_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        OutCoord Pa0;    ///< = dA = dMa_i (w_i \bar M_i A_0)  : affine part
        OutCoord Pa;    ///< = dA =  Omega_i x [ Ma_i (w_i \bar M_i A_0) ]  : affine part
        Real Pt;      ///< = dA = dMt_i (w_i)  : translation part
    };

    Vec<nbRef,unsigned int> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned int i=0;
        Mat33 InitialPos33=InitialPos.getAffine();

        for ( ; i<nbRef && w[i]>0; i++ )
        {

            Mat33 InitialTransform33;
            InitialTransform[index[i]].getOrientation().toMatrix(InitialTransform33);
            InCoord inverseInitialTransform = In::inverse(InitialTransform[index[i]]);
            Mat33 inverseInitialTransform33;
            inverseInitialTransform.getOrientation().toMatrix(inverseInitialTransform33);
            Mat33 inverseInitialTransformT = inverseInitialTransform33.transposed();

            const SpatialCoord& vectorInLocalCoordinates = inverseInitialTransform.pointToParent(InitialPos.getCenter());
            Jb[i].Pa0.getCenter()=vectorInLocalCoordinates*w[i];
            Jb[i].Pa.getCenter()=(InitialPos.getCenter() - InitialTransform[index[i]].getCenter() )*w[i];
            Jb[i].Pt=w[i];

            Jb[i].Pa0.getAffine()= inverseInitialTransform33 * InitialPos33 * w[i];
            Jb[i].Pa.getAffine()= InitialTransform33 * Jb[i].Pa0.getAffine();
        }
        if ( i<nbRef ) Jb[i].Pt=(Real)0; // used for loop terminations
    }

    OutCoord apply( const VecInCoord& d )  // Called in Apply
    {
        OutCoord res;
        res.getAffine().fill(0);
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {


            Jb[i].Pa.getCenter() =d[index[i]].rotate(Jb[i].Pa0.getCenter()); // = update of J according to current transform
            res.getCenter() += d[index[i]].getCenter( ) * Jb[i].Pt + Jb[i].Pa.getCenter();

            Mat33 Transform33;
            d[index[i]].getOrientation().toMatrix(Transform33);
            Jb[i].Pa.getAffine() = Transform33 * Jb[i].Pa0.getAffine();
            res.getAffine() += Jb[i].Pa.getAffine();
        }
        return res;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            res.getVCenter() +=  getLinear( d[index[i]] ) * Jb[i].Pt +  cross(getAngular(d[index[i]]), Jb[i].Pa.getCenter());
            const Mat33& Wx=crossProductMatrix(getAngular(d[index[i]]));
            res.getVAffine() += Wx * Jb[i].Pa.getAffine();
        }
        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */

        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {
            getLinear(res[index[i]]) +=  d.getVCenter() * Jb[i].Pt;
            getAngular(res[index[i]]) += cross(Jb[i].Pa.getCenter(), d.getVCenter());
            getAngular(res[index[i]])[0] += dot(Jb[i].Pa.getAffine()[1],d.getVAffine()[2]) - dot(Jb[i].Pa.getAffine()[2],d.getVAffine()[1]);
            getAngular(res[index[i]])[1] += dot(Jb[i].Pa.getAffine()[2],d.getVAffine()[0]) - dot(Jb[i].Pa.getAffine()[0],d.getVAffine()[2]);
            getAngular(res[index[i]])[2] += dot(Jb[i].Pa.getAffine()[0],d.getVAffine()[1]) - dot(Jb[i].Pa.getAffine()[1],d.getVAffine()[0]);
        }
    }

    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {
            InDeriv parentJacobianVec;
            getLinear(parentJacobianVec) +=  childJacobianVec.getVCenter() * Jb[i].Pt;
            getAngular(parentJacobianVec) += cross(Jb[i].Pa.getCenter(), childJacobianVec.getVCenter());
            getAngular(parentJacobianVec)[0] += dot(Jb[i].Pa.getAffine()[1],childJacobianVec.getVAffine()[2]) - dot(Jb[i].Pa.getAffine()[2],childJacobianVec.getVAffine()[1]);
            getAngular(parentJacobianVec)[1] += dot(Jb[i].Pa.getAffine()[2],childJacobianVec.getVAffine()[0]) - dot(Jb[i].Pa.getAffine()[0],childJacobianVec.getVAffine()[2]);
            getAngular(parentJacobianVec)[2] += dot(Jb[i].Pa.getAffine()[0],childJacobianVec.getVAffine()[1]) - dot(Jb[i].Pa.getAffine()[1],childJacobianVec.getVAffine()[0]);
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }



    inline friend std::ostream& operator<< ( std::ostream& o, const LinearBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, LinearBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};




//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Rigid -> Rigid =  -> Affine with polarDecomposition
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct LinearBlending<
        StdRigidTypes<3,typename _Material::Real>,
        Out, _Material, nbRef, 4
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdRigidTypes<3,InReal> In;
    typedef Vec<Out::spatial_dimensions, InReal> SpatialCoord; // = Vec3
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef Mat<3,3,Real> Mat33;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;
    typedef typename StdAffineTypes<3,OutReal>::Coord Affine;

    struct JacobianBlock
    {
        /** Linear blend skinning: A = \sum_i w_i M_i \bar M_i A_0  where \bar M_i is the inverse of M_i in the reference configuration, and A_0 is the position of A in the reference configuration.
          The variation of A when a change dM_i is applied is thus w_i dM_i \bar M_i A_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        Affine Pa0;    ///< = dA = dMa_i (w_i \bar M_i A_0)  : affine part
        Affine Pa;    ///< = dA =  Omega_i x [ Ma_i (w_i \bar M_i A_0) ]  : affine part
        Real Pt;      ///< = dA = dMt_i (w_i)  : translation part
    };

    Vec<nbRef,unsigned int> index;
    Vec<nbRef,JacobianBlock> Jb;
    Mat33 A,R,S,Ainv;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned int i=0;
        Mat33 InitialPos33;
        InitialPos.getOrientation().toMatrix(InitialPos33);

        for ( ; i<nbRef && w[i]>0; i++ )
        {

            Mat33 InitialTransform33;
            InitialTransform[index[i]].getOrientation().toMatrix(InitialTransform33);
            InCoord inverseInitialTransform = In::inverse(InitialTransform[index[i]]);
            Mat33 inverseInitialTransform33;
            inverseInitialTransform.getOrientation().toMatrix(inverseInitialTransform33);
            Mat33 inverseInitialTransformT = inverseInitialTransform33.transposed();

            const SpatialCoord& vectorInLocalCoordinates = inverseInitialTransform.pointToParent(InitialPos.getCenter());
            Jb[i].Pa0.getCenter()=vectorInLocalCoordinates*w[i];
            Jb[i].Pa.getCenter()=(InitialPos.getCenter() - InitialTransform[index[i]].getCenter() )*w[i];
            Jb[i].Pt=w[i];

            Jb[i].Pa0.getAffine()= inverseInitialTransform33 * InitialPos33 * w[i];
            Jb[i].Pa.getAffine()= InitialTransform33 * Jb[i].Pa0.getAffine();
        }
        if ( i<nbRef ) Jb[i].Pt=(Real)0; // used for loop terminations
    }

    OutCoord apply( const VecInCoord& d )  // Called in Apply
    {
        OutCoord res;
        A.fill(0);
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {


            Jb[i].Pa.getCenter() =d[index[i]].rotate(Jb[i].Pa0.getCenter()); // = update of J according to current transform
            res.getCenter() += d[index[i]].getCenter( ) * Jb[i].Pt + Jb[i].Pa.getCenter();

            Mat33 Transform33;
            d[index[i]].getOrientation().toMatrix(Transform33);
            Jb[i].Pa.getAffine() = Transform33 * Jb[i].Pa0.getAffine();
            A += Jb[i].Pa.getAffine();
        }
        helper::Decompose<Real>::polarDecomposition(A, R, S);
        Ainv.invert(A);

        res.getOrientation().fromMatrix(R);

        if(!finite(res.getOrientation()[0])) // hack to correct bug in fromMatrix (when m.z().z() = m.y().y() + epsilon and m.x().x()=1)
        {
            Real s = (Real)sqrt ((R.x().x() - (R.y().y() + R.z().z())) + 1.0f);
            res.getOrientation()[0] = s * 0.5f; // x OK
            if (s != 0.0f)
                s = 0.5f / s;
            res.getOrientation()[1] = (Real)((R.x().y() + R.y().x()) * s); // y OK
            res.getOrientation()[2] = (Real)((R.z().x() + R.x().z()) * s); // z OK
            res.getOrientation()[3] = (Real)((R.z().y() - R.y().z()) * s); // w OK
        }
        return res;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        Mat33 Adot;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            res.getVCenter() +=  getLinear( d[index[i]] ) * Jb[i].Pt +  cross(getAngular(d[index[i]]), Jb[i].Pa.getCenter());
            const Mat33& Wx=crossProductMatrix(getAngular(d[index[i]]));
            Adot += Wx * Jb[i].Pa.getAffine();
        }

        //w_x =R2.R^-1 ~ A2.A^-1  ->  Adot~ (w_x-I)A
        Mat33 w=Adot*Ainv;
        w[0][0]+=(Real)1.;	w[1][1]+=(Real)1.; w[2][2]+=(Real)1.;
        res.getAngular()[0]=(w[2][1]-w[1][2])*0.5;
        res.getAngular()[1]=(w[0][2]-w[2][0])*0.5;
        res.getAngular()[2]=(w[1][0]-w[0][1])*0.5;


        /*				Mat33 w=Adot*Sinv;
        res.getAngular()[0]=(w[2][1]-w[1][2])*0.5;
        res.getAngular()[1]=(w[0][2]-w[2][0])*0.5;
        res.getAngular()[2]=(w[1][0]-w[0][1])*0.5;
        */

        /*				Mat33 Sinv2,R2,S2,A2=Adot+A,I;
             helper::Decompose<Real>::polarDecomposition(A2, R2, S2);
        Quat q1; q1.fromMatrix(R);
        Quat q2; q2.fromMatrix(R2);
        Quat q=q2*q1.inverse();
        Real phi; q.quatToAxis(res.getAngular(), phi);
        res.getAngular()*=phi;
        */

        /*
        I[0][0]=I[1][1]=I[2][2]=(Real)1.;
        Sinv2.invert(S2);
        Mat33 w=Adot*Sinv2 + R*(S*Sinv2 - I);
        res.getAngular()[0]=(w[2][1]-w[1][2])*0.5;
        res.getAngular()[1]=(w[0][2]-w[2][0])*0.5;
        res.getAngular()[2]=(w[1][0]-w[0][1])*0.5;
        */

        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */
        //Adot~ (w_x-I)A
        Mat33 Adot=crossProductMatrix(d.getAngular());
        Adot[0][0]-=(Real)1.;	Adot[1][1]-=(Real)1.; Adot[2][2]-=(Real)1.;
        Adot=Adot*A;

        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {
            getLinear(res[index[i]]) +=  d.getVCenter() * Jb[i].Pt;
            getAngular(res[index[i]]) += cross(Jb[i].Pa.getCenter(), d.getVCenter());
            getAngular(res[index[i]])[0] += dot(Jb[i].Pa.getAffine()[1],Adot[2]) - dot(Jb[i].Pa.getAffine()[2],Adot[1]);
            getAngular(res[index[i]])[1] += dot(Jb[i].Pa.getAffine()[2],Adot[0]) - dot(Jb[i].Pa.getAffine()[0],Adot[2]);
            getAngular(res[index[i]])[2] += dot(Jb[i].Pa.getAffine()[0],Adot[1]) - dot(Jb[i].Pa.getAffine()[1],Adot[0]);
        }
    }

    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
        //Adot~ (w_x-I)A
        Mat33 Adot=crossProductMatrix(childJacobianVec.getAngular());
        Adot[0][0]-=(Real)1.;	Adot[1][1]-=(Real)1.; Adot[2][2]-=(Real)1.;
        Adot=Adot*A;

        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {
            InDeriv parentJacobianVec;
            getLinear(parentJacobianVec) +=  childJacobianVec.getVCenter() * Jb[i].Pt;
            getAngular(parentJacobianVec) += cross(Jb[i].Pa.getCenter(), childJacobianVec.getVCenter());
            getAngular(parentJacobianVec)[0] += dot(Jb[i].Pa.getAffine()[1],Adot[2]) - dot(Jb[i].Pa.getAffine()[2],Adot[1]);
            getAngular(parentJacobianVec)[1] += dot(Jb[i].Pa.getAffine()[2],Adot[0]) - dot(Jb[i].Pa.getAffine()[0],Adot[2]);
            getAngular(parentJacobianVec)[2] += dot(Jb[i].Pa.getAffine()[0],Adot[1]) - dot(Jb[i].Pa.getAffine()[1],Adot[0]);
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }



    inline friend std::ostream& operator<< ( std::ostream& o, const LinearBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 4>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, LinearBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 4>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};







//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Rigid->DeformationGradient first order
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class  _Material, int nbRef>
struct LinearBlending<
        StdRigidTypes<3,typename _Material::Real>,
        Out, _Material, nbRef,1
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdRigidTypes<3,InReal> In;
    typedef Vec<Out::spatial_dimensions, InReal> SpatialCoord; // = Vec3
    typedef Mat<Out::material_dimensions,Out::material_dimensions, InReal> MaterialFrame;
    typedef Vec<Out::material_dimensions, MaterialFrame> MaterialFrameGradient;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef Mat<3,3,Real> Mat33;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        SpatialCoord Pa0; ///< = dp = dMa_i (w_i \bar M_i p_0)  : affine part
        SpatialCoord Pa;  ///< = dp = Omega_i x [ Ma_i (w_i \bar M_i p_0) ]  : affine part
        Real Pt;      ///< = dp = dMt_i (w_i)  : translation part

        MaterialFrame Fa0;  ///< = dF = dMa_i (w_i \bar M_i + \bar M_i p_0 dw_i)
        MaterialFrame Fa;   ///< = dF = Omega_i x [ Ma_i  (w_i \bar M_i + \bar M_i p_0 dw_i) ]
        MaterialDeriv Ft; ///< = dF = dMt_i (dw_i)

    };

    Vec<nbRef,unsigned int> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& dw, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned int i=0;
        for ( ; i<nbRef && w[i]>0; i++ )
        {

            Mat33 InitialTransform33;
            InitialTransform[index[i]].getOrientation().toMatrix(InitialTransform33);
            InCoord inverseInitialTransform = In::inverse(InitialTransform[index[i]]);
            Mat33 inverseInitialTransform33;
            inverseInitialTransform.getOrientation().toMatrix(inverseInitialTransform33);
            Mat33 inverseInitialTransformT = inverseInitialTransform33.transposed();

            const SpatialCoord& vectorInLocalCoordinates = inverseInitialTransform.pointToParent(InitialPos.getCenter());
            Jb[i].Pa0=vectorInLocalCoordinates*w[i];
            Jb[i].Pa=(InitialPos.getCenter() - InitialTransform[index[i]].getCenter() )*w[i];
            Jb[i].Pt=w[i];

            Jb[i].Fa0= inverseInitialTransform33 * w[i] + covNN( vectorInLocalCoordinates, dw[i]);
            Jb[i].Fa= InitialTransform33 * Jb[i].Fa0;
            Jb[i].Ft=dw[i];

        }
        if ( i<nbRef ) Jb[i].Pt=(Real)0; // used for loop terminations
    }



    OutCoord apply( const VecInCoord& d )  // Called in Apply
    {
        OutCoord res;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            Jb[i].Pa =d[index[i]].rotate(Jb[i].Pa0); // = update of J according to current transform
            res.getCenter() += d[index[i]].getCenter( ) * Jb[i].Pt + Jb[i].Pa;

            Mat33 Transform33;
            d[index[i]].getOrientation().toMatrix(Transform33);
            Jb[i].Fa=Transform33 * Jb[i].Fa0;
            res.getMaterialFrame() += covNN( d[index[i]].getCenter( ), Jb[i].Ft) + Jb[i].Fa;

        }
        return res;
    }


    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            res.getCenter() +=  getLinear( d[index[i]] ) * Jb[i].Pt +  cross(getAngular(d[index[i]]), Jb[i].Pa);
            const Mat33& Wx=crossProductMatrix(getAngular(d[index[i]]));
            res.getMaterialFrame() += covNN( getLinear( d[index[i]] ) , Jb[i].Ft) + Wx * Jb[i].Fa;
        }
        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            getLinear(res[index[i]]) +=  d.getCenter() * Jb[i].Pt;
            getLinear(res[index[i]]) += d.getMaterialFrame() * Jb[i].Ft;

            getAngular(res[index[i]]) += cross(Jb[i].Pa, d.getCenter());

            getAngular(res[index[i]])[0] += dot(Jb[i].Fa[1],d.getMaterialFrame()[2]) - dot(Jb[i].Fa[2],d.getMaterialFrame()[1]);
            getAngular(res[index[i]])[1] += dot(Jb[i].Fa[2],d.getMaterialFrame()[0]) - dot(Jb[i].Fa[0],d.getMaterialFrame()[2]);
            getAngular(res[index[i]])[2] += dot(Jb[i].Fa[0],d.getMaterialFrame()[1]) - dot(Jb[i].Fa[1],d.getMaterialFrame()[0]);
        }
    }
    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            InDeriv parentJacobianVec;
            getLinear(parentJacobianVec) +=  childJacobianVec.getCenter() * Jb[i].Pt;
            getLinear(parentJacobianVec) += childJacobianVec.getMaterialFrame() * Jb[i].Ft;

            getAngular(parentJacobianVec) += cross(Jb[i].Pa, childJacobianVec.getCenter());

            getAngular(parentJacobianVec)[0] += dot(Jb[i].Fa[1],childJacobianVec.getMaterialFrame()[2]) - dot(Jb[i].Fa[2],childJacobianVec.getMaterialFrame()[1]);
            getAngular(parentJacobianVec)[1] += dot(Jb[i].Fa[2],childJacobianVec.getMaterialFrame()[0]) - dot(Jb[i].Fa[0],childJacobianVec.getMaterialFrame()[2]);
            getAngular(parentJacobianVec)[2] += dot(Jb[i].Fa[0],childJacobianVec.getMaterialFrame()[1]) - dot(Jb[i].Fa[1],childJacobianVec.getMaterialFrame()[0]);
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }
    inline friend std::ostream& operator<< ( std::ostream& o, const LinearBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, LinearBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }

};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Rigid->DeformationGradient second order
//////////////////////////////////////////////////////////////////////////////////


template<class Out, class  _Material, int nbRef>
struct LinearBlending<
        StdRigidTypes<3,typename _Material::Real>,
        Out, _Material, nbRef,2
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdRigidTypes<3,InReal> In;
    typedef Vec<Out::spatial_dimensions, InReal> SpatialCoord; // = Vec3
    typedef Mat<Out::material_dimensions,Out::material_dimensions, InReal> MaterialFrame;
    typedef Vec<Out::material_dimensions, MaterialFrame> MaterialFrameGradient;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef Mat<3,3,Real> Mat33;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        SpatialCoord Pa0; ///< = dp = dMa_i (w_i \bar M_i p_0)  : affine part
        SpatialCoord Pa;  ///< = dp = Omega_i x [ Ma_i (w_i \bar M_i p_0) ]  : affine part
        Real Pt;      ///< = dp = dMt_i (w_i)  : translation part

        MaterialFrame Fa0;  ///< = dF = dMa_i (w_i \bar M_i + \bar M_i p_0 dw_i)
        MaterialFrame Fa;   ///< = dF = Omega_i x [ Ma_i  (w_i \bar M_i + \bar M_i p_0 dw_i) ]
        MaterialDeriv Ft; ///< = dF = dMt_i (dw_i)

        MaterialFrameGradient dFa0;  ///< = d gradF_k = dMa_i ( grad(w_i)_k \bar M_i + \bar M_i p_0 grad(dw_i)_k + grad(\bar M_i p_0)_k dw_i)
        MaterialFrameGradient dFa;  ///< = d gradF_k = Omega_i x [ Ma_i ( grad(w_i)_k \bar M_i + \bar M_i p_0 grad(dw_i)_k + grad(\bar M_i p_0)_k dw_i) ]
        MaterialMat dFt;  ///< = d gradF_k = dMt_i (grad(dw_i)_k)



    };

    Vec<nbRef,unsigned int> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& dw, const Vec<nbRef,MaterialMat>&  ddw)
    {
        index = Index;
        unsigned int i=0;
        for ( ; i<nbRef && w[i]>0; i++ )
        {

            Mat33 InitialTransform33;
            InitialTransform[index[i]].getOrientation().toMatrix(InitialTransform33);
            InCoord inverseInitialTransform = In::inverse(InitialTransform[index[i]]);
            Mat33 inverseInitialTransform33;
            inverseInitialTransform.getOrientation().toMatrix(inverseInitialTransform33);
            Mat33 inverseInitialTransformT = inverseInitialTransform33.transposed();

            const SpatialCoord& vectorInLocalCoordinates = inverseInitialTransform.pointToParent(InitialPos.getCenter());
            Jb[i].Pa0=vectorInLocalCoordinates*w[i];
            Jb[i].Pa=(InitialPos.getCenter() - InitialTransform[index[i]].getCenter() )*w[i];
            Jb[i].Pt=w[i];

            Jb[i].Fa0= inverseInitialTransform33 * w[i] + covNN( vectorInLocalCoordinates, dw[i]);
            Jb[i].Fa= InitialTransform33 * Jb[i].Fa0;
            Jb[i].Ft=dw[i];

            Jb[i].dFt=ddw[i].transposed();
            for (unsigned int  k = 0; k < 3; ++k)
            {
                Jb[i].dFa0[k] = inverseInitialTransform33 * dw[i][k] + covNN( vectorInLocalCoordinates, Jb[i].dFt[k]) + covNN(inverseInitialTransformT[k],dw[i]); // dFa
                Jb[i].dFa[k] = InitialTransform33 *  Jb[i].dFa0[k];
            }

        }
        if ( i<nbRef ) Jb[i].Pt=(Real)0; // used for loop terminations
    }



    OutCoord apply( const VecInCoord& d )  // Called in Apply
    {
        OutCoord res;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            Jb[i].Pa =d[index[i]].rotate(Jb[i].Pa0); // = update of J according to current transform
            res.getCenter() += d[index[i]].getCenter( ) * Jb[i].Pt + Jb[i].Pa;

            Mat33 Transform33;
            d[index[i]].getOrientation().toMatrix(Transform33);
            Jb[i].Fa=Transform33 * Jb[i].Fa0;
            res.getMaterialFrame() += covNN( d[index[i]].getCenter( ), Jb[i].Ft) + Jb[i].Fa;

            for (unsigned int k = 0; k < 3; ++k)
            {
                Jb[i].dFa[k] = Transform33 * Jb[i].dFa0[k];
                res.getMaterialFrameGradient()[k] += covNN( d[index[i]].getCenter( ), Jb[i].dFt[k]) + Jb[i].dFa[k];
            }
        }
        return res;
    }


    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            res.getCenter() +=  getLinear( d[index[i]] ) * Jb[i].Pt +  cross(getAngular(d[index[i]]), Jb[i].Pa);
            const Mat33& Wx=crossProductMatrix(getAngular(d[index[i]]));
            res.getMaterialFrame() += covNN( getLinear( d[index[i]] ) , Jb[i].Ft) + Wx * Jb[i].Fa;
            for (unsigned int k = 0; k < 3; ++k)
                res.getMaterialFrameGradient()[k] += covNN( getLinear( d[index[i]] ) , Jb[i].dFt[k]) +  Wx * Jb[i].dFa[k];
        }
        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            getLinear(res[index[i]]) +=  d.getCenter() * Jb[i].Pt;
            getLinear(res[index[i]]) += d.getMaterialFrame() * Jb[i].Ft;
            for (unsigned int k = 0; k < 3; ++k) getLinear(res[index[i]]) += d.getMaterialFrameGradient()[k] * Jb[i].dFt[k];

            getAngular(res[index[i]]) += cross(Jb[i].Pa, d.getCenter());

            getAngular(res[index[i]])[0] += dot(Jb[i].Fa[1],d.getMaterialFrame()[2]) - dot(Jb[i].Fa[2],d.getMaterialFrame()[1]);
            getAngular(res[index[i]])[1] += dot(Jb[i].Fa[2],d.getMaterialFrame()[0]) - dot(Jb[i].Fa[0],d.getMaterialFrame()[2]);
            getAngular(res[index[i]])[2] += dot(Jb[i].Fa[0],d.getMaterialFrame()[1]) - dot(Jb[i].Fa[1],d.getMaterialFrame()[0]);

            for (unsigned int k = 0; k < 3; ++k)
            {
                getAngular(res[index[i]])[0] += dot(Jb[i].dFa[k][1],d.getMaterialFrameGradient()[k][2]) - dot(Jb[i].dFa[k][2],d.getMaterialFrameGradient()[k][1]);
                getAngular(res[index[i]])[1] += dot(Jb[i].dFa[k][2],d.getMaterialFrameGradient()[k][0]) - dot(Jb[i].dFa[k][0],d.getMaterialFrameGradient()[k][2]);
                getAngular(res[index[i]])[2] += dot(Jb[i].dFa[k][0],d.getMaterialFrameGradient()[k][1]) - dot(Jb[i].dFa[k][1],d.getMaterialFrameGradient()[k][0]);
            }
        }
    }
    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            InDeriv parentJacobianVec;
            getLinear(parentJacobianVec) +=  childJacobianVec.getCenter() * Jb[i].Pt;
            getLinear(parentJacobianVec) += childJacobianVec.getMaterialFrame() * Jb[i].Ft;
            for (unsigned int k = 0; k < 3; ++k) getLinear(parentJacobianVec) += childJacobianVec.getMaterialFrameGradient()[k] * Jb[i].dFt[k];

            getAngular(parentJacobianVec) += cross(Jb[i].Pa, childJacobianVec.getCenter());

            getAngular(parentJacobianVec)[0] += dot(Jb[i].Fa[1],childJacobianVec.getMaterialFrame()[2]) - dot(Jb[i].Fa[2],childJacobianVec.getMaterialFrame()[1]);
            getAngular(parentJacobianVec)[1] += dot(Jb[i].Fa[2],childJacobianVec.getMaterialFrame()[0]) - dot(Jb[i].Fa[0],childJacobianVec.getMaterialFrame()[2]);
            getAngular(parentJacobianVec)[2] += dot(Jb[i].Fa[0],childJacobianVec.getMaterialFrame()[1]) - dot(Jb[i].Fa[1],childJacobianVec.getMaterialFrame()[0]);

            for (unsigned int k = 0; k < 3; ++k)
            {
                getAngular(parentJacobianVec)[0] += dot(Jb[i].dFa[k][1],childJacobianVec.getMaterialFrameGradient()[k][2]) - dot(Jb[i].dFa[k][2],childJacobianVec.getMaterialFrameGradient()[k][1]);
                getAngular(parentJacobianVec)[1] += dot(Jb[i].dFa[k][2],childJacobianVec.getMaterialFrameGradient()[k][0]) - dot(Jb[i].dFa[k][0],childJacobianVec.getMaterialFrameGradient()[k][2]);
                getAngular(parentJacobianVec)[2] += dot(Jb[i].dFa[k][0],childJacobianVec.getMaterialFrameGradient()[k][1]) - dot(Jb[i].dFa[k][1],childJacobianVec.getMaterialFrameGradient()[k][0]);
            }
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }
    inline friend std::ostream& operator<< ( std::ostream& o, const LinearBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, LinearBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }

};

}


} // namespace sofa

#endif
