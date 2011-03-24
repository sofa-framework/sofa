/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_DEFAULTTYPE_DUALQUATBLENTYPES_INL
#define SOFA_DEFAULTTYPE_DUALQUATBLENTYPES_INL

#include "MappingTypes.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/gl/template.h>
#include "DualQuatTypes.h"

namespace sofa
{
namespace defaulttype
{

template<class Real,int Dim>
Vec<Dim*Dim,Real>& MattoVec(Mat<Dim,Dim,Real>& m) { return *reinterpret_cast<Vec<Dim*Dim,Real>*>( &m[0]); }
template<class Real,int Dim>
const Vec<Dim*Dim,Real>& MattoVec(const Mat<Dim,Dim,Real>& m) { return *reinterpret_cast<const Vec<Dim*Dim,Real>*>( &m[0]); }

template<class Real,int Dim>
Mat<Dim,Dim,Real>& VectoMat(Vec<Dim*Dim,Real>& v) { return *reinterpret_cast<Mat<Dim,Dim,Real>*>( &v[0]); }
template<class Real,int Dim>
const Mat<Dim,Dim,Real>& VectoMat(const Vec<Dim*Dim,Real>& v) { return *reinterpret_cast<const Mat<Dim,Dim,Real>*>( &v[0]); }

//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine
//////////////////////////////////////////////////////////////////////////////////

/// TO DO!!!!!  (currently = linear blending)

template<class Out, class _Material, int nbRef>
struct DualQuatBlendTypes<
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

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
        The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
        */
        OutCoord Pa;    ///< = dp = dMa_i (w_i \bar M_i p_0)  : affine part
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
            //                    inverseInitialTransform[index[i]] = In::inverse(InitialTransform[index[i]]);
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

    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlendTypes<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlendTypes<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }

};




//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine -> Affine =  -> DefGradient1 with dw=0
//////////////////////////////////////////////////////////////////////////////////

/// TO DO!!!!!  (currently = linear blending)

template<class Out, class _Material, int nbRef>
struct DualQuatBlendTypes<
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

    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlendTypes<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlendTypes<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine -> Rigid =  -> Affine with polarDecomposition
//////////////////////////////////////////////////////////////////////////////////

/// TO DO!!!!!  (currently = linear blending)

template<class Out, class _Material, int nbRef>
struct DualQuatBlendTypes<
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
    Mat33 A,R,S,Sinv;

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
        polar_decomp(A, R, S);
        Sinv.invert(S);
        result.getOrientation().fromMatrix(R);
        return result;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv result;
        Mat33 Adot;
        for ( unsigned int i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            result.getVCenter() += d[index[i]].getVCenter() * Jb[i].Pt + d[index[i]].getVAffine() * Jb[i].Pa.getCenter();
            Adot += d[index[i]].getVAffine() * Jb[i].Pa.getAffine();
        }
        //Adot ~ w x S
        Quat q; q.fromMatrix(Adot*Sinv);
        Real phi; q.quatToAxis(result.getAngular(), phi);
        result.getAngular()*=phi;

        //A2=A+Adot;
        //polar_decomp(A2, R2, S2);
        //q2.fromMatrix(R2);
        //q2*=q.inverse();
        //Real phi;
        //q2.quatToAxis(result.getAngular(), phi);
        //result.getAngular()*=phi;
        return result;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
        */
        //Adot ~ w x S
        Mat33 Adot=crossProductMatrix(d.getAngular())*S;

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
        //Adot ~ w x S
        Mat33 Adot=crossProductMatrix(childJacobianVec.getAngular())*S;

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

    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlendTypes<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 4>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlendTypes<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 4>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine->DeformationGradient first order
//////////////////////////////////////////////////////////////////////////////////

/// TO DO!!!!!  (currently = linear blending)

template<class Out, class _Material, int nbRef>
struct DualQuatBlendTypes<
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
    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlendTypes<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlendTypes<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine->DeformationGradient second order
//////////////////////////////////////////////////////////////////////////////////

/// TO DO!!!!!  (currently = linear blending)

template<class Out, class  _Material, int nbRef>
struct DualQuatBlendTypes<
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
    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlendTypes<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlendTypes<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }

};



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic
//////////////////////////////////////////////////////////////////////////////////

/// TO DO!!!!!  (currently = linear blending)

template<class Out, class _Material, int nbRef>
struct DualQuatBlendTypes<
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
    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlendTypes<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlendTypes<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }

};



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic -> Affine  =  -> DefGradient1 with dw=0
//////////////////////////////////////////////////////////////////////////////////

/// TO DO!!!!!  (currently = linear blending)

template<class Out, class _Material, int nbRef>
struct DualQuatBlendTypes<
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

    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlendTypes<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlendTypes<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic->DeformationGradient first order
//////////////////////////////////////////////////////////////////////////////////

/// TO DO!!!!!  (currently = linear blending)

template<class Out, class _Material, int nbRef>
struct DualQuatBlendTypes<
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
    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlendTypes<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlendTypes<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic->DeformationGradient second order
//////////////////////////////////////////////////////////////////////////////////

/// TO DO!!!!!  (currently = linear blending)

template<class Out, class _Material, int nbRef>
struct DualQuatBlendTypes<
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
    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlendTypes<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlendTypes<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};






//////////////////////////////////////////////////////////////////////////////////
////  Rigid->Vec
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct DualQuatBlendTypes<
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
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;
    typedef typename In::Deriv InDeriv;
    typedef DualQuatCoord<3,InReal> DQCoord;

    struct JacobianBlock
    {
        /** Dual quat skinning: p = sum_i w_i q_i*q0_i^-1 / |sum_i w_i q_i*q0_i^-1| (p_0)
          */

        Real w;
        Mat<4,4,Real> T0; Mat<4,4,Real> TE; ///< Real/Dual part of blended quaternion Jacobian : db = [T0,TE] dq

        Mat<3,3,Real> Pa; ///< dp = Pa.Omega_i  : affine part
        Mat<3,3,Real> Pt; ///< dp = Pt.dt_i : translation part
    };

    Vec<nbRef,unsigned int> index;
    OutCoord P0;	///< initial position
    DQCoord b;		///< linearly blended dual quaternions : b= sum_i w_i q_i*q0_i^-1
    DQCoord bn;		///< normalized dual quaternion : bn=b/|b|

    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        P0 = InitialPos;
        unsigned int i = 0 ;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            Jb[i].w= w[i];
            DQCoord T0inv=InitialTransform[index[i]]; T0inv.invert();
            T0inv.multLeft_getJ(Jb[i].T0,Jb[i].TE);
            Jb[i].T0*=w[i]; Jb[i].TE*=w[i];
        }
        if ( i<nbRef ) Jb[i].w=(Real)0; // used for loop terminations
        apply(InitialTransform);
    }


    OutCoord apply( const VecInCoord& in )  // Called in Apply
    {
        OutCoord result;

        DQCoord q;		// frame current position in DQ form
        b.clear();
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0.; i++ )
        {
            q=in[index[i]];
            // weighted relative transform : w_i.q_i*q0_i^-1
            q.getDual() = Jb[i].TE*q.getOrientation() + Jb[i].T0*q.getDual();
            q.getOrientation() = Jb[i].T0*q.getOrientation();
            b+=q;
        }

        bn=b; bn.normalize();
        Mat<4,4,Real> N0; Mat<4,4,Real> NE; // Real/Dual part of the normalization Jacobian : dbn = [N0,NE] db
        b.normalize_getJ( N0 , NE );

        result = bn.pointToParent( P0 );
        Mat<3,4,Real> Q0; Mat<3,4,Real> QE; // Real/Dual part of the transformation Jacobian : dP = [Q0,QE] dbn
        bn.pointToParent_getJ( Q0 , QE , P0 );

        Mat<3,4,Real> QN0 = Q0*N0 + QE*NE , QNE = QE * N0;
        Mat<4,3,Real> TL0 , TLE;

        for ( unsigned int i=0; i<nbRef && Jb[i].w>0.; i++ )
        {
            q=in[index[i]];
            q.velocity_getJ ( TL0 , TLE );  // Real/Dual part of quaternion Jacobian : dq_i = [L0,LE] [Omega_i, dt_i]
            TLE  = Jb[i].TE * TL0 + Jb[i].T0 * TLE;
            TL0  = Jb[i].T0 * TL0 ;
            // dP = QNTL [Omega_i, dt_i]
            Jb[i].Pa = QN0 * TL0 + QNE * TLE;
            Jb[i].Pt = QNE * TL0 ;
        }

        return result;
    }

    OutDeriv mult( const VecInDeriv& in ) // Called in ApplyJ
    {
        OutDeriv result;

        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            result += Jb[i].Pt * getLinear( in[index[i]] )  + Jb[i].Pa * getAngular(in[index[i]]);
        }
        return result;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            getLinear(res[index[i]])  += Jb[i].Pt.transposed() * d;
            getAngular(res[index[i]]) += Jb[i].Pa.transposed() * d;
        }
    }

    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            InDeriv parentJacobianVec;
            getLinear(parentJacobianVec)  += Jb[i].Pt.transposed() * childJacobianVec;
            getAngular(parentJacobianVec) += Jb[i].Pa.transposed() * childJacobianVec;
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }
    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlendTypes<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlendTypes<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }

};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Rigid -> Affine =  -> DefGradient1 with dw=0
//////////////////////////////////////////////////////////////////////////////////

/// TO DO!!!!!

template<class Out, class _Material, int nbRef>
struct DualQuatBlendTypes<
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

            Jb[i].Pa0.getAffine()= inverseInitialTransform33 * w[i];
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

    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlendTypes<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlendTypes<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Rigid -> Rigid =  -> Affine with polarDecomposition
//////////////////////////////////////////////////////////////////////////////////

/// TO DO!!!!!

template<class Out, class _Material, int nbRef>
struct DualQuatBlendTypes<
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
    Mat33 A,R,S,Sinv;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
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
            Jb[i].Pa0.getCenter()=vectorInLocalCoordinates*w[i];
            Jb[i].Pa.getCenter()=(InitialPos.getCenter() - InitialTransform[index[i]].getCenter() )*w[i];
            Jb[i].Pt=w[i];

            Jb[i].Pa0.getAffine()= inverseInitialTransform33 * w[i];
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
        polar_decomp(A, R, S);
        Sinv.invert(S);
        res.getOrientation().fromMatrix(R);
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
        //Adot ~ w x S
        Quat q; q.fromMatrix(Adot*Sinv);
        Real phi; q.quatToAxis(res.getAngular(), phi);
        res.getAngular()*=phi;

        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
        */
        //Adot ~ w x S
        Mat33 Adot=crossProductMatrix(d.getAngular())*S;

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
        //Adot ~ w x S
        Mat33 Adot=crossProductMatrix(childJacobianVec.getAngular())*S;

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



    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlendTypes<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 4>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlendTypes<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 4>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Rigid->DeformationGradient first order
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class  _Material, int nbRef>
struct DualQuatBlendTypes<
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
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;
    typedef typename In::Deriv InDeriv;
    typedef DualQuatCoord<3,InReal> DQCoord;

    struct JacobianBlock
    {
        /** Dual quat skinning: p = sum_i w_i q_i*q0_i^-1 / |sum_i w_i q_i*q0_i^-1| (p_0)
          */

        Real w;
        MaterialDeriv dw;

        Mat<4,4,Real> T0; Mat<4,4,Real> TE; ///< Real/Dual part of blended quaternion Jacobian : db = [T0,TE] dq

        Mat<3,3,Real> Pa; ///< dp = Pa.Omega_i  : affine part
        Mat<3,3,Real> Pt; ///< dp = Pt.dt_i : translation part

        Mat<9,3,Real> Fa; ///< dF = Fa.Omega_i  : affine part
        Mat<9,3,Real> Ft; ///< dF = Ft.dt_i : translation part
    };

    Vec<nbRef,unsigned int> index;
    SpatialCoord P0;	///< initial position
    DQCoord b;		///< linearly blended dual quaternions : b= sum_i w_i q_i*q0_i^-1
    DQCoord bn;		///< normalized dual quaternion : bn=b/|b|

    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& dw, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        P0 = InitialPos.getCenter();
        unsigned int i = 0 ;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            Jb[i].w= w[i];
            DQCoord T0inv=InitialTransform[index[i]]; T0inv.invert();
            T0inv.multLeft_getJ(Jb[i].T0,Jb[i].TE);
            Jb[i].T0*=w[i]; Jb[i].TE*=w[i];
            Jb[i].dw=dw[i];
        }
        if ( i<nbRef ) Jb[i].w=(Real)0; // used for loop terminations

        apply(InitialTransform);
    }

    OutCoord apply( const VecInCoord& in )  // Called in Apply
    {
        OutCoord res;

        unsigned int i,j,k,kk;

        DQCoord q;		// frame current position in DQ form
        b.clear();
        Mat<4,3,Real> W0 , WE; // Real/Dual part of the blended quaternion spatial deriv : db = [W0,WE] dp
        for ( i=0; i<nbRef && Jb[i].w>0.; i++ )
        {
            q=in[index[i]];
            // weighted relative transform : w_i.q_i*q0_i^-1
            q.getDual()= Jb[i].TE*q.getOrientation() + Jb[i].T0*q.getDual();
            q.getOrientation() = Jb[i].T0*q.getOrientation();
            b+=q;
            for ( j=0; j<4 ; j++ )
                for ( k=0; k<3 ; k++ )
                {
                    W0[j][k]= q.getOrientation()[j]*Jb[i].dw[k]/Jb[i].w;
                    WE[j][k]= q.getDual()[j]*Jb[i].dw[k]/Jb[i].w;
                }
        }

        bn=b; bn.normalize();
        Mat<4,4,Real> N0; Mat<4,4,Real> NE; // Real/Dual part of the normalization Jacobian : dbn = [N0,NE] db
        b.normalize_getJ( N0 , NE );

        res.getCenter() = bn.pointToParent( P0 );
        Mat<3,4,Real> Q0; Mat<3,4,Real> QE; // Real/Dual part of the transformation Jacobian : dP = [Q0,QE] dbn
        bn.pointToParent_getJ( Q0 , QE , P0 );

        Mat<3,4,Real> QN0 = Q0*N0 + QE*NE , QNE = QE*N0;
        bn.toRotationMatrix(res.getMaterialFrame());
        res.getMaterialFrame() += QN0*W0 + QNE*WE; // defgradient F = R + QNW
        Mat<4,3,Real> L0 , LE , TL0 , TLE;

        Mat<3,4,Real> QNT0 , QNTE ;
        Mat<4,3,Real> NTL0 , NTLE ;
        Mat<4,3,Real> NW0 = N0*W0 , NWE = NE*W0 + N0*WE;
        Mat<3,4,Real> dQ0 , dQE;
        Mat<4,4,Real> dN0 , dNE;
        Mat<4,3,Real> dW0 , dWE;
        DQCoord dq ;
        Mat<3,3,Real> dF;

        for ( i=0; i<nbRef && Jb[i].w>0.; i++ )
        {
            q=in[index[i]];
            q.velocity_getJ ( L0 , LE );  // Real/Dual part of quaternion Jacobian : dq_i = [L0,LE] [Omega_i, dt_i]
            TLE  = Jb[i].TE * L0 + Jb[i].T0 * LE;
            TL0  = Jb[i].T0 * L0 ;
            // dP = QNTL [Omega_i, dt_i]
            Jb[i].Pa = QN0 * TL0 + QNE * TLE;
            Jb[i].Pt = QNE * TL0 ;

            NTL0 = N0 * TL0 ;				NTLE = NE * TL0 + N0 * TLE;
            QNT0 = QN0 * Jb[i].T0 + QNE * Jb[i].TE;		QNTE = QNE * Jb[i].T0 ;
            QNT0/=Jb[i].w; QNTE/=Jb[i].w; // remove pre-multiplication of T by w

            // dF = dR + dQNW + QdNW + QNdW
            Jb[i].Fa.fill(0);
            Jb[i].Ft.fill(0);
            for ( j=0; j<3 ; j++ )
            {
                // dR
                for ( k=0; k<4 ; k++ )  {dq.getOrientation()[k]=NTL0[k][j]; dq.getDual()[k]=NTLE[k][j];}
                dF=bn.rotation_applyH(dq); for ( k=0; k<9 ; k++ ) Jb[i].Fa[k][j]+=MattoVec(dF)[k];

                // dQNW
                dq.pointToParent_getJ( dQ0 , dQE , P0 );
                dF = dQ0*NW0 + dQE*NWE ;  for ( k=0; k<9 ; k++ ) Jb[i].Fa[k][j]+=MattoVec(dF)[k];
                dq.getDual()=dq.getOrientation(); dq.getOrientation().fill(0);
                dq.pointToParent_getJ( dQ0 , dQE , P0 );
                dF = dQ0*NW0 ;  for ( k=0; k<9 ; k++ ) Jb[i].Ft[k][j]+=MattoVec(dF)[k];

                // QdNW
                for ( k=0; k<4 ; k++ )  {dq.getOrientation()[k]=TL0[k][j]; dq.getDual()[k]=TLE[k][j];}
                b.normalize_getdJ ( dN0 , dNE , dq );
                dF = Q0*dN0*W0 + QE*(dNE*W0 + dN0*WE);  for ( k=0; k<9 ; k++ ) Jb[i].Fa[k][j]+=MattoVec(dF)[k];
                dq.getDual()=dq.getOrientation(); dq.getOrientation().fill(0);
                b.normalize_getdJ ( dN0 , dNE , dq );
                dF = QE*dNE*W0 ;  for ( k=0; k<9 ; k++ ) Jb[i].Ft[k][j]+=MattoVec(dF)[k];

                // QNdW
                for ( k=0; k<4 ; k++ )
                    for ( kk=0; kk<3 ; kk++ )
                    {
                        dW0[k][kk]= L0[k][j]*Jb[i].dw[kk];
                        dWE[k][kk]= LE[k][j]*Jb[i].dw[kk];
                    }
                dF = QNT0*dW0 + QNTE*dWE;  for ( k=0; k<9 ; k++ ) Jb[i].Fa[k][j]+=MattoVec(dF)[k];
                dF = QNTE*dW0;  for ( k=0; k<9 ; k++ ) Jb[i].Ft[k][j]+=MattoVec(dF)[k];
            }
        }

        return res;
    }


    OutDeriv mult( const VecInDeriv& in ) // Called in ApplyJ
    {
        OutDeriv res;

        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            res.getCenter() += Jb[i].Pt * getLinear( in[index[i]] )  + Jb[i].Pa * getAngular(in[index[i]]);
            MattoVec(res.getMaterialFrame()) += Jb[i].Ft * getLinear( in[index[i]] )  + Jb[i].Fa * getAngular(in[index[i]]);
        }
        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            getLinear(res[index[i]])  += Jb[i].Pt.transposed() * d.getCenter();
            getAngular(res[index[i]]) += Jb[i].Pa.transposed() * d.getCenter();
            getLinear(res[index[i]])  += Jb[i].Ft.transposed() * MattoVec(d.getMaterialFrame());
            getAngular(res[index[i]]) += Jb[i].Fa.transposed() * MattoVec(d.getMaterialFrame());
        }
    }

    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0.; i++ )
        {

            InDeriv parentJacobianVec;
            getLinear(parentJacobianVec)  += Jb[i].Pt.transposed() * childJacobianVec.getCenter();
            getAngular(parentJacobianVec) += Jb[i].Pa.transposed() * childJacobianVec.getCenter();
            getLinear(parentJacobianVec)  += Jb[i].Ft.transposed() * MattoVec(childJacobianVec.getMaterialFrame());
            getAngular(parentJacobianVec) += Jb[i].Fa.transposed() * MattoVec(childJacobianVec.getMaterialFrame());
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }
    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlendTypes<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlendTypes<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }

};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Rigid->DeformationGradient second order
//////////////////////////////////////////////////////////////////////////////////


template<class Out, class  _Material, int nbRef>
struct DualQuatBlendTypes<
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
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;
    typedef typename In::Deriv InDeriv;
    typedef DualQuatCoord<3,InReal> DQCoord;

    struct JacobianBlock
    {
        /** Dual quat skinning: p = sum_i w_i q_i*q0_i^-1 / |sum_i w_i q_i*q0_i^-1| (p_0)
          */

        Real w;
        MaterialDeriv dw;
        MaterialMat ddw;

        Mat<4,4,Real> T0; Mat<4,4,Real> TE; ///< Real/Dual part of blended quaternion Jacobian : db = [T0,TE] dq

        Mat<3,3,Real> Pa; ///< dp = Pa.Omega_i  : affine part
        Mat<3,3,Real> Pt; ///< dp = Pt.dt_i : translation part

        Mat<9,3,Real> Fa; ///< dF = Fa.Omega_i  : affine part
        Mat<9,3,Real> Ft; ///< dF = Ft.dt_i : translation part

        Vec<3,Mat<9,3,Real> > dFa; ///< d gradF_k = dFa_k.Omega_i  : affine part
        Vec<3,Mat<9,3,Real> > dFt; ///< d gradF_k = dFt_k.dt_i : translation part
    };

    Vec<nbRef,unsigned int> index;
    SpatialCoord P0;	///< initial position
    DQCoord b;		///< linearly blended dual quaternions : b= sum_i w_i q_i*q0_i^-1
    DQCoord bn;		///< normalized dual quaternion : bn=b/|b|

    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& dw, const Vec<nbRef,MaterialMat>&  ddw)
    {
        index = Index;
        P0 = InitialPos.getCenter();
        unsigned int i = 0 ;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            Jb[i].w= w[i];
            DQCoord T0inv=InitialTransform[index[i]]; T0inv.invert();
            T0inv.multLeft_getJ(Jb[i].T0,Jb[i].TE);
            Jb[i].T0*=w[i]; Jb[i].TE*=w[i];
            Jb[i].dw=dw[i];
            Jb[i].ddw=ddw[i];
        }
        if ( i<nbRef ) Jb[i].w=(Real)0; // used for loop terminations
        apply(InitialTransform);
    }






    OutCoord apply( const Vec<nbRef,DQCoord>& in )
    {
        OutCoord res;

        unsigned int i,j,k,kk;

        DQCoord q,b2,bn2;
        b2.clear();
        Mat<4,3,Real> W0 , WE; // Real/Dual part of the blended quaternion spatial deriv : db = [W0,WE] dp
        /// specific to D332
        Vec<3,Mat<4,3,Real> > gradW0 , gradWE; // spatial deriv of W
        ///
        for ( i=0; i<nbRef && Jb[i].w>0.; i++ )
        {
            // weighted relative transform : w_i.q_i*q0_i^-1
            q=in[i];
            q.getDual()= Jb[i].TE*q.getOrientation() + Jb[i].T0*q.getDual();
            q.getOrientation() = Jb[i].T0*q.getOrientation();
            b2+=q;
            for ( j=0; j<4 ; j++ )
                for ( k=0; k<3 ; k++ )
                {
                    W0[j][k]= q.getOrientation()[j]*Jb[i].dw[k]/Jb[i].w;
                    WE[j][k]= q.getDual()[j]*Jb[i].dw[k]/Jb[i].w;
                    /// specific to D332
                    for ( kk=0; kk<3 ; kk++ )
                    {
                        gradW0[kk][j][k]= q.getOrientation()[j]*Jb[i].ddw[k][kk]/Jb[i].w;
                        gradWE[kk][j][k]= q.getDual()[j]*Jb[i].ddw[k][kk]/Jb[i].w;
                    }
                    ///
                }
        }

        bn2=b2; bn2.normalize();
        Mat<4,4,Real> N0; Mat<4,4,Real> NE; // Real/Dual part of the normalization Jacobian : dbn = [N0,NE] db
        b2.normalize_getJ( N0 , NE );
        res.getCenter() = bn2.pointToParent( P0 );
        Mat<3,4,Real> Q0; Mat<3,4,Real> QE; // Real/Dual part of the transformation Jacobian : dP = [Q0,QE] dbn
        bn2.pointToParent_getJ( Q0 , QE , P0 );
        Mat<3,4,Real> QN0 = Q0*N0 + QE*NE , QNE = QE*N0;
        bn2.toRotationMatrix(res.getMaterialFrame());
        res.getMaterialFrame() += QN0*W0 + QNE*WE; // defgradient F = R + QNW

        Mat<4,3,Real> NW0 = N0*W0 , NWE = NE*W0 + N0*WE;
        Mat<3,4,Real> dQ0 , dQE;
        Mat<4,4,Real> dN0 , dNE;
        Mat<3,3,Real> dF;
        DQCoord dq ;

        /// specific to D332
        // gradF = gradR + gradQ.N.W + Q.gradN.W + Q.N.gradW
        for ( j=0; j<3 ; j++ )
        {
            res.getMaterialFrameGradient()[j].fill(0);
            // gradR
            for ( k=0; k<4 ; k++ )  {dq.getOrientation()[k]=NW0[k][j]; dq.getDual()[k]=NWE[k][j];}
            dF=bn2.rotation_applyH(dq); res.getMaterialFrameGradient()[j]+=dF;
            // gradQ.N.W
            dq.pointToParent_getJ( dQ0 , dQE , P0 );
            dF = dQ0*NW0 + dQE*NWE ;  res.getMaterialFrameGradient()[j]+=dF;
            // Q.gradN.W
            for ( k=0; k<4 ; k++ )  {dq.getOrientation()[k]=W0[k][j]; dq.getDual()[k]=WE[k][j];}
            b2.normalize_getdJ ( dN0 , dNE , dq );
            dF = Q0*dN0*W0 + QE*(dNE*W0 + dN0*WE);  res.getMaterialFrameGradient()[j]+=dF;
            // Q.N.gradW
            dF = QN0*gradW0[j] + QNE*gradWE[j];  res.getMaterialFrameGradient()[j]+=dF;
        }

        return res;
    }




    OutCoord apply( const VecInCoord& in )  // Called in Apply
    {
        OutCoord res;

        unsigned int i,j,k,kk;

        Vec<nbRef,DQCoord> DQin;
        for ( i=0; i<nbRef && Jb[i].w>0.; i++ ) DQin[i]=in[index[i]];
        res=apply( DQin );

        Mat<4,3,Real> L0 , LE ;
        Mat<3,3,Real> dF;
        DQCoord dq;
        OutCoord res2;
        Real mult=(Real)0.000001;

        // here we do not compute the Jacobian dF/dq_i as in D331 (it is too complex for gradF)
        // so we estimate dF/dq_i ~ [ F(q_i+mult*dq_i)- F(q_i) ] /mult

        for ( i=0; i<nbRef && Jb[i].w>0.; i++ )
        {

            DQin[i].velocity_getJ ( L0 , LE );  // Real/Dual part of quaternion Jacobian : dq_i = [L0,LE] [Omega_i, dt_i]

            for ( j=0; j<3 ; j++ )
            {

                for ( k=0; k<4 ; k++ )  {dq.getOrientation()[k]=L0[k][j]; dq.getDual()[k]=LE[k][j];}
                DQin[i]+=dq*mult;

                res2=apply( DQin );

                for ( k=0; k<3 ; k++ ) Jb[i].Pa[k][j]=(res2.getCenter()[k]-res.getCenter()[k])/mult;
                dF=(res2.getMaterialFrame()-res.getMaterialFrame())/mult;
                for ( k=0; k<9 ; k++ ) Jb[i].Fa[k][j]=MattoVec(dF)[k];

                for ( kk=0; kk<3 ; kk++ )
                {
                    dF=(res2.getMaterialFrameGradient()[kk]-res.getMaterialFrameGradient()[kk])/mult;
                    for ( k=0; k<9 ; k++ ) Jb[i].dFa[kk][k][j]=MattoVec(dF)[k];
                }

                DQin[i]+=dq*(-mult);

                DQin[i].getDual()+=dq.getOrientation()*mult;

                res2=apply( DQin );

                for ( k=0; k<3 ; k++ ) Jb[i].Pt[k][j]=(res2.getCenter()[k]-res.getCenter()[k])/mult;
                dF=(res2.getMaterialFrame()-res.getMaterialFrame())/mult;
                for ( k=0; k<9 ; k++ ) Jb[i].Ft[k][j]=MattoVec(dF)[k];

                for ( kk=0; kk<3 ; kk++ )
                {
                    dF=(res2.getMaterialFrameGradient()[kk]-res.getMaterialFrameGradient()[kk])/mult;
                    for ( k=0; k<9 ; k++ ) Jb[i].dFt[kk][k][j]=MattoVec(dF)[k];
                }

                DQin[i].getDual()+=dq.getOrientation()*(-mult);

            }
        }

        return res;
    }


    OutDeriv mult( const VecInDeriv& in ) // Called in ApplyJ
    {
        OutDeriv res;

        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            res.getCenter() += Jb[i].Pt * getLinear( in[index[i]] )  + Jb[i].Pa * getAngular(in[index[i]]);
            MattoVec(res.getMaterialFrame()) += Jb[i].Ft * getLinear( in[index[i]] )  + Jb[i].Fa * getAngular(in[index[i]]);
            for (unsigned int k = 0; k < 3; ++k)
                MattoVec(res.getMaterialFrameGradient()[k]) += Jb[i].dFt[k] * getLinear( in[index[i]] )  + Jb[i].dFa[k] * getAngular(in[index[i]]);
        }
        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            getLinear(res[index[i]])  += Jb[i].Pt.transposed() * d.getCenter();
            getAngular(res[index[i]]) += Jb[i].Pa.transposed() * d.getCenter();
            getLinear(res[index[i]])  += Jb[i].Ft.transposed() * MattoVec(d.getMaterialFrame());
            getAngular(res[index[i]]) += Jb[i].Fa.transposed() * MattoVec(d.getMaterialFrame());
            for (unsigned int k = 0; k < 3; ++k)
            {
                getLinear(res[index[i]])  += Jb[i].dFt[k].transposed() * MattoVec(d.getMaterialFrameGradient()[k]);
                getAngular(res[index[i]]) += Jb[i].dFa[k].transposed() * MattoVec(d.getMaterialFrameGradient()[k]);
            }
        }
    }

    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0.; i++ )
        {

            InDeriv parentJacobianVec;
            getLinear(parentJacobianVec)  += Jb[i].Pt.transposed() * childJacobianVec.getCenter();
            getAngular(parentJacobianVec) += Jb[i].Pa.transposed() * childJacobianVec.getCenter();
            getLinear(parentJacobianVec)  += Jb[i].Ft.transposed() * MattoVec(childJacobianVec.getMaterialFrame());
            getAngular(parentJacobianVec) += Jb[i].Fa.transposed() * MattoVec(childJacobianVec.getMaterialFrame());
            for (unsigned int k = 0; k < 3; ++k)
            {
                getLinear(parentJacobianVec)  += Jb[i].dFt[k].transposed() * MattoVec(childJacobianVec.getMaterialFrameGradient()[k]);
                getAngular(parentJacobianVec) += Jb[i].dFa[k].transposed() * MattoVec(childJacobianVec.getMaterialFrameGradient()[k]);
            }
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }
    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlendTypes<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlendTypes<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }

};

//////////////////////////////////////////////////////////////////////////////////
}
}

#endif
