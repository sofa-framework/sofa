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
#define SOFA_COMPONENT_MAPPING_FRAMEBLENDINGMAPPING_CPP

#include "FrameBlendingMapping.inl"
#include "MappingTypes.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/gl/template.h>

namespace sofa
{
namespace defaulttype
{

/*
template<class _Real, int Dim>
inline const Vec<3,_Real>& center(const Vec<Dim,_Real>& c)
{
    return DeformationGradientTypes<3, 3, 2, _Real>::center(c);
}

template<class _Real, int Dim>
inline Vec<3,_Real>& center(Vec<Dim,_Real>& c)
{
    return DeformationGradientTypes<3, 3, 2, _Real>::center(c);
}

template<class _Real>
inline const Vec<3,_Real>& center(const Vec<3,_Real>& c)
{
    return c;
}

template<class _Real>
inline Vec<3,_Real>& center(Vec<3,_Real>& c)
{
    return c;
}


template<int N, class R > Mat<N,N,R> identity()
{
    Mat<N,N,R> m;
    for (unsigned i=0; i<N; i++)
        m[i][i] = (R)1.0;
    return m;
}

template<int N, class R > Mat<N,N,R> diag( const R& d )
{
    Mat<N,N,R> m;
    for (unsigned i=0; i<N; i++)
        m[i][i] = d;
    return m;
}
*/

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
struct LinearBlendTypes<
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

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        OutCoord Pa;    ///< = dp = dMa_i (w_i \bar M_i p_0)  : affine part
        Real Pt;      ///< = dp = dMt_i (w_i)  : translation part
    };

    Vec<nbRef,unsigned> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned i=0;
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
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            result += d[index[i]].getCenter() * Jb[i].Pt + d[index[i]].getAffine() * Jb[i].Pa;
        }
        return result;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutCoord result;
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            result += d[index[i]].getVCenter() * Jb[i].Pt + d[index[i]].getVAffine() * Jb[i].Pa;
        }
        return result;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */

        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {

            res[index[i]].getVCenter() += d * Jb[i].Pt;
            for (unsigned int j = 0; j < 3; ++j) res[index[i]].getVAffine()[j] += Jb[i].Pa * d[j];
        }
    }


};



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine->DeformationGradient first order
//////////////////////////////////////////////////////////////////////////////////
template<class Out, class _Material, int nbRef>
struct LinearBlendTypes<
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

    Vec<nbRef,unsigned> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& dw, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned i=0;
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
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res.getCenter() += d[index[i]].getCenter( ) * Jb[i].Pt + d[index[i]].getAffine( ) * Jb[i].Pa;
            res.getMaterialFrame() += covNN( d[index[i]].getCenter(), Jb[i].Ft) + d[index[i]].getAffine() * Jb[i].Fa;
        }
        return res;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res.getCenter() += d[index[i]].getVCenter( ) * Jb[i].Pt + d[index[i]].getVAffine( ) * Jb[i].Pa;
            res.getMaterialFrame() += covNN( d[index[i]].getVCenter(), Jb[i].Ft) + d[index[i]].getVAffine( ) * Jb[i].Fa;
        }
        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {

            res[index[i]].getVCenter() +=  d.getCenter() * Jb[i].Pt;
            res[index[i]].getVCenter() += d.getMaterialFrame() * Jb[i].Ft;

            for (unsigned int j = 0; j < 3; ++j)
            {
                res[index[i]].getVAffine()[j] += Jb[i].Pa * d.getCenter()[j];
                res[index[i]].getVAffine()[j] += Jb[i].Fa * (d.getMaterialFrame()[j]);
            }
        }
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine->DeformationGradient second order
//////////////////////////////////////////////////////////////////////////////////


template<class Out, class  _Material, int nbRef>
struct LinearBlendTypes<
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

    Vec<nbRef,unsigned> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& dw, const Vec<nbRef,MaterialMat>&  ddw)
    {
        index = Index;
        unsigned i=0;
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
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res.getCenter() += d[index[i]].getCenter( ) * Jb[i].Pt + d[index[i]].getAffine( ) * Jb[i].Pa;
            res.getMaterialFrame() += covNN( d[index[i]].getCenter( ), Jb[i].Ft) + d[index[i]].getAffine( ) * Jb[i].Fa;
            for (unsigned int k = 0; k < 3; ++k)
                res.getMaterialFrameGradient()[k] += covNN( d[index[i]].getCenter( ), Jb[i].dFt[k]) + d[index[i]].getAffine( ) * Jb[i].dFa[k];
        }
        return res;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
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
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
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

};



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic
//////////////////////////////////////////////////////////////////////////////////
template<class Out, class _Material, int nbRef>
struct LinearBlendTypes<
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

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        QuadraticCoord Pa;    ///< = dp = dMa_i (w_i \bar M_i p_0)  : affine part
        Real Pt;      ///< = dp = dMt_i (w_i)  : translation part
    };

    Vec<nbRef,unsigned> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned i=0;
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
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res +=  d[index[i]].getCenter() * Jb[i].Pt +  d[index[i]].getQuadratic() * Jb[i].Pa;
        }
        return res;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res +=  d[index[i]].getVCenter() * Jb[i].Pt +  d[index[i]].getVQuadratic() * Jb[i].Pa;
        }
        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            res[index[i]].getVCenter() += d * Jb[i].Pt;
            for (unsigned int j = 0; j < 3; ++j) res[index[i]].getVQuadratic()[j] += Jb[i].Pa * d[j];
        }
    }

};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic->DeformationGradient first order
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct LinearBlendTypes<
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

    Vec<nbRef,unsigned> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& dw, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned i=0;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            InCoord inverseInitialTransform = In::inverse(InitialTransform[index[i]]);
            Affine invaff = inverseInitialTransform.getAffine();
            QuadraticCoord vectorInLocalCoordinates = In::convertToQuadraticCoord( (invaff*InitialPos.getCenter() + inverseInitialTransform.getCenter()) );
            Jb[i].Pa=vectorInLocalCoordinates*w[i];
            Jb[i].Pt=w[i];
            Jb[i].Fa=covMN(vectorInLocalCoordinates, dw[i]);
            // use only the inverse of the linear part (affine+translation), squared and crossterms part undefined
            for (unsigned int ii=0; ii<3; ++ii) for (unsigned int j=0; j<3; ++j)  Jb[i].Fa[ii][j]+=invaff[ii][j] * w[i];
            Jb[i].Ft=dw[i];
        }
        if ( i<nbRef ) Jb[i].Pt=(Real)0; // used for loop terminations
    }


    OutCoord apply( const VecInCoord& d ) // Called in Apply
    {
        OutCoord res;
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res.getCenter() += d[index[i]].getCenter( ) * Jb[i].Pt + d[index[i]].getQuadratic ( ) * Jb[i].Pa;
            res.getMaterialFrame() += covNN( d[index[i]].getCenter( ), Jb[i].Ft) + d[index[i]].getQuadratic( ) * Jb[i].Fa;
        }
        return res;
    }

    OutDeriv mult( const VecInDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            res.getCenter() += d[index[i]].getVCenter( ) * Jb[i].Pt + d[index[i]].getVQuadratic ( ) * Jb[i].Pa;
            res.getMaterialFrame() += covNN( d[index[i]].getVCenter( ), Jb[i].Ft) + d[index[i]].getVQuadratic( ) * Jb[i].Fa;
        }
        return res;
    }


    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {

            res[index[i]].getVCenter() += d.getCenter() * Jb[i].Pt;
            res[index[i]].getVCenter() += d.getMaterialFrame() * Jb[i].Ft;

            for (unsigned int j = 0; j < 3; ++j)
            {
                res[index[i]].getVQuadratic()[j] += Jb[i].Pa * d.getCenter()[j];
                res[index[i]].getVQuadratic()[j] += Jb[i].Fa * (d.getMaterialFrame()[j]);
            }
        }
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic->DeformationGradient second order
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct LinearBlendTypes<
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

    Vec<nbRef,unsigned> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& dw, const Vec<nbRef,MaterialMat>&  ddw)
    {
        index = Index;
        unsigned i=0;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            InCoord inverseInitialTransform = In::inverse(InitialTransform[index[i]]);
            Affine invaff = inverseInitialTransform.getAffine();
            QuadraticCoord vectorInLocalCoordinates = In::convertToQuadraticCoord( (invaff*InitialPos.getCenter() + inverseInitialTransform.getCenter()) );
            Jb[i].Pa=vectorInLocalCoordinates*w[i];
            Jb[i].Pt=w[i];
            Jb[i].Fa=covMN(vectorInLocalCoordinates, dw[i]);
            // use only the inverse of the linear part (affine+translation), squared and crossterms part undefined
            for (unsigned int ii=0; ii<3; ++ii) for (unsigned int j=0; j<3; ++j)  Jb[i].Fa[ii][j]+=invaff[ii][j] * w[i];
            Jb[i].Ft=dw[i];

            Jb[i].dFt=ddw[i].transposed();
            MaterialFrame inverseInitialTransformT=invaff.transposed();
            for (unsigned int k = 0; k < 3; ++k)
            {
                Jb[i].dFa[k] = covMN( vectorInLocalCoordinates, Jb[i].dFt[k]);
                MaterialFrame m=covNN(inverseInitialTransformT[k],dw[i] ); // dFa
                for (unsigned int ii=0; ii<3; ++ii) for (unsigned int j=0; j<3; ++j)  Jb[i].dFa[k][ii][j]+=m[ii][j]+inverseInitialTransform.getAffine()[ii][j] * dw[i][k];
            }
        }
        if ( i<nbRef ) Jb[i].Pt=(Real)0; // used for loop terminations
    }

    OutCoord apply( const VecInCoord& d ) // Called in Apply
    {
        OutCoord res;
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
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
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
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
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
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
};



//////////////////////////////////////////////////////////////////////////////////
////  Rigid->Vec
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct LinearBlendTypes<
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

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        OutCoord Pa0; ///< weighted point in local frame:   dp = dMa_i (w_i \bar M_i p_0)  : affine part
        OutCoord Pa;  ///< rotated point :  dp = Omega_i x [ Ma_i (w_i \bar M_i p_0) ]  : affine part
        Real Pt;      ///< = dp = dMt_i (w_i)  : translation part
    };

    Vec<nbRef,unsigned> index;
    Vec<nbRef,JacobianBlock> Jb;


    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned i=0;
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
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {
            Jb[i].Pa=in[index[i]].rotate(Jb[i].Pa0); // = update of J according to current transform
            result += in[index[i]].getCenter() * Jb[i].Pt + Jb[i].Pa;
        }
        return result;
    }

    OutDeriv mult( const VecInDeriv& in ) // Called in ApplyJ
    {
        OutDeriv result;
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {
            result += getLinear( in[index[i]] ) * Jb[i].Pt + cross(getAngular(in[index[i]]), Jb[i].Pa);
        }
        return result;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec6 product, and apply the transpose of this matrix
          */
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0; i++ )
        {
            getLinear(res[index[i]])  += d * Jb[i].Pt;
            getAngular(res[index[i]]) += cross(Jb[i].Pa, d);
        }
    }

};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Rigid->DeformationGradient first order
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class  _Material, int nbRef>
struct LinearBlendTypes<
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

    Vec<nbRef,unsigned> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& dw, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        unsigned i=0;
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
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
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
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            res.getCenter() +=  getLinear( d[index[i]] ) * Jb[i].Pt +  cross(getAngular(d[index[i]]), Jb[i].Pa);
            const Mat33& Wx=crossProductMatrix(getAngular(d[index[i]]));
            res.getMaterialFrame() += covNN( getLinear( d[index[i]] ) , Jb[i].Ft) + Wx * Jb[i].Fa;
        }
        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
        {

            getLinear(res[index[i]]) +=  d.getCenter() * Jb[i].Pt;
            getLinear(res[index[i]]) += d.getMaterialFrame() * Jb[i].Ft;

            getAngular(res[index[i]]) += cross(Jb[i].Pa, d.getCenter());

            getAngular(res[index[i]])[0] += dot(Jb[i].Fa[1],d.getMaterialFrame()[2]) - dot(Jb[i].Fa[2],d.getMaterialFrame()[1]);
            getAngular(res[index[i]])[1] += dot(Jb[i].Fa[2],d.getMaterialFrame()[0]) - dot(Jb[i].Fa[0],d.getMaterialFrame()[2]);
            getAngular(res[index[i]])[2] += dot(Jb[i].Fa[0],d.getMaterialFrame()[1]) - dot(Jb[i].Fa[1],d.getMaterialFrame()[0]);
        }
    }

};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Rigid->DeformationGradient second order
//////////////////////////////////////////////////////////////////////////////////


template<class Out, class  _Material, int nbRef>
struct LinearBlendTypes<
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

    Vec<nbRef,unsigned> index;
    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& dw, const Vec<nbRef,MaterialMat>&  ddw)
    {
        index = Index;
        unsigned i=0;
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
            for (unsigned int k = 0; k < 3; ++k)
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
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
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
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
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
        for ( unsigned i=0; i<nbRef && Jb[i].Pt>0.; i++ )
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

};

}
//////////////////////////////////////////////////////////////////////////////////

namespace component
{

namespace mapping
{
SOFA_DECL_CLASS(FrameBlendingMapping);

using namespace defaulttype;
using namespace core;

//////////////////////////////////////////////////////////////////////////////////
////  Instanciations
//////////////////////////////////////////////////////////////////////////////////

// Register in the Factory
int FrameBlendingMappingClass = core::RegisterObject("skin a model from a set of rigid dofs")

#ifndef SOFA_FLOAT
        .add< FrameBlendingMapping< Affine3dTypes, Vec3dTypes > >()
        .add< FrameBlendingMapping< Affine3dTypes, ExtVec3dTypes > >()
        .add< FrameBlendingMapping< Affine3dTypes, DeformationGradient331dTypes > >()
        .add< FrameBlendingMapping< Affine3dTypes, DeformationGradient332dTypes > >()
        .add< FrameBlendingMapping< Quadratic3dTypes, Vec3dTypes > >()
        .add< FrameBlendingMapping< Quadratic3dTypes, ExtVec3dTypes > >()
        .add< FrameBlendingMapping< Quadratic3dTypes, DeformationGradient331dTypes > >()
        .add< FrameBlendingMapping< Quadratic3dTypes, DeformationGradient332dTypes > >()
        .add< FrameBlendingMapping< Rigid3dTypes, Vec3dTypes > >()
        .add< FrameBlendingMapping< Rigid3dTypes, ExtVec3dTypes > >()
        .add< FrameBlendingMapping< Rigid3dTypes, DeformationGradient331dTypes > >()
        .add< FrameBlendingMapping< Rigid3dTypes, DeformationGradient332dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        // .add< FrameBlendingMapping< Affine3fTypes, Vec3fTypes > >()
        // .add< FrameBlendingMapping< Affine3fTypes, ExtVec3fTypes > >()
        //.add< FrameBlendingMapping< Affine3dTypes, DeformationGradient332fTypes > >()
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< FrameBlendingMapping< Affine3dTypes, Vec3fTypes > >()
        .add< FrameBlendingMapping< Affine3dTypes, ExtVec3fTypes > >()
//                                .add< FrameBlendingMapping< Affine3fTypes, ExtVec3dTypes > >()
        .add< FrameBlendingMapping< Quadratic3dTypes, Vec3fTypes > >()
        .add< FrameBlendingMapping< Quadratic3dTypes, ExtVec3fTypes > >()
        .add< FrameBlendingMapping< Rigid3dTypes, Vec3fTypes > >()
        .add< FrameBlendingMapping< Rigid3dTypes, ExtVec3fTypes > >()
#endif
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, Vec3dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, ExtVec3dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, DeformationGradient331dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, DeformationGradient332dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, Vec3dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, ExtVec3dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, DeformationGradient331dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, DeformationGradient332dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, Vec3dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, ExtVec3dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, DeformationGradient331dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, DeformationGradient332dTypes >;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
//template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, Vec3fTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, ExtVec3fTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, DeformationGradient332fTypes >;
#endif //SOFA_DOUBLE
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, DeformationGradient331fTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, Vec3fTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, ExtVec3fTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, ExtVec3dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, Vec3fTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, ExtVec3fTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, Vec3fTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, ExtVec3fTypes >;
#endif //SOFA_DOUBLE
#endif //SOFA_FLOAT


} // namespace mapping

} // namespace component

} // namespace sofa

