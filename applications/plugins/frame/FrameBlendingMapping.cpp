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

#include "AffineTypes.h"
#include "QuadraticTypes.h"
#include <sofa/defaulttype/RigidTypes.h>
#include "FrameBlendingMapping.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace defaulttype
{

//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine->Points types
//////////////////////////////////////////////////////////////////////////////////

template<class _Material>
struct LinearBlendTypes<
        StdAffineTypes<3,typename _Material::Real>,
        StdVectorTypes< Vec<3,typename _Material::Real>, Vec<3,typename _Material::Real>, typename _Material::Real >,
        _Material
        >
{
    typedef _Material Material;
    typedef typename Material::Real Real;
    typedef typename Material::VecReal VecReal;
    typedef typename Material::Gradient MaterialDeriv;
    typedef typename Material::VecGradient VecMaterialDeriv;
    typedef typename Material::Hessian MaterialMat;
    typedef typename Material::VecHessian VecMaterialMat;
    typedef StdAffineTypes<3,Real> In;
    typedef StdVectorTypes< Vec<3,Real>, Vec<3,Real>, Real > Out;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename Out::VecCoord VecOutCoord;
    typedef vector<unsigned> VecIndex;

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        JacobianBlock() {}
        JacobianBlock(const OutCoord& o, const Real w):Pa(o),Pt(w) {}
        OutCoord Pa;		///< = dp = dMa_i (w_i \bar M_i p_0)  : affine part
        Real Pt;			///< = dp = dMt_i (w_i)  : translation part
    };

    InCoord inverseInitialTransform;
    JacobianBlock Jb;

//void init( const VecInCoord&, const VecOutCoord&, const VecIndex&, const VecReal&, const VecMaterialDeriv&, const VecMaterialMat&  ){}
    void init( const InCoord& InitialTransform, const OutCoord& InitialPos, const Real& w, const MaterialDeriv& /*dw*/, const MaterialMat&  /*ddw*/)
    {
        inverseInitialTransform = In::inverse(InitialTransform);
        Jb.Pa=(inverseInitialTransform.getAffine()*InitialPos + inverseInitialTransform.getCenter()) *w;
        Jb.Pt=w;
    }

    void updateJacobian( const InCoord& /*currentTransform*/)
    {
        return ;
    }


    OutCoord mult( const InCoord& d ) // Called in Apply
    {
        return d.getCenter() * Jb.Pt + d.getAffine() * Jb.Pa;
    }


    OutDeriv mult( const InDeriv& d ) // Called in ApplyJ
    {
        return getVCenter( d ) * Jb.Pt + getVAffine( d ) * Jb.Pa;
    }

    InDeriv multTranspose( const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */
        InDeriv res;
        for (unsigned int i = 0; i < 3; ++i)
        {
            getVCenter(res)[i] = Jb.Pt * d[i];
            getVAffine(res)[i] = Jb.Pa * d[i];
        }
        return res;
    }

    //return InDeriv (
    //        Jb.Pa[0]*d[0], Jb.Pa[1]*d[0], Jb.Pa[2]*d[0], d[0]*Jb.Pt,
    //        Jb.Pa[0]*d[1], Jb.Pa[1]*d[1], Jb.Pa[2]*d[1], d[1]*Jb.Pt,
    //        Jb.Pa[0]*d[2], Jb.Pa[1]*d[2], Jb.Pa[2]*d[2], d[2]*Jb.Pt
    //        );


};



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine->DeformationGradient first order
//////////////////////////////////////////////////////////////////////////////////

template<class _Material>
struct LinearBlendTypes<
        StdAffineTypes<3,typename _Material::Real>,
        DeformationGradientTypes<3, 3, 1, typename  _Material::Real>,
        _Material
        >
{
    typedef _Material Material;
    typedef typename Material::Real Real;
    typedef typename Material::VecReal VecReal;
    typedef typename Material::Gradient MaterialDeriv;
    typedef typename Material::VecGradient VecMaterialDeriv;
    typedef typename Material::Hessian MaterialMat;
    typedef typename Material::VecHessian VecMaterialMat;
    typedef StdAffineTypes<3,Real> In;
    typedef DeformationGradientTypes<3, 3, 1, Real> Out;
    typedef typename Out::SpatialCoord SpatialCoord; // = Vec3
    typedef typename Out::MaterialFrame MaterialFrame;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename Out::VecCoord VecOutCoord;
    typedef vector<unsigned> VecIndex;

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        JacobianBlock() {}
        JacobianBlock(const OutCoord& o, const Real w, MaterialDeriv dw):Pa(Out::center(o)),Pt(w),Fa(Out::materialFrame(o)),Ft(dw) {}
        SpatialCoord Pa;	///< = dp = dMa_i (w_i \bar M_i p_0)  : affine part
        Real Pt;			///< = dp = dMt_i (w_i)  : translation part
        MaterialFrame Fa;  ///< = dF = dMa_i (w_i \bar M_i + \bar M_i p_0 dw_i)
        MaterialDeriv Ft;	///< = dF = dMt_i (dw_i)
    };

    InCoord inverseInitialTransform;
    JacobianBlock Jb;

    void init( const InCoord& InitialTransform, const OutCoord& InitialPos, const Real& w, const MaterialDeriv& dw, const MaterialMat&  /*ddw*/)
    {
        inverseInitialTransform = In::inverse(InitialTransform);
        const SpatialCoord& vectorInLocalCoordinates = (inverseInitialTransform.getAffine()*Out::center(InitialPos) + inverseInitialTransform.getCenter());
        Jb.Pa=vectorInLocalCoordinates*w;
        Jb.Pt=w;
        Jb.Fa=inverseInitialTransform.getAffine() * w + covNN( vectorInLocalCoordinates, dw);
        Jb.Ft=dw;
    }

    void updateJacobian( const InCoord& /*currentTransform*/)
    {
        return ;
    }


    OutCoord mult( const InCoord& d ) // Called in Apply
    {
        OutCoord res;
        Out::center(res) = d.getCenter( ) * Jb.Pt + d.getAffine( ) * Jb.Pa;
        Out::materialFrame(res) = covNN( d.getCenter( ), Jb.Ft) + d.getAffine( ) * Jb.Fa;
        return res;
    }

    OutDeriv mult( const InDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        Out::center(res) = getVCenter( d ) * Jb.Pt + getVAffine( d ) * Jb.Pa;
        Out::materialFrame(res) = covNN( getVCenter( d ), Jb.Ft) + getVAffine( d ) * Jb.Fa;
        return res;
    }

    InDeriv multTranspose( const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */
        InDeriv res;
        for (unsigned int i = 0; i < 3; ++i)
        {
            getVCenter(res)[i] = Jb.Pt * Out::center(d)[i];
            getVAffine(res)[i] = Jb.Pa * Out::center(d)[i];

            for (unsigned int j = 0; j < 3; ++j) getVCenter(res)[i] += Jb.Ft[j] * Out::materialFrame(d)[i][j];
            getVAffine(res)[i] += Jb.Fa * (Out::materialFrame(d)[i]);
        }
        return res;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine->DeformationGradient second order
//////////////////////////////////////////////////////////////////////////////////


template<class  _Material>
struct LinearBlendTypes<
        StdAffineTypes<3,typename _Material::Real>,
        DeformationGradientTypes<3, 3, 2, typename _Material::Real>,
        _Material
        >
{
    typedef _Material Material;
    typedef typename Material::Real Real;
    typedef typename Material::VecReal VecReal;
    typedef typename Material::Gradient MaterialDeriv;
    typedef typename Material::VecGradient VecMaterialDeriv;
    typedef typename Material::Hessian MaterialMat;
    typedef typename Material::VecHessian VecMaterialMat;
    typedef StdAffineTypes<3,Real> In;
    typedef DeformationGradientTypes<3, 3, 2, Real> Out;
    typedef typename Out::SpatialCoord SpatialCoord; // = Vec3
    typedef typename Out::MaterialFrame MaterialFrame;
    typedef typename Out::MaterialFrameGradient MaterialFrameGradient;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename Out::VecCoord VecOutCoord;
    typedef vector<unsigned> VecIndex;

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        JacobianBlock() {}
        JacobianBlock(const OutCoord& o, const Real w, MaterialDeriv dw,MaterialMat ddwT):Pa(Out::center(o)),Pt(w),Fa(Out::materialFrame(o)),Ft(dw),dFa(Out::materialFrameGradient(o)),dFt(ddwT) {}
        SpatialCoord Pa;	///< = dp = dMa_i (w_i \bar M_i p_0)  : affine part
        Real Pt;			///< = dp = dMt_i (w_i)  : translation part
        MaterialFrame Fa;  ///< = dF = dMa_i (w_i \bar M_i + \bar M_i p_0 dw_i)
        MaterialDeriv Ft;	///< = dF = dMt_i (dw_i)
        MaterialFrameGradient dFa;  ///< = d gradF_k = dMa_i ( grad(w_i)_k \bar M_i + \bar M_i p_0 grad(dw_i)_k + grad(\bar M_i p_0)_k dw_i)
        MaterialMat dFt;	///< = d gradF_k = dMt_i (grad(dw_i)_k)
    };

    InCoord inverseInitialTransform;
    JacobianBlock Jb;

    void init( const InCoord& InitialTransform, const OutCoord& InitialPos, const Real& w, const MaterialDeriv& dw, const MaterialMat&  ddw)
    {
        inverseInitialTransform = In::inverse(InitialTransform);
        const SpatialCoord& vectorInLocalCoordinates = (inverseInitialTransform.getAffine()*Out::center(InitialPos) + inverseInitialTransform.getCenter());
        Jb.Pa=vectorInLocalCoordinates*w;
        Jb.Pt=w;
        Jb.Fa=inverseInitialTransform.getAffine() * w + covNN( vectorInLocalCoordinates, dw);
        Jb.Ft=dw;
        Jb.dFt=ddw.transposed();
        const MaterialFrame& inverseInitialTransformT=inverseInitialTransform.getAffine().transposed();
        for (unsigned int k = 0; k < 3; ++k) Jb.dFa[k] = inverseInitialTransform.getAffine() * dw[k] + covNN( vectorInLocalCoordinates, Jb.dFt[k]) + covNN(inverseInitialTransformT[k],dw); // dFa
    }

    void updateJacobian( const InCoord& /*currentTransform*/)
    {
        return ;
    }

    OutCoord mult( const InCoord& d ) // Called in Apply
    {
        OutCoord res;
        Out::center(res) = d.getCenter( ) * Jb.Pt + d.getAffine( ) * Jb.Pa;
        Out::materialFrame(res) = covNN( d.getCenter( ), Jb.Ft) + d.getAffine( ) * Jb.Fa;
        for (unsigned int k = 0; k < 3; ++k) Out::materialFrameGradient(res)[k] = covNN( d.getCenter( ), Jb.dFt[k]) + d.getAffine( ) * Jb.dFa[k];
        return res;
    }

    OutDeriv mult( const InDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        Out::center(res) = getVCenter( d ) * Jb.Pt + getVAffine( d ) * Jb.Pa;
        Out::materialFrame(res) = covNN( getVCenter( d ), Jb.Ft) + getVAffine( d ) * Jb.Fa;
        for (unsigned int k = 0; k < 3; ++k) Out::materialFrameGradient(res)[k] = covNN( getVCenter( d ), Jb.dFt[k]) + getVAffine( d ) * Jb.dFa[k];
        return res;
    }


    InDeriv multTranspose( const OutDeriv& d ) // Called in ApplyJT
    {
        //To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
        InDeriv res;
        for (unsigned int i = 0; i < 3; ++i)
        {
            getVCenter(res)[i] = Jb.Pt * Out::center(d)[i];
            getVAffine(res)[i] = Jb.Pa * Out::center(d)[i];

            for (unsigned int j = 0; j < 3; ++j) getVCenter(res)[i] += Jb.Ft[j] * Out::materialFrame(d)[i][j];
            getVAffine(res)[i] += Jb.Fa * (Out::materialFrame(d)[i]);

            for (unsigned int k = 0; k < 3; ++k)
            {
                for (unsigned int j = 0; j < 3; ++j) getVCenter(res)[i] += Jb.dFt[k][j] * (Out::materialFrameGradient(d))[k][i][j];
                getVAffine(res)[i] += Jb.dFa[k] * (Out::materialFrameGradient(d)[k][i]);
            }
        }
        return res;
    }
};



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic types
//////////////////////////////////////////////////////////////////////////////////
template<class _Material>
struct LinearBlendTypes<
        StdQuadraticTypes<3,typename _Material::Real>,
        StdVectorTypes< Vec<3,typename _Material::Real>, Vec<3,typename _Material::Real>, typename _Material::Real >,
        _Material
        >
{
    typedef _Material Material;
    typedef typename Material::Real Real;
    typedef typename Material::VecReal VecReal;
    typedef typename Material::Gradient MaterialDeriv;
    typedef typename Material::VecGradient VecMaterialDeriv;
    typedef typename Material::Hessian MaterialMat;
    typedef typename Material::VecHessian VecMaterialMat;
    typedef StdQuadraticTypes<3,Real> In;
    typedef StdVectorTypes< Vec<3,Real>, Vec<3,Real>, Real > Out;
    typedef typename In::Coord InCoord;
    typedef typename InCoord::Pos2 SpatialCoord2; // vec9
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename Out::VecCoord VecOutCoord;
    typedef vector<unsigned> VecIndex;

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        JacobianBlock() {}
        JacobianBlock(const SpatialCoord2& o2, const Real w):Pa(o2),Pt(w) {}
        SpatialCoord2 Pa;		///< = dp = dMa_i (w_i \bar M_i p_0)  : affine part
        Real Pt;			///< = dp = dMt_i (w_i)  : translation part
    };

    InCoord inverseInitialTransform;
    JacobianBlock Jb;

    void init( const InCoord& InitialTransform, const OutCoord& InitialPos, const Real& w, const MaterialDeriv& /*dw*/, const MaterialMat&  /*ddw*/)
    {
        inverseInitialTransform = In::inverse(InitialTransform);
        const SpatialCoord2& vectorInLocalCoordinates = In::convertToQuadraticCoord( (inverseInitialTransform.getAffine()*InitialPos + inverseInitialTransform.getCenter()) );
        Jb.Pa=vectorInLocalCoordinates*w;
        Jb.Pt=w;
    }

    void updateJacobian( const InCoord& /*currentTransform*/)
    {
        return ;
    }

    OutCoord mult( const InCoord& d ) // Called in Apply
    {
        return d.getCenter( ) * Jb.Pt + d.getQuadratic( ) * Jb.Pa;
    }

    OutDeriv mult( const InDeriv& d ) // Called in ApplyJ
    {
        return getVCenter( d ) * Jb.Pt + getVQuadratic( d ) * Jb.Pa;
    }

    InDeriv multTranspose( const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */
        InDeriv res;
        for (unsigned int i = 0; i < 3; ++i)
        {
            getVCenter(res)[i] = Jb.Pt * d[i];
            getVQuadratic(res)[i] = Jb.Pa * d[i];
        }
        return res;
    }

};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic->DeformationGradient first order
//////////////////////////////////////////////////////////////////////////////////

template<class _Material>
struct LinearBlendTypes<
        StdQuadraticTypes<3,typename _Material::Real>,
        DeformationGradientTypes<3, 3, 1, typename  _Material::Real>,
        _Material
        >
{
    typedef _Material Material;
    typedef typename Material::Real Real;
    typedef typename Material::VecReal VecReal;
    typedef typename Material::Gradient MaterialDeriv;
    typedef typename Material::VecGradient VecMaterialDeriv;
    typedef typename Material::Hessian MaterialMat;
    typedef typename Material::VecHessian VecMaterialMat;
    typedef StdQuadraticTypes<3,Real> In;
    typedef DeformationGradientTypes<3, 3, 1, Real> Out;
    typedef typename Out::SpatialCoord SpatialCoord; // = Vec3
    typedef typename Out::MaterialFrame MaterialFrame;
    typedef typename In::Coord InCoord;
    typedef typename In::Vec9 SpatialCoord2; // vec9
    typedef Mat<9,3,Real> MaterialFrame2;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename Out::VecCoord VecOutCoord;
    typedef vector<unsigned> VecIndex;

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        JacobianBlock() {}
        JacobianBlock(const SpatialCoord2& p2, const Real w,  const MaterialFrame2& f2,  const MaterialDeriv dw):Pa(p2),Pt(w),Fa(f2),Ft(dw) {}
        SpatialCoord2 Pa;	///< = dp = dMa_i (w_i \bar M_i p_0)  : affine part
        Real Pt;			///< = dp = dMt_i (w_i)  : translation part
        MaterialFrame2 Fa;  ///< = dF = dMa_i (w_i \bar M_i + \bar M_i p_0 dw_i)
        MaterialDeriv Ft;	///< = dF = dMt_i (dw_i)
    };

    InCoord inverseInitialTransform;
    JacobianBlock Jb;

    void init( const InCoord& InitialTransform, const OutCoord& InitialPos, const Real& w, const MaterialDeriv& dw, const MaterialMat&  /*ddw*/)
    {
        inverseInitialTransform = In::inverse(InitialTransform);
        const SpatialCoord2& vectorInLocalCoordinates = In::convertToQuadraticCoord( (inverseInitialTransform.getAffine()*Out::center(InitialPos) + inverseInitialTransform.getCenter()) );
        Jb.Pa=vectorInLocalCoordinates*w;
        Jb.Pt=w;
        Jb.Fa=covMN(vectorInLocalCoordinates, dw);
        // use only the inverse of the linear part (affine+translation), squared and crossterms part undefined
        for(unsigned int i=0; i<3; ++i) for(unsigned int j=0; j<3; ++j)  Jb.Fa[i][j]+=inverseInitialTransform.getAffine()[i][j] * w;
        Jb.Ft=dw;
    }

    void updateJacobian( const InCoord& /*currentTransform*/)
    {
        return ;
    }


    OutCoord mult( const InCoord& d ) // Called in Apply
    {
        OutCoord res;
        Out::center(res) = d.getCenter( ) * Jb.Pt + d.getQuadratic ( ) * Jb.Pa;
        Out::materialFrame(res) = covNN( d.getCenter( ), Jb.Ft) + d.getQuadratic( ) * Jb.Fa;
        return res;
    }

    OutDeriv mult( const InDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        Out::center(res) = getVCenter( d ) * Jb.Pt + getVQuadratic ( d ) * Jb.Pa;
        Out::materialFrame(res) = covNN( getVCenter( d ), Jb.Ft) + getVQuadratic( d ) * Jb.Fa;
        return res;
    }


    InDeriv multTranspose( const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */
        InDeriv res;
        for (unsigned int i = 0; i < 3; ++i)
        {
            getVCenter(res)[i] = Jb.Pt * Out::center(d)[i];
            getVQuadratic(res)[i] = Jb.Pa * Out::center(d)[i] ;

            for (unsigned int j = 0; j < 3; ++j) getVCenter(res)[i] += Jb.Ft[j] * (Out::materialFrame(d)[i][j]);
            getVQuadratic(res)[i] += Jb.Fa * (Out::materialFrame(d)[i]);
        }
        return res;
    }
};




//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic->DeformationGradient second order
//////////////////////////////////////////////////////////////////////////////////

template<class _Material>
struct LinearBlendTypes<
        StdQuadraticTypes<3,typename _Material::Real>,
        DeformationGradientTypes<3, 3, 2, typename  _Material::Real>,
        _Material
        >
{
    typedef _Material Material;
    typedef typename Material::Real Real;
    typedef typename Material::VecReal VecReal;
    typedef typename Material::Gradient MaterialDeriv;
    typedef typename Material::VecGradient VecMaterialDeriv;
    typedef typename Material::Hessian MaterialMat;
    typedef typename Material::VecHessian VecMaterialMat;
    typedef StdQuadraticTypes<3,Real> In;
    typedef DeformationGradientTypes<3, 3, 2, Real> Out;
    typedef typename Out::SpatialCoord SpatialCoord; // = Vec3
    typedef typename Out::MaterialFrame MaterialFrame;
    typedef typename Out::MaterialFrameGradient MaterialFrameGradient;
    typedef typename In::Coord InCoord;
    typedef typename In::Vec9 SpatialCoord2; // vec9
    typedef Mat<9,3,Real> MaterialFrame2;
    typedef Vec<3,MaterialFrame2> MaterialFrameGradient2;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename Out::VecCoord VecOutCoord;
    typedef vector<unsigned> VecIndex;

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        JacobianBlock() {}
        JacobianBlock(const SpatialCoord2& p2, const Real& w,  const MaterialFrame2& f2,  const MaterialDeriv& dw,  const MaterialFrameGradient2& df2,  const MaterialMat& ddw):Pa(p2),Pt(w),Fa(f2),Ft(dw),dFa(df2),dFt(ddw) {}
        SpatialCoord2 Pa;	///< = dp = dMa_i (w_i \bar M_i p_0)  : affine part
        Real Pt;			///< = dp = dMt_i (w_i)  : translation part
        MaterialFrame2 Fa;  ///< = dF = dMa_i (w_i \bar M_i + \bar M_i p_0 dw_i)
        MaterialDeriv Ft;	///< = dF = dMt_i (dw_i)
        MaterialFrameGradient2 dFa;  ///< = d gradF_k = dMa_i ( grad(w_i)_k \bar M_i + \bar M_i p_0 grad(dw_i)_k + grad(\bar M_i p_0)_k dw_i)
        MaterialMat dFt;	///< = d gradF_k = dMt_i (grad(dw_i)_k)
    };

    InCoord inverseInitialTransform;
    JacobianBlock Jb;

    void init( const InCoord& InitialTransform, const OutCoord& InitialPos, const Real& w, const MaterialDeriv& dw, const MaterialMat&  ddw)
    {
        inverseInitialTransform = In::inverse(InitialTransform);
        const SpatialCoord2& vectorInLocalCoordinates = In::convertToQuadraticCoord( (inverseInitialTransform.getAffine()*Out::center(InitialPos) + inverseInitialTransform.getCenter()) );
        Jb.Pa=vectorInLocalCoordinates*w;
        Jb.Pt=w;
        Jb.Fa=covMN(vectorInLocalCoordinates, dw);
        // use only the inverse of the linear part (affine+translation), squared and crossterms part undefined
        for(unsigned int i=0; i<3; ++i) for(unsigned int j=0; j<3; ++j)  Jb.Fa[i][j]+=inverseInitialTransform.getAffine()[i][j] * w;
        Jb.Ft=dw;

        Jb.dFt=ddw.transposed();
        const MaterialFrame& inverseInitialTransformT=inverseInitialTransform.getAffine().transposed();
        for (unsigned int k = 0; k < 3; ++k)
        {
            Jb.dFa[k] = covMN( vectorInLocalCoordinates, Jb.dFt[k]);
            const MaterialFrame& m=covNN(inverseInitialTransformT[k],dw); // dFa
            for(unsigned int i=0; i<3; ++i)
                for(unsigned int j=0; j<3; ++j)
                {
                    Jb.dFa[k][i][j]+=m[i][j]+inverseInitialTransform.getAffine()[i][j] * dw[k];
                }
        }
    }

    void updateJacobian( const InCoord& /*currentTransform*/)
    {
        return ;
    }

    OutCoord mult( const InCoord& d ) // Called in Apply
    {
        OutDeriv res;
        Out::center(res) = d.getCenter( ) * Jb.Pt + d.getQuadratic ( ) * Jb.Pa;
        Out::materialFrame(res) = covNN( d.getCenter( ), Jb.Ft) + d.getQuadratic( ) * Jb.Fa;
        for (unsigned int k = 0; k < 3; ++k) Out::materialFrameGradient(res)[k] = covNN( d.getCenter( ), Jb.dFt[k]) + d.getQuadratic( ) * Jb.dFa[k];
        return res;
    }

    OutDeriv mult( const InDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        Out::center(res) = getVCenter( d ) * Jb.Pt + getVQuadratic ( d ) * Jb.Pa;
        Out::materialFrame(res) = covNN( getVCenter( d ), Jb.Ft) + getVQuadratic( d ) * Jb.Fa;
        for (unsigned int k = 0; k < 3; ++k) Out::materialFrameGradient(res)[k] = covNN( getVCenter( d ), Jb.dFt[k]) + getVQuadratic( d ) * Jb.dFa[k];
        return res;
    }


    InDeriv multTranspose( const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */
        InDeriv res;
        for (unsigned int i = 0; i < 3; ++i)
        {
            getVCenter(res)[i] = Jb.Pt * Out::center(d)[i];
            getVQuadratic(res)[i] = Jb.Pa * Out::center(d)[i] ;

            for (unsigned int j = 0; j < 3; ++j) getVCenter(res)[i] += Jb.Ft[j] * Out::materialFrame(d)[i][j];
            getVQuadratic(res)[i] += Jb.Fa * (Out::materialFrame(d)[i]);

            for (unsigned int k = 0; k < 3; ++k)
            {
                for (unsigned int j = 0; j < 3; ++j) getVCenter(res)[i] += Jb.dFt[k][j] * Out::materialFrameGradient(d)[k][i][j];
                getVQuadratic(res)[i] += Jb.dFa[k] * (Out::materialFrameGradient(d)[k][i]);
            }
        }
        return res;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Rigid types
//////////////////////////////////////////////////////////////////////////////////

template<class _Material>
struct LinearBlendTypes<
        StdRigidTypes<3,typename _Material::Real>,
        StdVectorTypes< Vec<3,typename _Material::Real>, Vec<3,typename _Material::Real>, typename _Material::Real >,
        _Material
        >
{
    typedef _Material Material;
    typedef typename Material::Real Real;
    typedef typename Material::VecReal VecReal;
    typedef typename Material::Gradient MaterialDeriv;
    typedef typename Material::VecGradient VecMaterialDeriv;
    typedef typename Material::Hessian MaterialMat;
    typedef typename Material::VecHessian VecMaterialMat;
    typedef StdRigidTypes<3,Real> In;
    typedef StdVectorTypes< Vec<3,Real>, Vec<3,Real>, Real > Out;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename Out::VecCoord VecOutCoord;
    typedef vector<unsigned> VecIndex;

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        JacobianBlock() {}
        JacobianBlock(const OutCoord& o, const Real w):Pa(o),Pt(w) {}
        OutCoord Pa;		///< = dp = dMa_i (w_i \bar M_i p_0)  : affine part
        Real Pt;			///< = dp = dMt_i (w_i)  : translation part
    };

    InCoord inverseInitialTransform;
    JacobianBlock Jb;

    void init( const InCoord& InitialTransform, const OutCoord& InitialPos, const Real& w, const MaterialDeriv& /*dw*/, const MaterialMat&  /*ddw*/)
    {
        inverseInitialTransform = In::inverse(InitialTransform);
        //TO DO
        //Jb.Pa=;
        Jb.Pt=w;
    }

    void updateJacobian( const InCoord& currentTransform)
    {
        //TO DO
        //Jb.Pa=currentTransform.getOrientation().rotate(  inverseInitialTransform.pointToParent(InitialPos) *w);
    }


    /// Transform a Vec
    OutCoord mult( const InCoord& f ) // Called in Apply
    {
        //TO DO
        return OutCoord(); //f.pointToParent(v);
    }

    OutDeriv mult( const InDeriv& d )
    {
        //TO DO
        return getLinear( d ) * Jb.Pt + cross(getAngular(d), Jb.Pa);
    }

    InDeriv multTranspose( const OutDeriv& d )
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */
        return InDeriv (
                d[0], d[1], d[2],
                Jb.Pa[1]*d[2]-Jb.Pa[2]*d[1],
                Jb.Pa[2]*d[0]-Jb.Pa[0]*d[2],
                Jb.Pa[0]*d[1]-Jb.Pa[1]*d[0]
                );
    }

};

}


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
//.add< FrameBlendingMapping< Affine3dTypes, ExtVec3fTypes > >()
        .add< FrameBlendingMapping< Affine3dTypes, DeformationGradient331dTypes > >()
        .add< FrameBlendingMapping< Affine3dTypes, DeformationGradient332dTypes > >()
        .add< FrameBlendingMapping< Quadratic3dTypes, Vec3dTypes > >()
//.add< FrameBlendingMapping< Quadratic3dTypes, ExtVec3fTypes > >()
        .add< FrameBlendingMapping< Quadratic3dTypes, DeformationGradient331dTypes > >()
        .add< FrameBlendingMapping< Quadratic3dTypes, DeformationGradient332dTypes > >()
//.add< FrameBlendingMapping< Rigid3dTypes, Vec3dTypes > >()
//.add< FrameBlendingMapping< Rigid3dTypes, ExtVec3fTypes > >()
//.add< FrameBlendingMapping< Rigid3dTypes, DeformationGradient331dTypes > >()
//.add< FrameBlendingMapping< Rigid3dTypes, DeformationGradient332dTypes > >()
#endif
#ifndef SOFA_DOUBLE
// .add< FrameBlendingMapping< Affine3fTypes, Vec3fTypes > >()
// .add< FrameBlendingMapping< Affine3fTypes, ExtVec3fTypes > >()
//.add< FrameBlendingMapping< Affine3dTypes, DeformationGradient332fTypes > >()
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
// .add< FrameBlendingMapping< Affine3dTypes, Vec3fTypes > >()
// .add< FrameBlendingMapping< Affine3fTypes, Vec3dTypes > >()
//.add< FrameBlendingMapping< Affine3dTypes, DeformationGradient332fTypes > >()
//.add< FrameBlendingMapping< Affine3fTypes, DeformationGradient332dTypes > >()
#endif
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, Vec3dTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, ExtVec3fTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, DeformationGradient331dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, DeformationGradient332dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, Vec3dTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, ExtVec3fTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, DeformationGradient331dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, DeformationGradient332dTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, Vec3dTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, ExtVec3fTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, DeformationGradient331dTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, DeformationGradient332dTypes >;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
//template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, Vec3fTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, ExtVec3fTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, DeformationGradient332fTypes >;
#endif //SOFA_DOUBLE
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
//template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, Vec3fTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, Vec3dTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, DeformationGradient332fTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, DeformationGradient332dTypes >;
#endif //SOFA_DOUBLE
#endif //SOFA_FLOAT


} // namespace mapping

} // namespace component

} // namespace sofa

