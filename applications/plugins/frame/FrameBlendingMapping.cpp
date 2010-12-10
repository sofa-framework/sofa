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

    void init( const VecInCoord&, const VecOutCoord&, const VecIndex&, const VecReal&, const VecMaterialDeriv&, const VecMaterialMat&  ) {}

    /// Transform a Vec
    OutCoord mult( const InCoord& f, const OutCoord& v )
    {
        return f.getCenter() + f.getAffine()*v;
    }

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        JacobianBlock() {}
        JacobianBlock(const OutCoord& o, const Real w):weightedVectorInLocalCoordinates(o),w(w) {}
        OutCoord weightedVectorInLocalCoordinates;  ///< = dp_0/ dMa_i = w_i \bar M_i p_0  : affine part
        Real w; ///< = dp_0/ dMt_i = w_i : translation part
    };


    JacobianBlock computeJacobianBlock( const InCoord& /*currentTransform*/, const InCoord& inverseInitialMatrix, const OutCoord& initialChildPosition, Real w, const MaterialDeriv /*dw*/,  const MaterialMat /*ddw*/ )
    {
        return JacobianBlock( mult(inverseInitialMatrix,initialChildPosition)*w, w );
    }

    bool mustRecomputeJacobian(  )
    {
        // No update is needed
        return false;
    }

    OutDeriv mult( const JacobianBlock& block, const InDeriv& d )
    {
        return getVCenter( d ) * block.w + getVAffine( d ) * block.weightedVectorInLocalCoordinates;
    }

    InDeriv multTranspose( const JacobianBlock& block, const OutDeriv& d )
    {
        const OutDeriv& j=block.weightedVectorInLocalCoordinates;
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */
        return InDeriv (
                j[0]*d[0], j[1]*d[0], j[2]*d[0], d[0]*block.w,
                j[0]*d[1], j[1]*d[1], j[2]*d[1], d[1]*block.w,
                j[0]*d[2], j[1]*d[2], j[2]*d[2], d[2]*block.w
                );
    }

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

    void init( const VecInCoord&, const VecOutCoord&, const VecIndex&, const VecReal&, const VecMaterialDeriv&, const VecMaterialMat&  ) {}

    /// Transform a Vec
    SpatialCoord mult( const InCoord& f, const SpatialCoord& v ) // Called in Apply
    {
        return f.getCenter() + f.getAffine()*Out::center(v);
    }

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        JacobianBlock() {}
        JacobianBlock(const OutCoord& o, const Real w, MaterialDeriv dw):weightedVectorInLocalCoordinates(Out::center(o)),weightedVectorInLocalCoordinatesDerivative(Out::materialFrame(o)),w(w),dw(dw) {}
        SpatialCoord weightedVectorInLocalCoordinates;  ///< = dp_0/ dMa_i = w_i \bar M_i p_0  : affine part
        Real w; ///< = dp_0/ dMt_i = w_i : translation part
        MaterialFrame weightedVectorInLocalCoordinatesDerivative;  ///< = d [dp_0/ dMa_i]  / dMa_i = d( w_i \bar M_i p_0 ) / d p_0 = \bar M_i p_0 dw + w_i \bar M_i
        MaterialDeriv dw; ///< =d  [dp_0/ dMa_i] / dMt_i = = dw_i
    };


    JacobianBlock computeJacobianBlock( const InCoord& /*currentTransform*/, const InCoord& inverseInitialMatrix, const OutCoord& initialChildPosition, Real w, const MaterialDeriv dw,  const MaterialMat /*ddw*/ )
    {
        OutCoord jacobian;
        SpatialCoord vectorInLocalCoordinates = mult(inverseInitialMatrix,Out::center(initialChildPosition));
        Out::center(jacobian) = vectorInLocalCoordinates*w;
        Out::materialFrame(jacobian) = inverseInitialMatrix.getAffine() * w + covNN( vectorInLocalCoordinates, dw);
        return JacobianBlock( jacobian, w, dw);
    }


    bool mustRecomputeJacobian(  )
    {
        // No update is needed
        return false;
    }

    OutDeriv mult( const JacobianBlock& block, const InDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        Out::center(res) = getVCenter( d ) * block.w + getVAffine( d ) * block.weightedVectorInLocalCoordinates;
        Out::materialFrame(res) = covNN( getVCenter( d ), block.dw) + getVAffine( d ) * block.weightedVectorInLocalCoordinatesDerivative;
        return res;
    }


    InDeriv multTranspose( const JacobianBlock& block, const OutDeriv& d ) // Called in ApplyJT
    {
        const SpatialCoord& jt=block.weightedVectorInLocalCoordinates;
        const MaterialFrame& ja=block.weightedVectorInLocalCoordinatesDerivative;
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */
        InDeriv res;
        for (unsigned int i = 0; i < 3; ++i)
        {
            getVCenter(res)[i] = block.w * Out::center(d)[i];
            for (unsigned int j = 0; j < 3; ++j)
                getVCenter(res)[i] += block.dw[j] * Out::materialFrame(d)[i][j];
            getVAffine(res)[i] = jt * Out::center(d)[i] + ja * (Out::materialFrame(d)[i]);
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
    typedef Mat<3,3,Real> Mat33;
    typedef typename In::VecCoord VecInCoord;
    typedef typename Out::VecCoord VecOutCoord;
    typedef vector<unsigned> VecIndex;

    void init( const VecInCoord&, const VecOutCoord&, const VecIndex&, const VecReal&, const VecMaterialDeriv&, const VecMaterialMat&  ) {}

    /// Transform a Vec
    SpatialCoord mult( const InCoord& f, const SpatialCoord& v ) // Called in Apply
    {
        return f.getCenter() + f.getAffine()*Out::center(v);
    }

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        JacobianBlock() {}
        JacobianBlock(const OutCoord& o, const Real w, MaterialDeriv dw, MaterialMat ddw):weightedVectorInLocalCoordinates(Out::center(o)),weightedVectorInLocalCoordinatesDerivative(Out::materialFrame(o)),weightedVectorInLocalCoordinatesScdDerivative(Out::materialFrameGradient(o)),w(w),dw(dw),ddw(ddw) {}
        SpatialCoord weightedVectorInLocalCoordinates;  ///< = dp_0/ dMa_i = w_i \bar M_i p_0  : affine part
        Real w; ///< = dp_0/ dMt_i = w_i : translation part
        MaterialFrame weightedVectorInLocalCoordinatesDerivative;  ///< = d weightedVectorInLocalCoordinates  / dMa_i = d( w_i \bar M_i p_0 ) / d p_0 = \bar M_i p_0 dw + w_i \bar M_i
        MaterialDeriv dw; ///< =d  weightedVectorInLocalCoordinates / dMt_i = = dw_i
        MaterialFrameGradient weightedVectorInLocalCoordinatesScdDerivative;  ///< = d weightedVectorInLocalCoordinatesDerivative  / dMa_i = \bar M_i p_0 ddw_i + dw_i \bar M_i + d\bar M_i p_0 dw
        MaterialMat ddw; ///< =d  weightedVectorInLocalCoordinatesDerivative / dMt_i = = ddw_i
    };


    JacobianBlock computeJacobianBlock( const InCoord& /*currentTransform*/, const InCoord& inverseInitialMatrix, const OutCoord& initialChildPosition, Real w, const MaterialDeriv dw,  const MaterialMat ddw )
    {
        OutCoord jacobian;
        SpatialCoord vectorInLocalCoordinates = mult(inverseInitialMatrix,Out::center(initialChildPosition));
        Out::center(jacobian) = vectorInLocalCoordinates*w;
        Out::materialFrame(jacobian) = inverseInitialMatrix.getAffine() * w + covNN( vectorInLocalCoordinates, dw);
        MaterialMat ddwT = ddw.transpose();
        Mat33 inverseInitialMatrixT = inverseInitialMatrix.getAffine().transpose();
        for (unsigned int i = 0; i < 3; ++i)
            Out::materialFrameGradient(jacobian)[i] = covNN( vectorInLocalCoordinates, ddwT[i]) + inverseInitialMatrix.getAffine() * dw[i] + covNN(inverseInitialMatrixT[i],ddw[i]);
        return JacobianBlock( jacobian, w, dw, ddw);
    }

    bool mustRecomputeJacobian(  )
    {
        // No update is needed
        return false;
    }

    /*
    static OutDeriv mult( const JacobianBlock& block, const InDeriv& d ) // Called in ApplyJ
    {
        OutDeriv res;
        Out::center(res) = getVCenter( d ) * block.w + getVAffine( d ) * block.weightedVectorInLocalCoordinates;
        Out::materialFrame(res) = covNN( getVCenter( d ), block.dw) + getVAffine( d ) * block.weightedVectorInLocalCoordinatesDerivative;
        for (unsigned int i = 0; i < 3; ++i)
          Out::materialFrameGradient(res)[i] = covNN( getVCenter( d ), block.dw) + getVAffine( d ) * block.weightedVectorInLocalCoordinatesScdDerivative;
        return res;
    }


    static InDeriv multTranspose( const JacobianBlock& block, const OutDeriv& d ) // Called in ApplyJT
    {
        const SpatialCoord& jt=block.weightedVectorInLocalCoordinates;
        const MaterialFrame& ja=block.weightedVectorInLocalCoordinatesDerivative;
        //To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
       InDeriv res;
        for (unsigned int i = 0; i < 3; ++i)
        {
           getVCenter(res)[i] = block.w * Out::center(d)[i];
            for (unsigned int j = 0; j < 3; ++j)
               getVCenter(res)[i] += block.dw[j] * Out::materialFrame(d)[i][j];
            getVAffine(res)[i] = jt * Out::center(d)[i] + ja * (Out::materialFrame(d)[i]);
        }
        return res;
    }*/
};



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic types
//////////////////////////////////////////////////////////////////////////////////
/*
        template<class _Real, class _MaterialDeriv, class _MaterialMat>
        struct LinearBlendTypes<
                StdQuadraticTypes<3,_Real>,
                StdVectorTypes< Vec<3,_Real>, Vec<3,_Real>, _Real >,
                _MaterialDeriv, _MaterialMat
                >
        {
            typedef _Real Real;
            typedef _MaterialDeriv MaterialDeriv;
            typedef _MaterialMat MaterialMat;
            typedef StdQuadraticTypes<3,Real> In;
            typedef StdVectorTypes< Vec<3,_Real>, Vec<3,_Real>, _Real > Out;
            typedef typename In::Coord InCoord;
            typedef typename Out::Coord OutCoord;
            typedef typename Out::Deriv OutDeriv;
            typedef typename In::Deriv InDeriv;

            /// Transform a Vec
            static OutCoord mult( const InCoord& f, const OutCoord& v )
            {
                return f.pointToParent(v);
            }

            struct JacobianBlock
            {
                // Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
                 // The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.

                JacobianBlock(){}
                JacobianBlock(const OutCoord& o, const Real w):weightedVectorInLocalCoordinates(o),w(w){}
                OutCoord weightedVectorInLocalCoordinates;  ///< = dp_0/ dMa_i = w_i \bar M_i p_0  : affine part
                Real w; ///< = dp_0/ dMt_i = w_i : translation part
            };

            static JacobianBlock computeJacobianBlock( const InCoord& inverseInitialMatrix, const OutCoord& initialChildPosition, Real w, const MaterialDeriv dw,  const MaterialMat ddw )
            {
                return JacobianBlock( mult(inverseInitialMatrix,initialChildPosition)*w, w);
            }

            static OutDeriv mult( const JacobianBlock& block, const InDeriv& d )
            {
                return getLinear( d ) * block.w + cross(getAngular(d), block.weightedVectorInLocalCoordinates);
            }

            static InDeriv multTranspose( const JacobianBlock& block, const OutDeriv& d )
            {
                const OutDeriv& j=block.weightedVectorInLocalCoordinates;
                // To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
                return InDeriv (
                        d[0], d[1], d[2],
                        j[1]*d[2]-j[2]*d[1],
                        j[2]*d[0]-j[0]*d[2],
                        j[0]*d[1]-j[1]*d[0]
                        );
            }

        };

        */

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

    void init( const VecInCoord&, const VecOutCoord&, const VecIndex&, const VecReal&, const VecMaterialDeriv&, const VecMaterialMat&  ) {}

    /// Transform a Vec
    OutCoord mult( const InCoord& f, const OutCoord& v )
    {
        return f.pointToParent(v);
    }

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 )  in homogeneous coordinates.
          */
        JacobianBlock() {}
        JacobianBlock(const OutCoord& o, const Real w):weightedVectorInWorldCoordinates(o),w(w) {}
        OutCoord weightedVectorInWorldCoordinates;  ///< = dp_0/ dMa_i = w_i \bar M_i p_0  : affine part
        Real w; ///< = dp_0/ dMt_i = w_i : translation part
    };

    JacobianBlock computeJacobianBlock( const InCoord& currentTransform, const InCoord& inverseInitialTransform, const OutCoord& initialChildPosition, Real w, const MaterialDeriv /*dw*/,  const MaterialMat /*ddw*/ )
    {
        return JacobianBlock( currentTransform.getOrientation().rotate(mult(inverseInitialTransform,initialChildPosition)*w), w);
    }

    bool mustRecomputeJacobian(  )
    {
        // Update is needed
        return true;
    }

    OutDeriv mult( const JacobianBlock& block, const InDeriv& d )
    {
        return getLinear( d ) * block.w + cross(getAngular(d), block.weightedVectorInWorldCoordinates);
    }

    InDeriv multTranspose( const JacobianBlock& block, const OutDeriv& d )
    {
        const OutDeriv& j=block.weightedVectorInWorldCoordinates;
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */
        return InDeriv (
                d[0], d[1], d[2],
                j[1]*d[2]-j[2]*d[1],
                j[2]*d[0]-j[0]*d[2],
                j[0]*d[1]-j[1]*d[0]
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
//.add< FrameBlendingMapping< Affine3dTypes, DeformationGradient331dTypes > >()
//.add< FrameBlendingMapping< Quadratic3dTypes, Vec3dTypes > >()
        .add< FrameBlendingMapping< Rigid3dTypes, Vec3dTypes > >()
// .add< FrameBlendingMapping< Affine3dTypes, ExtVec3fTypes > >()
//.add< FrameBlendingMapping< Affine3dTypes, DeformationGradient331dTypes > >()
//.add< FrameBlendingMapping< Affine3dTypes, DeformationGradient332dTypes > >()
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
//template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, Vec3dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, Vec3dTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, ExtVec3fTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, DeformationGradient331dTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, DeformationGradient332dTypes >;
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

