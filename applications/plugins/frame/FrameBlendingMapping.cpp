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
#define SOFA_COMPONENT_MAPPING_FrameBlendingMAPPING_CPP

#include "AffineTypes.h"
#include <sofa/defaulttype/RigidTypes.h>
#include "FrameBlendingMapping.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace defaulttype
{

//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine types
//////////////////////////////////////////////////////////////////////////////////

template<class _Real, class _MaterialDeriv, class _MaterialMat>
struct LinearBlendTypes<
        StdAffineTypes<3,_Real>,
        StdVectorTypes< Vec<3,_Real>, Vec<3,_Real>, _Real >,
        _MaterialDeriv, _MaterialMat
        >
{
    typedef _Real Real;
    typedef _MaterialDeriv MaterialDeriv;
    typedef _MaterialMat MaterialMat;
    typedef StdAffineTypes<3,Real> In;
    typedef StdVectorTypes< Vec<3,_Real>, Vec<3,_Real>, _Real > Out;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::Deriv InDeriv;

    /// Transform a Vec
    static OutCoord mult( const InCoord& f, const OutCoord& v )
    {
        return f.getCenter() + f.getAffine()*v;
    }

    struct JacobianBlock
    {
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 ).
          */
        JacobianBlock() {}
        JacobianBlock(const OutDeriv& o ):weightedVectorInLocalCoordinates(o) {}
        OutDeriv weightedVectorInLocalCoordinates;  ///< = w_i \bar M_i p_0
    };


    static JacobianBlock computeJacobianBlock( const InCoord& inverseInitialMatrix, const OutCoord& initialChildPosition, Real w, const MaterialDeriv /*dw*/,  const MaterialMat /*ddw*/ )
    {
        return JacobianBlock( mult(inverseInitialMatrix,initialChildPosition)*w );
    }

    static OutDeriv mult( const JacobianBlock& block, const InDeriv& d )
    {
        return getVCenter( d ) + getVAffine( d ) * block.weightedVectorInLocalCoordinates;
    }

    static InDeriv multTranspose( const JacobianBlock& block, const OutDeriv& d )
    {
        const OutDeriv& j=block.weightedVectorInLocalCoordinates;
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */
        return InDeriv (
                j[0]*d[0], j[1]*d[0], j[2]*d[0], d[0],
                j[0]*d[1], j[1]*d[1], j[2]*d[1], d[1],
                j[0]*d[2], j[1]*d[2], j[2]*d[2], d[2]
                );
    }

};



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Rigid types
//////////////////////////////////////////////////////////////////////////////////

template<class _Real, class _MaterialDeriv, class _MaterialMat>
struct LinearBlendTypes<
        StdRigidTypes<3,_Real>,
        StdVectorTypes< Vec<3,_Real>, Vec<3,_Real>, _Real >,
        _MaterialDeriv, _MaterialMat
        >
{
    typedef _Real Real;
    typedef _MaterialDeriv MaterialDeriv;
    typedef _MaterialMat MaterialMat;
    typedef StdRigidTypes<3,Real> In;
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
        /** Linear blend skinning: p = \sum_i w_i M_i \bar M_i p_0  where \bar M_i is the inverse of M_i in the reference configuration, and p_0 is the position of p in the reference configuration.
          The variation of p when a change dM_i is applied is thus w_i dM_i \bar M_i p_0, which we can compute as: dM_i * ( w_i \bar M_i p_0 ).
          */
        JacobianBlock() {}
        JacobianBlock(const OutDeriv& o ):weightedVectorInLocalCoordinates(o) {}
        OutDeriv weightedVectorInLocalCoordinates;  ///< = w_i \bar M_i p_0
    };


    static JacobianBlock computeJacobianBlock( const InCoord& inverseInitialMatrix, const OutCoord& initialChildPosition, Real w, const MaterialDeriv /*dw*/,  const MaterialMat /*ddw*/ )
    {
        return JacobianBlock( mult(inverseInitialMatrix,initialChildPosition)*w );
    }

    static OutDeriv mult( const JacobianBlock& block, const InDeriv& d )
    {
        return getLinear( d ) + cross(getAngular(d), block.weightedVectorInLocalCoordinates);
    }

    static InDeriv multTranspose( const JacobianBlock& block, const OutDeriv& d )
    {
        const OutDeriv& j=block.weightedVectorInLocalCoordinates;
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
        .add< FrameBlendingMapping< Rigid3dTypes, Vec3dTypes > >()
// .add< FrameBlendingMapping< Affine3dTypes, ExtVec3fTypes > >()
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
template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, Vec3dTypes >;
//template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, ExtVec3fTypes >;
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

