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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_H
#define SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_H

#include <sofa/core/Mapping.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <vector>
#include <sofa/helper/SVector.h>
#include <sofa/component/component.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Mat.h>

#ifdef SOFA_DEV
#include <sofa/helper/DualQuat.h>
#endif

namespace sofa
{

namespace component
{

namespace mapping
{

using sofa::helper::vector;
using sofa::helper::SVector;

template <class TIn, class TOut>
class SkinningMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(SkinningMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;

    // Input types
    typedef TIn In;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Real InReal;

    // Output types
    typedef TOut Out;
    typedef typename Out::VecCoord VecOutCoord;
    typedef typename Out::VecDeriv VecOutDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real OutReal;

#ifdef SOFA_DEV
    typedef helper::DualQuatCoord3<OutReal> DQCoord;
    typedef defaulttype::Mat<4,4,OutReal> Mat44;
    typedef defaulttype::Mat<3,3,OutReal> Mat33;
    typedef defaulttype::Mat<3,4,OutReal> Mat34;
    typedef defaulttype::Mat<4,3,OutReal> Mat43;
#endif

protected:

    helper::ParticleMask* maskFrom;
    helper::ParticleMask* maskTo;

    Data<VecOutCoord> f_initPos;  // initial child coordinates in the world reference frame

    // data for linear blending
    vector<vector<OutCoord> > f_localPos; /// initial child coordinates in local frame x weight :   dp = dMa_i (w_i \bar M_i f_localPos)
    vector<vector<OutCoord> > f_rotatedPos;  /// rotated child coordinates :  dp = Omega_i x f_rotatedPos  :

    // data for dual quat blending
#ifdef SOFA_DEV
    vector<vector< Mat44 > > f_T0; /// Real part of blended quaternion Jacobian : db = [T0,TE] dq
    vector<vector< Mat44 > > f_TE; /// Dual part of blended quaternion Jacobian : db = [T0,TE] dq
    vector<vector< Mat33 > > f_Pa; /// dp = Pa.Omega_i  : affine part
    vector<vector< Mat33 > > f_Pt; /// dp = Pt.dt_i : translation part
    Data<bool> useDQ;  // use dual quat blending instead of linear blending
#endif

    Data<unsigned int> nbRef; // Number of primitives influencing each point.
    Data< vector<SVector<unsigned int> > > f_index; // indices of primitives influencing each point.
    Data< vector<SVector<InReal> > > weight;
    void updateWeights();

public:
    Data<unsigned int> showFromIndex;
    Data<bool> showWeights;
protected:
    SkinningMapping ();
    virtual ~SkinningMapping();
public:
    void init();
    void reinit();

    void apply(typename Out::VecCoord& out, const typename In::VecCoord& in);
    void applyJ(typename Out::VecDeriv& out, const typename In::VecDeriv& in);
    void applyJT(typename In::VecDeriv& out, const typename Out::VecDeriv& in);
    void applyJT(typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in);

    SeqTriangles triangles; // Topology of toModel (used for weight display)
    void draw(const core::visual::VisualParams* vparams);

};

using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::ExtVec3fTypes;
using sofa::defaulttype::Rigid3dTypes;
using sofa::defaulttype::Rigid3fTypes;

#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_RIGID_API SkinningMapping< Rigid3dTypes, Vec3dTypes >;
extern template class SOFA_RIGID_API SkinningMapping< Rigid3dTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_RIGID_API SkinningMapping< Rigid3fTypes, Vec3fTypes >;
extern template class SOFA_RIGID_API SkinningMapping< Rigid3fTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_RIGID_API SkinningMapping< Rigid3dTypes, Vec3fTypes >;
extern template class SOFA_RIGID_API SkinningMapping< Rigid3fTypes, Vec3dTypes >;
#endif
#endif
#endif //defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_CPP)



} // namespace mapping

} // namespace component

} // namespace sofa

#endif
