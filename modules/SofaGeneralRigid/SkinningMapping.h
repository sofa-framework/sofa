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
#ifndef SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_H
#define SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_H
#include "config.h"

#include <sofa/core/Mapping.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <vector>
#include <sofa/helper/SVector.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Mat.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>

#ifdef SOFA_DEV
#include <sofa/helper/DualQuat.h>
#endif

namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
class SkinningMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(SkinningMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef SReal Real;

    // Input types
    typedef TIn In;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Real InReal;

    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;

    // Output types
    typedef TOut Out;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real OutReal;

    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;

    typedef sofa::defaulttype::Mat<OutDeriv::total_size,InDeriv::total_size,Real>     MatBlock;
    typedef sofa::component::linearsolver::EigenSparseMatrix<In, Out> SparseJMatrixEigen;

    typedef typename Inherit::ForceMask ForceMask;

#ifdef SOFA_DEV
    typedef helper::DualQuatCoord3<OutReal> DQCoord;
    typedef defaulttype::Mat<4,4,OutReal> Mat44;
    typedef defaulttype::Mat<3,3,OutReal> Mat33;
    typedef defaulttype::Mat<3,4,OutReal> Mat34;
    typedef defaulttype::Mat<4,3,OutReal> Mat43;
#endif

protected:


    Data<OutVecCoord> f_initPos;  // initial child coordinates in the world reference frame

    // data for linear blending
    helper:: vector<helper::vector<OutCoord> > f_localPos; /// initial child coordinates in local frame x weight :   dp = dMa_i (w_i \bar M_i f_localPos)
    helper::vector<helper::vector<OutCoord> > f_rotatedPos;  /// rotated child coordinates :  dp = Omega_i x f_rotatedPos  :
    SparseJMatrixEigen   _J; /// jacobian matrix for compliant API

    // data for dual quat blending
#ifdef SOFA_DEV
    helper::vector<helper::vector< Mat44 > > f_T0; /// Real part of blended quaternion Jacobian : db = [T0,TE] dq
    helper::vector<helper::vector< Mat44 > > f_TE; /// Dual part of blended quaternion Jacobian : db = [T0,TE] dq
    helper::vector<helper::vector< Mat33 > > f_Pa; /// dp = Pa.Omega_i  : affine part
    helper::vector<helper::vector< Mat33 > > f_Pt; /// dp = Pt.dt_i : translation part
    Data<bool> useDQ;  // use dual quat blending instead of linear blending
#endif

    Data< helper::vector<unsigned int> > nbRef; // Number of primitives influencing each point.
    Data< helper::vector<sofa::helper::SVector<unsigned int> > > f_index; // indices of primitives influencing each point.
    Data< helper::vector<sofa::helper::SVector<InReal> > > weight;
    void updateWeights();

public:
    void setWeights(const helper::vector<sofa::helper::SVector<InReal> >& weights, const helper::vector<sofa::helper::SVector<unsigned int> >& indices, const helper::vector<unsigned int>& nbrefs);

public:
    Data<unsigned int> showFromIndex;
    Data<bool> showWeights;
protected:
    SkinningMapping ();
    virtual ~SkinningMapping();
    
public:
    void init();
    void reinit();

    virtual void apply( const sofa::core::MechanicalParams* mparams, OutDataVecCoord& out, const InDataVecCoord& in);
    //void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    virtual void applyJ( const sofa::core::MechanicalParams* mparams, OutDataVecDeriv& out, const InDataVecDeriv& in);
    //void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    virtual void applyJT( const sofa::core::MechanicalParams* mparams, InDataVecDeriv& out, const OutDataVecDeriv& in);
    //void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    virtual void applyJT( const sofa::core::ConstraintParams* cparams, InDataMatrixDeriv& out, const OutDataMatrixDeriv& in);
    //void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in );

    // additional Compliant methods
    const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs();
    const sofa::defaulttype::BaseMatrix* getJ();

    SeqTriangles triangles; // Topology of toModel (used for weight display)
    void draw(const core::visual::VisualParams* vparams);

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_RIGID_API SkinningMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_GENERAL_RIGID_API SkinningMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::ExtVec3fTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_RIGID_API SkinningMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_GENERAL_RIGID_API SkinningMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::ExtVec3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_RIGID_API SkinningMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_GENERAL_RIGID_API SkinningMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Vec3dTypes >;
#endif
#endif
#endif //defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_CPP)



} // namespace mapping

} // namespace component

} // namespace sofa

#endif
