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
#ifndef SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_H
#define SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_H

#include <sofa/core/Mapping.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>
#include <sofa/component/component.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <vector>
#include <memory>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
class IdentityMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(IdentityMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef typename In::Real			Real;
    typedef typename In::VecCoord		InVecCoord;
    typedef typename In::VecDeriv		InVecDeriv;
    typedef typename In::Coord			InCoord;
    typedef typename In::Deriv			InDeriv;
    typedef typename In::MatrixDeriv	InMatrixDeriv;

    typedef typename Out::VecCoord		VecCoord;
    typedef typename Out::VecDeriv		VecDeriv;
    typedef typename Out::Coord			Coord;
    typedef typename Out::Deriv			Deriv;
    typedef typename Out::MatrixDeriv	MatrixDeriv;

    typedef Out OutDataTypes;
    typedef typename OutDataTypes::Real     OutReal;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::VecDeriv OutVecDeriv;

    enum
    {
        N = OutDataTypes::spatial_dimensions
    };
    enum
    {
        NIn = sofa::defaulttype::DataTypeInfo<InDeriv>::Size
    };
    enum
    {
        NOut = sofa::defaulttype::DataTypeInfo<Deriv>::Size
    };

    typedef defaulttype::Mat<N, N, Real> Mat;
    typedef defaulttype::Mat<NOut, NIn, Real> MBloc;
    typedef sofa::component::linearsolver::CompressedRowSparseMatrix<MBloc> MatrixType;

    helper::ParticleMask* maskFrom;
    helper::ParticleMask* maskTo;
    //enum { N=((int)Deriv::static_size < (int)InDeriv::static_size ? (int)Deriv::static_size : (int)InDeriv::static_size) };

    core::behavior::BaseMechanicalState *stateFrom;
    core::behavior::BaseMechanicalState *stateTo;
protected:
    IdentityMapping()
        : Inherit(),
          maskFrom(NULL),
          maskTo(NULL),
          matrixJ(),
          updateJ(false),
          stateFrom(NULL),
          stateTo(NULL)
    {
    }

    virtual ~IdentityMapping()
    {
    }
public:
    /// Return true if the destination model has the same topology as the source model.
    ///
    /// This is the case for mapping keeping a one-to-one correspondance between
    /// input and output DOFs (mostly identity or data-conversion mappings).
    virtual bool sameTopology() const { return true; }

    void init();

    void apply(const core::MechanicalParams *mparams /* PARAMS FIRST */, Data<VecCoord>& out, const Data<InVecCoord>& in);

    void applyJ(const core::MechanicalParams *mparams /* PARAMS FIRST */, Data<VecDeriv>& out, const Data<InVecDeriv>& in);

    void applyJT(const core::MechanicalParams *mparams /* PARAMS FIRST */, Data<InVecDeriv>& out, const Data<VecDeriv>& in);

    void applyJT(const core::ConstraintParams *cparams /* PARAMS FIRST */, Data<InMatrixDeriv>& out, const Data<MatrixDeriv>& in);

    const sofa::defaulttype::BaseMatrix* getJ();

    virtual void handleTopologyChange();

protected:
    std::auto_ptr<MatrixType> matrixJ;
    bool updateJ;
};

template <int N, int M, class Real>
struct IdentityMappingMatrixHelper;

using sofa::defaulttype::Vec1dTypes;
using sofa::defaulttype::Vec2dTypes;
using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec6dTypes;
using sofa::defaulttype::Vec1fTypes;
using sofa::defaulttype::Vec2fTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::Vec6fTypes;
using sofa::defaulttype::ExtVec1fTypes;
using sofa::defaulttype::ExtVec2fTypes;
using sofa::defaulttype::ExtVec3fTypes;
using sofa::defaulttype::Rigid2dTypes;
using sofa::defaulttype::Rigid3dTypes;
using sofa::defaulttype::Rigid2fTypes;
using sofa::defaulttype::Rigid3fTypes;

#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_CPP)
#pragma warning(disable : 4231)

#ifndef SOFA_FLOAT
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec2dTypes, Vec2dTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec1dTypes, Vec1dTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec6dTypes, Vec6dTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec3dTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3dTypes, Rigid3dTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid2dTypes, Rigid2dTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3dTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid2dTypes, Vec2dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec3fTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec2fTypes, Vec2fTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec1fTypes, Vec1fTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec6fTypes, Vec6fTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec3fTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3fTypes, Rigid3fTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid2fTypes, Rigid2fTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3fTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3fTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid2fTypes, Vec2fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec3fTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec2dTypes, Vec2fTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec2fTypes, Vec2dTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec1dTypes, Vec1fTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec1fTypes, Vec1dTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec6dTypes, Vec6fTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec6fTypes, Vec6dTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3fTypes, Rigid3dTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3dTypes, Rigid3fTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid2dTypes, Rigid2fTypes >;
extern template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid2fTypes, Rigid2dTypes >;
#endif
#endif

#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
