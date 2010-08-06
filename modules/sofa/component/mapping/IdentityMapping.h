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
#ifndef SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_H
#define SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_H

#include <sofa/core/behavior/MechanicalMapping.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/MappedModel.h>
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

template <class BasicMapping>
class IdentityMapping : public BasicMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(IdentityMapping,BasicMapping), BasicMapping);

    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;

    typedef typename In::DataTypes InDataTypes;
    typedef typename In::Real      Real;
    typedef typename In::VecCoord  InVecCoord;
    typedef typename In::VecDeriv  InVecDeriv;
    typedef typename In::Coord     InCoord;
    typedef typename In::Deriv     InDeriv;

    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord    Coord;
    typedef typename Out::Deriv    Deriv;

    typedef typename Out::DataTypes        OutDataTypes;
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

    IdentityMapping(In* from, Out* to)
        : Inherit(from, to),
          matrixJ(),
          updateJ(false)
    {
        maskFrom = NULL;
        if ((stateFrom = dynamic_cast< core::behavior::BaseMechanicalState *>(from)))
            maskFrom = &stateFrom->forceMask;
        maskTo = NULL;
        if ((stateTo = dynamic_cast< core::behavior::BaseMechanicalState *>(to)))
            maskTo = &stateTo->forceMask;
    }

    virtual ~IdentityMapping()
    {
    }

    /// Return true if the destination model has the same topology as the source model.
    ///
    /// This is the case for mapping keeping a one-to-one correspondance between
    /// input and output DOFs (mostly identity or data-conversion mappings).
    virtual bool sameTopology() const { return true; }

    void apply(VecCoord& out, const InVecCoord& in);

    void applyJ(VecDeriv& out, const InVecDeriv& in);

    void applyJT(InVecDeriv& out, const VecDeriv& in);

    void applyJT(typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in);

    const sofa::defaulttype::BaseMatrix* getJ();

    virtual void handleTopologyChange();

protected:
    std::auto_ptr<MatrixType> matrixJ;
    bool updateJ;
};

template <int N, int M, class Real>
struct IdentityMappingMatrixHelper;

using core::Mapping;
using core::behavior::MechanicalMapping;
using core::behavior::MappedModel;
using core::behavior::State;
using core::behavior::MechanicalState;

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
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec2dTypes>, MechanicalState<Vec2dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Vec1dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec6dTypes>, MechanicalState<Vec6dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3dTypes> > >;
// extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Rigid3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3dTypes>, MappedModel<Rigid3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid2dTypes>, MechanicalState<Rigid2dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid2dTypes>, MappedModel<Rigid2dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > >;
// extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid2dTypes>, MechanicalState<Vec2dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid2dTypes>, MappedModel<Vec2dTypes> > >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec2fTypes>, MechanicalState<Vec2fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Vec1fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec6fTypes>, MechanicalState<Vec6fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3fTypes> > >;
// extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Rigid3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3fTypes>, MappedModel<Rigid3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid2fTypes>, MechanicalState<Rigid2fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid2fTypes>, MappedModel<Rigid2fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid2fTypes>, MechanicalState<Vec2fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid2fTypes>, MappedModel<Vec2fTypes> > >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec2dTypes>, MechanicalState<Vec2fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec2fTypes>, MechanicalState<Vec2dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Vec1fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Vec1dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec6dTypes>, MechanicalState<Vec6fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec6fTypes>, MechanicalState<Vec6dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Rigid3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Rigid3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3fTypes>, MappedModel<Rigid3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3dTypes>, MappedModel<Rigid3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid2dTypes>, MappedModel<Rigid2fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid2fTypes>, MappedModel<Rigid2dTypes> > >;
#endif
#endif

#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
