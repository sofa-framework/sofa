/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/mapping/linear/config.h>

#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <sofa/core/Mapping.h>
#include <sofa/core/Mapping.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/vector.h>


namespace sofa::component::mapping::linear
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

    typedef type::Mat<N, N, Real> Mat;

protected:
    IdentityMapping()
        : Inherit()
    {
        Js.resize( 1 );
        Js[0] = &J;
    }

    virtual ~IdentityMapping()
    {
    }

public:
    /// Return true if the destination model has the same topology as the source model.
    ///
    /// This is the case for mapping keeping a one-to-one correspondance between
    /// input and output DOFs (mostly identity or data-conversion mappings).
    bool sameTopology() const override { return true; }

    void init() override;

    void apply(const core::MechanicalParams *mparams, Data<VecCoord>& out, const Data<InVecCoord>& in) override;

    void applyJ(const core::MechanicalParams *mparams, Data<VecDeriv>& out, const Data<InVecDeriv>& in) override;

    void applyJT(const core::MechanicalParams *mparams, Data<InVecDeriv>& out, const Data<VecDeriv>& in) override;

    void applyJT(const core::ConstraintParams *cparams, Data<InMatrixDeriv>& out, const Data<MatrixDeriv>& in) override;

    const sofa::linearalgebra::BaseMatrix* getJ() override;

    void handleTopologyChange() override;


protected:

    typedef linearalgebra::EigenSparseMatrix<TIn, TOut> eigen_type;
    eigen_type J;

    typedef type::vector< linearalgebra::BaseMatrix* > js_type;
    js_type Js;

public:

    const js_type* getJs() override;

};

#if !defined(SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_CPP)

extern template class SOFA_COMPONENT_MAPPING_LINEAR_API IdentityMapping< defaulttype::Vec3Types, defaulttype::Vec3Types >;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API IdentityMapping< defaulttype::Vec2Types, defaulttype::Vec2Types >;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API IdentityMapping< defaulttype::Vec1Types, defaulttype::Vec1Types >;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API IdentityMapping< defaulttype::Vec6Types, defaulttype::Vec6Types >;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API IdentityMapping< defaulttype::Vec6Types, defaulttype::Vec3Types >;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API IdentityMapping< defaulttype::Rigid3Types, defaulttype::Rigid3Types >;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API IdentityMapping< defaulttype::Rigid2Types, defaulttype::Rigid2Types >;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API IdentityMapping< defaulttype::Rigid3Types, defaulttype::Vec3Types >;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API IdentityMapping< defaulttype::Rigid2Types, defaulttype::Vec2Types >;

#endif

} // namespace sofa::component::mapping::linear
