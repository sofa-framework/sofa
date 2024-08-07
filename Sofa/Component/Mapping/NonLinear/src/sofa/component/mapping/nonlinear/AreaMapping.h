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

#include <sofa/component/mapping/nonlinear/config.h>
#include <sofa/core/Mapping.h>
#include <sofa/component/mapping/nonlinear/NonLinearMappingData.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>


namespace sofa::component::mapping::nonlinear
{

template <class TIn, class TOut>
class AreaMapping : public core::Mapping<TIn, TOut>, public StabilizedNonLinearMappingData
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(AreaMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    using In = TIn;
    using Out = TOut;

    using Real = Real_t<Out>;

    typedef linearalgebra::EigenSparseMatrix<TIn,TOut> SparseMatrixEigen;
    static constexpr auto Nin = In::deriv_total_size;

    SingleLink<AreaMapping<TIn, TOut>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    static sofa::type::Mat<3,3,sofa::type::Mat<3,3,Real>> computeSecondDerivativeArea(const sofa::type::fixed_array<sofa::type::Vec3, 3>& triangleVertices);

    void init() override;

    void apply(const core::MechanicalParams* mparams, DataVecCoord_t<Out>& out, const DataVecCoord_t<In>& in) override;
    void applyJ(const core::MechanicalParams* mparams, DataVecDeriv_t<Out>& out, const DataVecDeriv_t<In>& in) override;
    void applyJT(const core::MechanicalParams* mparams, DataVecDeriv_t<In>& out, const DataVecDeriv_t<Out>& in) override;
    void applyJT(const core::ConstraintParams *cparams, DataMatrixDeriv_t<In>& out, const DataMatrixDeriv_t<Out>& in) override;
    void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentForceId, core::ConstMultiVecDerivId childForceId) override;

    void updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForceId) override;
    const linearalgebra::BaseMatrix* getK() override;
    void buildGeometricStiffnessMatrix(sofa::core::GeometricStiffnessMatrix* matrices) override;

    const type::vector<sofa::linearalgebra::BaseMatrix*>* getJs() override;

protected:
    AreaMapping();

    using SparseKMatrixEigen = linearalgebra::EigenSparseMatrix<TIn,TIn>;

    SparseMatrixEigen jacobian; ///< Jacobian of the mapping
    type::vector<linearalgebra::BaseMatrix*> baseMatrices; ///< Jacobian of the mapping, in a vector
    typename AreaMapping::SparseKMatrixEigen K; ///< Assembled geometric stiffness matrix

    /**
     * @brief Represents an entry in the Jacobian matrix.
     *
     * The JacobianEntry struct is used to store information about an entry in the
     * Jacobian matrix, specifically the vertex identifier and the corresponding
     * Jacobian value. It also provides a comparison operator for sorting entries
     * by vertex ID.
     */
    struct JacobianEntry
    {
        sofa::Index vertexId;
        typename In::Coord jacobianValue;
        bool operator<(const JacobianEntry& other) const { return vertexId < other.vertexId;}
    };

    const VecCoord_t<In>* m_vertices{nullptr};


};

#if !defined(SOFA_COMPONENT_MAPPING_NONLINEAR_AREAMAPPING_CPP)
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API AreaMapping< defaulttype::Vec3Types, defaulttype::Vec1Types >;
#endif

}
