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
class AreaMapping : public core::Mapping<TIn, TOut>, public NonLinearMappingData<true>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(AreaMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef TIn In;
    typedef TOut Out;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real Real;

    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;

    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;

    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;

    typedef linearalgebra::EigenSparseMatrix<TIn,TOut> SparseMatrixEigen;
    static constexpr auto Nin = In::deriv_total_size;

    Data<bool> d_computeRestArea;
    Data<type::vector<Real>> d_restArea;

    SingleLink<AreaMapping<TIn, TOut>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;


    void init() override;

    void apply(const core::MechanicalParams* mparams, OutDataVecCoord& out, const InDataVecCoord& in) override;
    void applyJ(const core::MechanicalParams* mparams, OutDataVecDeriv& out, const InDataVecDeriv& in) override;
    void applyJT(const core::MechanicalParams* mparams, InDataVecDeriv& out, const OutDataVecDeriv& in) override;
    void applyJT(const core::ConstraintParams *cparams, InDataMatrixDeriv& out, const OutDataMatrixDeriv& in) override;

protected:
    AreaMapping();

    SparseMatrixEigen jacobian; ///< Jacobian of the mapping

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
        bool operator<(const JacobianEntry& other) { return vertexId < other.vertexId;}
    };
};

#if !defined(SOFA_COMPONENT_MAPPING_NONLINEAR_AREAMAPPING_CPP)
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API AreaMapping< defaulttype::Vec3Types, defaulttype::Vec1Types >;
#endif

}
