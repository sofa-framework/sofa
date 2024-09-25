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
#include <sofa/component/mapping/nonlinear/NonLinearMappingData.h>
#include <sofa/core/Mapping.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>


namespace sofa::component::mapping::nonlinear
{

template <class TIn, class TOut, bool HasStabilizedGeometricStiffness>
class BaseNonLinearMapping : public core::Mapping<TIn, TOut>, public NonLinearMappingData<HasStabilizedGeometricStiffness>
{
public:
    SOFA_CLASS(
        SOFA_TEMPLATE3(BaseNonLinearMapping,TIn,TOut, HasStabilizedGeometricStiffness),
        SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    using In = TIn;
    using Out = TOut;

    using Real = Real_t<Out>;

    typedef linearalgebra::EigenSparseMatrix<TIn,TOut> SparseMatrixEigen;
    static constexpr auto Nin = In::deriv_total_size;

    void init() override;

    void applyJ(const core::MechanicalParams* mparams, DataVecDeriv_t<Out>& out, const DataVecDeriv_t<In>& in) final;
    void applyJT(const core::MechanicalParams* mparams, DataVecDeriv_t<In>& out, const DataVecDeriv_t<Out>& in) final;
    void applyJT(const core::ConstraintParams *cparams, DataMatrixDeriv_t<In>& out, const DataMatrixDeriv_t<Out>& in) final;
    void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentForceId, core::ConstMultiVecDerivId childForceId) final;

    const linearalgebra::BaseMatrix* getK() final;
    const type::vector<sofa::linearalgebra::BaseMatrix*>* getJs() override;

    void updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForceId) final;

protected:

    using SparseKMatrixEigen = linearalgebra::EigenSparseMatrix<TIn,TIn>;

    virtual void matrixFreeApplyDJT(const core::MechanicalParams* mparams,
                                   Real kFactor,
                                   Data<VecDeriv_t<In> >& parentForce,
                                   const Data<VecDeriv_t<In> >& parentDisplacement,
                                   const Data<VecDeriv_t<Out> >& childForce) = 0;

    SparseMatrixEigen jacobian; ///< Jacobian of the mapping

    virtual void doUpdateK(
        const core::MechanicalParams* mparams, const Data<VecDeriv_t<Out> >& childForce,
        SparseKMatrixEigen& matrix) = 0;

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
        typename In::CPos jacobianValue;
        bool operator<(const JacobianEntry& other) const { return vertexId < other.vertexId;}
    };

private:

    SparseKMatrixEigen m_geometricStiffnessMatrix; ///< Assembled geometric stiffness matrix
    type::vector<linearalgebra::BaseMatrix*> m_baseMatrices;
};

}
