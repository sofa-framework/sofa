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
#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::mapping::nonlinear
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class InDataTypes, class OutDataTypes>
class DistanceFromTargetMappingInternalData
{
public:
};


struct BaseDistanceFromTargetMapping
{
    virtual void updateTarget( unsigned index, SReal x, SReal y, SReal z ) = 0;
};



/** Maps point positions to distances from target points.
 *   Only a subset of the parent points is mapped. This can be used to constrain the trajectories of one or several particles.

 *   @tparam TIn: parent point positions
 *   @tparam TOut: distance from each point to a target position, minus a rest distance.

 *   (changed class name on Feb. 4, 2014, previous name was DistanceMapping)
 * @author Francois Faure
  */
template <class TIn, class TOut>
class DistanceFromTargetMapping : public core::Mapping<TIn, TOut>, public BaseDistanceFromTargetMapping, public NonLinearMappingData<true>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(DistanceFromTargetMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real Real;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef linearalgebra::EigenSparseMatrix<TIn,TOut>    SparseMatrixEigen;
    typedef linearalgebra::EigenSparseMatrix<In,In>    SparseKMatrixEigen;
    enum {Nin = In::deriv_total_size, Nout = Out::deriv_total_size };
    typedef type::Vec<In::deriv_total_size> Direction;

    Data<type::vector<unsigned>> f_indices;          ///< Indices of the parent points
    Data<InVecCoord> f_targetPositions;              ///< Positions to compute the distances from
    Data<type::vector<Real>> f_restDistances;        ///< Rest lengths of the connections

    /// Add a target with a desired distance
    void createTarget( unsigned index, const InCoord& position, Real distance);

    /// Update the position of a target
    void updateTarget( unsigned index, const InCoord& position);
    void updateTarget( unsigned index, SReal x, SReal y, SReal z ) override;

    /// Remove all targets
    void clear();

    void init() override;

    void apply(const core::MechanicalParams *mparams, Data<OutVecCoord>& out, const Data<InVecCoord>& in) override;

    void applyJ(const core::MechanicalParams *mparams, Data<OutVecDeriv>& out, const Data<InVecDeriv>& in) override;

    void applyJT(const core::MechanicalParams *mparams, Data<InVecDeriv>& out, const Data<OutVecDeriv>& in) override;

    void applyJT(const core::ConstraintParams *cparams, Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in) override;

    void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentForce, core::ConstMultiVecDerivId  childForce ) override;

    const sofa::linearalgebra::BaseMatrix* getJ() override;
    virtual const type::vector<sofa::linearalgebra::BaseMatrix*>* getJs() override;

    void updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForce ) override;
    const linearalgebra::BaseMatrix* getK() override;
    void buildGeometricStiffnessMatrix(sofa::core::GeometricStiffnessMatrix* matrices) override;

    void draw(const core::visual::VisualParams* vparams) override;
    Data<float> d_showObjectScale; ///< Scale for object display
    Data<sofa::type::RGBAColor> d_color; ///< Color for object display. (default=[1.0,1.0,0.0,1.0])

protected:
    DistanceFromTargetMapping();
    ~DistanceFromTargetMapping() override;

    SparseMatrixEigen jacobian;                      ///< Jacobian of the mapping
    type::vector<linearalgebra::BaseMatrix*> baseMatrices;   ///< Jacobian of the mapping, in a vector
    SparseKMatrixEigen K;  ///< Assembled geometric stiffness matrix
    type::vector<Direction> directions;                         ///< Unit vectors in the directions of the lines
    type::vector< Real > invlengths;                          ///< inverse of current distances. Null represents the infinity (null distance)

    /// r=b-a only for position (eventual rotation, affine transform... remains null)
    void computeCoordPositionDifference( Direction& r, const InCoord& a, const InCoord& b );

};


#if !defined(SOFA_COMPONENT_MAPPING_DistanceFromTargetMapping_CPP)
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API DistanceFromTargetMapping< defaulttype::Vec3Types, defaulttype::Vec1Types >;
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API DistanceFromTargetMapping< defaulttype::Vec1Types, defaulttype::Vec1Types >;
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API DistanceFromTargetMapping< defaulttype::Rigid3Types, defaulttype::Vec1Types >;


#endif

} // namespace sofa::component::mapping::nonlinear
