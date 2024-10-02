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
#include <sofa/component/mapping/nonlinear/BaseNonLinearMapping.h>
#include <sofa/component/mapping/nonlinear/NonLinearMappingData.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/core/objectmodel/RenamedData.h>

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
class DistanceFromTargetMapping : public BaseNonLinearMapping<TIn, TOut, true>, public BaseDistanceFromTargetMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(DistanceFromTargetMapping,TIn,TOut), SOFA_TEMPLATE3(BaseNonLinearMapping,TIn,TOut, true));

    using In = TIn;
    using Out = TOut;
    using Real = Real_t<Out>;
    static constexpr auto Nin = In::deriv_total_size;
    static constexpr auto Nout = Out::deriv_total_size;

    using InCoord = Coord_t<In>;
    using InVecCoord = VecCoord_t<In>;

    typedef type::Vec<In::deriv_total_size> Direction;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MAPPING_NONLINEAR()
    sofa::core::objectmodel::RenamedData<type::vector<unsigned>> f_indices;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MAPPING_NONLINEAR()
    sofa::core::objectmodel::RenamedData<InVecCoord> f_targetPositions;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MAPPING_NONLINEAR()
    sofa::core::objectmodel::RenamedData<type::vector<Real>> f_restDistances;

// d_showObjectScale and d_color are already up-to-date with the new naming convention, so they do not require deprecation notices.

    Data<type::vector<unsigned>> d_indices; ///< Indices of the parent points
    Data<InVecCoord> d_targetPositions; ///< Positions to compute the distances from
    Data<type::vector<Real>> d_restDistances; ///< Rest lengths of the connections

    /// Add a target with a desired distance
    void createTarget( unsigned index, const InCoord& position, Real distance);

    /// Update the position of a target
    void updateTarget( unsigned index, const InCoord& position);
    void updateTarget( unsigned index, SReal x, SReal y, SReal z ) override;

    /// Remove all targets
    void clear();

    void init() override;

    void apply(const core::MechanicalParams *mparams, DataVecCoord_t<Out>& out, const DataVecCoord_t<In>& in) override;
    void buildGeometricStiffnessMatrix(sofa::core::GeometricStiffnessMatrix* matrices) override;

    void draw(const core::visual::VisualParams* vparams) override;
    Data<float> d_showObjectScale; ///< Scale for object display
    Data<sofa::type::RGBAColor> d_color; ///< Color for object display. (default=[1.0,1.0,0.0,1.0])

protected:
    DistanceFromTargetMapping();
    ~DistanceFromTargetMapping() override;

    void matrixFreeApplyDJT(const core::MechanicalParams* mparams, Real kFactor,
                        Data<VecDeriv_t<In> >& parentForce,
                        const Data<VecDeriv_t<In> >& parentDisplacement,
                        const Data<VecDeriv_t<Out> >& childForce) override;

    using typename Inherit1::SparseKMatrixEigen;

    void doUpdateK(
        const core::MechanicalParams* mparams, const Data<VecDeriv_t<Out> >& childForce,
        SparseKMatrixEigen& matrix) override;

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
