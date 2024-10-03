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
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/RGBAColor.h>

#include <sofa/core/objectmodel/RenamedData.h>

namespace sofa::component::mapping::nonlinear
{


/** Maps point positions to distances (in distance unit).
 * @tparam TIn parent point positions
 * @tparam TOut corresponds to a scalar value: distance between point pairs, minus a rest distance.
 * The pairs are given in an EdgeSetTopologyContainer in the same node.
 * If the rest lengths are not defined, they are set using the initial values.
 * If computeDistance is set to true, the rest lengths are set to 0.
 * (Changed class name on Feb. 4, 2014, previous name was ExtensionMapping)

 * @author Francois Faure
 */
template <class TIn, class TOut>
class DistanceMapping : public BaseNonLinearMapping<TIn, TOut, true>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(DistanceMapping,TIn,TOut), SOFA_TEMPLATE3(BaseNonLinearMapping,TIn,TOut,true));

    typedef TIn In;
    typedef TOut Out;

    using Real = Real_t<Out>;

    static constexpr auto Nin = In::deriv_total_size;

    typedef sofa::core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef type::Vec<In::spatial_dimensions,Real> Direction;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MAPPING_NONLINEAR()
    sofa::core::objectmodel::RenamedData<bool> f_computeDistance;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MAPPING_NONLINEAR()
    sofa::core::objectmodel::RenamedData<type::vector<Real>> f_restLengths;


    Data<bool> d_computeDistance; ///< if 'computeDistance = true', then rest length of each element equal 0, otherwise rest length is the initial lenght of each of them
    Data<type::vector<Real>> d_restLengths; ///< Rest lengths of the connections
    Data<Real> d_showObjectScale; ///< Scale for object display
    Data<sofa::type::RGBAColor> d_color; ///< Color for object display. (default=[1.0,1.0,0.0,1.0])

    /// Link to be set to the topology container in the component graph. 
    SingleLink<DistanceMapping<TIn, TOut>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    void init() override;

    void apply(const core::MechanicalParams *mparams, DataVecCoord_t<Out>& out, const DataVecCoord_t<In>& in) override;
    void buildGeometricStiffnessMatrix(sofa::core::GeometricStiffnessMatrix* matrices) override;

    void computeBBox(const core::ExecParams* params, bool onlyVisible) override;
    void draw(const core::visual::VisualParams* vparams) override;

protected:
    DistanceMapping();

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
    void computeCoordPositionDifference( Direction& r, const Coord_t<In>& a, const Coord_t<In>& b );

    using JacobianEntry = typename Inherit1::JacobianEntry;
};





#if !defined(SOFA_COMPONENT_MAPPING_DistanceMapping_CPP)
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API DistanceMapping< defaulttype::Vec3Types, defaulttype::Vec1Types >;
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API DistanceMapping< defaulttype::Rigid3Types, defaulttype::Vec1Types >;
#endif

} // namespace sofa::component::mapping::nonlinear
