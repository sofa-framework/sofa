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
#include <sofa/component/engine/select/config.h>

#include <sofa/core/behavior/PairInteractionProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/vector.h>
#include <sofa/core/topology/TopologySubsetIndices.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/config.h>
#include <sofa/topology/Edge.h>

namespace sofa::component::engine::select
{

/**
 * Given two mechanical states, find correspondance between degrees of freedom, based on the minimal distance.
 *
 * Project all the points from the second mechanical state on the first one. This done by finding the point in the
 * first mechanical state closest to each point in the second mechanical state. If the distance is less than a provided
 * distance (named radius), the indices of the degrees of freedom in their respective mechanical states is added to
 * an output list.
 *
 */
template <class DataTypes>
class NearestPointROI : public sofa::core::DataEngine, public core::behavior::PairStateAccessor<DataTypes, DataTypes>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(NearestPointROI, DataTypes), sofa::core::DataEngine, SOFA_TEMPLATE2(core::behavior::PairStateAccessor, DataTypes, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef type::vector<unsigned int> SetIndexArray;
    typedef sofa::core::topology::TopologySubsetIndices SetIndex;

    SetIndex d_inputIndices1; ///< Only these indices are considered in the first model
    SetIndex d_inputIndices2; ///< Only these indices are considered in the second model
    Data<Real> f_radius; ///< Radius to search corresponding fixed point if no indices are given
    Data<bool> d_useRestPosition; ///< If true will use rest position only at init. Otherwise will recompute the maps at each update. Default is true.

    /// Output Data
    ///@{
    SetIndex f_indices1; ///< Indices of the source points on the first model
    SetIndex f_indices2; ///< Indices of the fixed points on the second model
    Data< sofa::type::vector<sofa::topology::Edge> > d_edges; ///< List of edges. The indices point to a list composed as an interleaved fusion of output degrees of freedom. It could be used to fuse two mechanical objects and create a topology from the fusion.
    Data< type::vector<unsigned> > d_indexPairs;        ///< Two indices per child: the parent, and the index within the parent. Could be used with a SubsetMultiMapping
    Data< type::vector<Real> > d_distances; ///< List of distances between pairs of points
    ///@}

    explicit NearestPointROI(core::behavior::MechanicalState<DataTypes> * = nullptr, core::behavior::MechanicalState<DataTypes> *mm2 = nullptr);
    ~NearestPointROI() override;

    void init() override;
    void reinit() override;
    void doUpdate() override;

protected:
    void computeNearestPointMaps(const VecCoord& x1, const VecCoord& x2);
};


#if !defined(SOFA_COMPONENT_ENGINE_NearestPointROI_CPP)
extern template class SOFA_COMPONENT_ENGINE_SELECT_API NearestPointROI<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API NearestPointROI<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API NearestPointROI<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API NearestPointROI<defaulttype::Vec6Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API NearestPointROI<defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API NearestPointROI<defaulttype::Rigid2Types>;
#endif

} //namespace sofa::component::engine::select
