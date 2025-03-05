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

#include <SofaMatrix/config.h>

#include <sofa/core/DataEngine.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/core/topology/BaseTopologyData.h>

#include <sofa/helper/OptionsGroup.h>

extern "C" {
#include <metis.h>
}

namespace sofa::component::linearsolver
{

/**
 * A DataEngine to reorder the degrees of freedom in a mesh in order to reduce fill-in in sparse matrix factorization.
 *
 * In other terms, the algorithm minimizes the number of non-zeros entries in the factorization of the sparse matrix of
 * a FEM problem by reordering the degrees of freedom.
 *
 * The implementation is based on METIS and Eigen.
 *
 * Note: some of the direct linear solvers embed such a reordering internally (e.g. SparseLDLSolver).
 */
template <class DataTypes>
class FillReducingOrdering : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(FillReducingOrdering,DataTypes),core::DataEngine);

    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

protected:

    FillReducingOrdering();
    ~FillReducingOrdering() override {}

    void init() override;
    void reinit() override;
    void reorderByEigen();
    void doUpdate() override;

    SingleLink<FillReducingOrdering<DataTypes>, core::behavior::MechanicalState<DataTypes>, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> l_mstate;
    SingleLink<FillReducingOrdering<DataTypes>, core::topology::BaseMeshTopology,           BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> l_topology;

    Data<sofa::helper::OptionsGroup> d_orderingMethod; ///< Ordering method.
                                                       ///< nestedDissection is the multilevel nested dissection algorithm implemented in the METIS library
                                                       ///< approximateMinimumDegree is the approximate minimum degree algorithm implemented in the Eigen library.

    /// Output vector of indices mapping the reordered vertices to the initial list
    Data< sofa::type::vector<idx_t> > d_permutation;
    /// Output vector of indices mapping the initial vertices to the reordered list
    Data< sofa::type::vector<idx_t> > d_invPermutation;

    /// Reordered position vector
    Data< VecCoord > d_position;
    /// Reordered hexahedra
    Data< sofa::type::vector<sofa::topology::Hexahedron> > d_hexahedra;
    /// Reordered tetrahedra
    Data< sofa::type::vector<sofa::topology::Tetrahedron> > d_tetrahedra;


    void reorderByMetis();

    /// Build the required mesh data structure for the METIS_MeshToNodal call
    template<class Elements>
    void initializeFromElements(const Elements& elements, std::vector<idx_t>& eptr, std::vector<idx_t>& eind, unsigned int& eptr_id, unsigned int& eind_id);

    /// Update the topology order from the permutation array
    template<class Element>
    void updateElements(const sofa::type::vector<Element>& inElementSequence, Data< sofa::type::vector<Element> >& outElementSequenceData);

    void updateMesh();
};

#if !defined(SOFA_COMPONENT_ENGINE_FillReducingOrdering_CPP)
extern template class SOFA_SOFAMATRIX_API FillReducingOrdering<defaulttype::Vec3Types>;
#endif

}// namespace sofa::component::linearsolver
