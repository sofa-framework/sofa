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

#include <MultiThreading/config.h>
#include <sofa/simulation/task/TaskSchedulerUser.h>

#include <sofa/component/solidmechanics/fem/elastic/HexahedronFEMForceField.h>

namespace multithreading::component::forcefield::solidmechanics::fem::elastic
{

/**
 * Parallel implementation of HexahedronFEMForceField
 *
 * This implementation is the most efficient when:
 * 1) the number of hexahedron is large (> 1000)
 * 2) the global system matrix is not assembled. It is usually the case with a CGLinearSolver templated with GraphScattered types.
 * 3) the method is 'large'. If the method is 'polar' or 'small', addForce is executed sequentially, but addDForce in parallel.
 *
 * The following methods are executed in parallel:
 * - addForce for method 'large'.
 * - addDForce
 *
 * The method addKToMatrix is not executed in parallel. This method is called with an assembled system, usually with
 * a direct solver or a CGLinearSolver templated with types different from GraphScattered. In this case, the most
 * time-consumming step is to invert the matrix. This is where efforts should be put to accelerate the simulation.
 */
template<class DataTypes>
class SOFA_MULTITHREADING_PLUGIN_API ParallelHexahedronFEMForceField :
    virtual public sofa::component::solidmechanics::fem::elastic::HexahedronFEMForceField<DataTypes>,
    public sofa::simulation::TaskSchedulerUser
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ParallelHexahedronFEMForceField, DataTypes), SOFA_TEMPLATE(sofa::component::solidmechanics::fem::elastic::HexahedronFEMForceField, DataTypes));

    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef sofa::core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef sofa::core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef typename Coord::value_type Real;
    typedef sofa::helper::ReadAccessor< sofa::Data< VecCoord > > RDataRefVecCoord;
    typedef sofa::helper::WriteAccessor< sofa::Data< VecDeriv > > WDataRefVecDeriv;
    typedef sofa::core::topology::BaseMeshTopology::Hexa Element;
    typedef sofa::core::topology::BaseMeshTopology::SeqHexahedra VecElement;
    typedef sofa::type::Mat<24, 24, Real> ElementStiffness;
    typedef sofa::helper::vector<ElementStiffness> VecElementStiffness;

    void init() override;

    void addForce (const sofa::core::MechanicalParams* mparams, DataVecDeriv& f,
                   const DataVecCoord& x, const DataVecDeriv& v) override;

    /**
     * The computation is done in 2 steps:
     * 1) Elements are visited in parallel: a force derivative is computed inside each element
     * 2) Vertices are visited in parallel: the force derivative in all adjacent hexahedra are
     * accumulated in the vertices
     */
    void addDForce (const sofa::core::MechanicalParams* mparams, DataVecDeriv& df,
                    const DataVecDeriv& dx) override;

protected:

    // code duplicated from HexahedronFEMForceField::accumulateForceLarge but adapted to be thread-safe
    void computeTaskForceLarge(RDataRefVecCoord& p, sofa::Index elementId, const Element& elem,
                               const VecElementStiffness& elementStiffnesses, SReal& OutPotentialEnery,
                               sofa::type::Vec<8, Deriv>& OutF);

    /// Assuming a vertex has 8 adjacent hexahedra, the array stores where the vertex is referenced in each of the adjacent hexahedra
    using HexaAroundVerticesIndex = sofa::type::fixed_array<sofa::Size, 8>;

    /// Where all vertex ids are stored in their adjacent hexahedra
    sofa::type::vector<HexaAroundVerticesIndex> m_vertexIdInAdjacentHexahedra;

    /// A list of DF corresponding to all elements. It is stored as a class member to avoid to reallocate it
    sofa::type::vector<sofa::type::Vec<8, Deriv> > m_elementsDf;

    /// Cache the list of hexahedra around vertices
    sofa::type::vector<sofa::core::topology::BaseMeshTopology::HexahedraAroundVertex> m_around;

private:
    bool updateStiffnessMatrices; /// cache to avoid calling 'getValue' on d_updateStiffnessMatrix
};

#if  !defined(SOFA_MULTITHREADING_PARALLELHEXAHEDRONFEMFORCEFIELD_CPP)
extern template class SOFA_MULTITHREADING_PLUGIN_API ParallelHexahedronFEMForceField<sofa::defaulttype::Vec3Types>;
#endif

} //namespace sofa::component::forcefield
