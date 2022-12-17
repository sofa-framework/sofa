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

#include <sofa/component/solidmechanics/fem/elastic/TetrahedronFEMForceField.h>
#include <sofa/simulation/CpuTask.h>
#include <sofa/simulation/TaskScheduler.h>

#include <thread>

namespace sofa::component::forcefield
{

template<class DataTypes>
class AccumulateForceLargeTasks;

template<class DataTypes>
class AddDForceTask;

/**
 * Parallel implementation of TetrahedronFEMForceField
 *
 * This implementation is the most efficient when:
 * 1) the number of tetrahedron is large (> 1000)
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
class SOFA_MULTITHREADING_PLUGIN_API ParallelTetrahedronFEMForceField : virtual public sofa::component::solidmechanics::fem::elastic::TetrahedronFEMForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ParallelTetrahedronFEMForceField, DataTypes), SOFA_TEMPLATE(sofa::component::solidmechanics::fem::elastic::TetrahedronFEMForceField, DataTypes));

    using VecCoord = typename DataTypes::VecCoord;
    using VecDeriv = typename DataTypes::VecDeriv;
    using VecReal = typename DataTypes::VecReal;
    using Coord = typename DataTypes::Coord;
    using Deriv = typename DataTypes::Deriv;
    using Real = typename Coord::value_type;

    using DataVecDeriv = core::objectmodel::Data<VecDeriv>;
    using DataVecCoord = core::objectmodel::Data<VecCoord>;

    using VecElement = core::topology::BaseMeshTopology::SeqTetrahedra;



    void init() override;

    void addForce (const core::MechanicalParams* mparams, DataVecDeriv& d_f,
                   const DataVecCoord& d_x, const DataVecDeriv& d_v) override;




    void addDForce (const core::MechanicalParams* mparams, DataVecDeriv& d_df,
                    const DataVecDeriv& d_dx) override;

protected:

    template<class Function>
    void addDForceGeneric(VecDeriv& df, const VecDeriv& dx, Real kFactor,
                           const VecElement& indexedElements, Function f);

    void addDForceSmall(VecDeriv& df, const VecDeriv& dx, Real kFactor,
                           const VecElement& indexedElements);
    void addDForceCorotational(VecDeriv& df, const VecDeriv& dx, Real kFactor,
                           const VecElement& indexedElements);

    void initTaskScheduler();

    sofa::simulation::TaskScheduler* m_taskScheduler { nullptr };

    std::map<std::thread::id, VecDeriv> m_threadLocal_df;

};

#if  !defined(SOFA_MULTITHREADING_PARALLELTETRAHEDRONFEMFORCEFIELD_CPP)
extern template class SOFA_MULTITHREADING_PLUGIN_API ParallelTetrahedronFEMForceField<defaulttype::Vec3Types>;
#endif

} //namespace sofa::component::forcefield
