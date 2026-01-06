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

#include <sofa/component/solidmechanics/fem/elastic/TetrahedronFEMForceField.h>
#include <sofa/simulation/task/CpuTask.h>
#include <sofa/simulation/task/TaskScheduler.h>

#include <thread>

namespace multithreading::component::forcefield::solidmechanics::fem::elastic
{

/**
 * Parallel implementation of TetrahedronFEMForceField
 *
 * This implementation is the most efficient when:
 * 1) the number of tetrahedron is large (> 1000)
 *
 * The following methods are executed in parallel:
 * - addDForce
 * - addKToMatrix
 */
template<class DataTypes>
class SOFA_MULTITHREADING_PLUGIN_API ParallelTetrahedronFEMForceField :
    virtual public sofa::component::solidmechanics::fem::elastic::TetrahedronFEMForceField<DataTypes>,
    public sofa::simulation::TaskSchedulerUser
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ParallelTetrahedronFEMForceField, DataTypes), SOFA_TEMPLATE(sofa::component::solidmechanics::fem::elastic::TetrahedronFEMForceField, DataTypes));

    using VecCoord = typename DataTypes::VecCoord;
    using VecDeriv = typename DataTypes::VecDeriv;
    using VecReal = typename DataTypes::VecReal;
    using Coord = typename DataTypes::Coord;
    using Deriv = typename DataTypes::Deriv;
    using Real = typename Coord::value_type;

    using DataVecDeriv = sofa::core::objectmodel::Data<VecDeriv>;
    using DataVecCoord = sofa::core::objectmodel::Data<VecCoord>;

    using Element = sofa::core::topology::BaseMeshTopology::Tetra;
    using VecElement = sofa::core::topology::BaseMeshTopology::SeqTetrahedra;

    using StiffnessMatrix = sofa::type::Mat<12, 12, Real>;
    using Transformation = sofa::type::MatNoInit<3, 3, Real>;


    void init() override;

    void addForce (const sofa::core::MechanicalParams* mparams, DataVecDeriv& d_f,
                   const DataVecCoord& d_x, const DataVecDeriv& d_v) override;

    void addDForce (const sofa::core::MechanicalParams* mparams, DataVecDeriv& d_df,
                    const DataVecDeriv& d_dx) override;

    void addKToMatrix(sofa::linearalgebra::BaseMatrix *mat, SReal kFactor, unsigned int &offset) override;

protected:

    template<class Function>
    void addDForceGeneric(VecDeriv& df, const VecDeriv& dx, Real kFactor,
                           const VecElement& indexedElements, Function f);

    void addDForceSmall(VecDeriv& df, const VecDeriv& dx, Real kFactor,
                           const VecElement& indexedElements);
    void addDForceCorotational(VecDeriv& df, const VecDeriv& dx, Real kFactor,
                           const VecElement& indexedElements);

    void drawTrianglesFromTetrahedra(const sofa::core::visual::VisualParams* vparams,
                                     bool showVonMisesStressPerElement, bool drawVonMisesStress,
                                     const VecCoord& x,
                                     const VecReal& youngModulus, bool heterogeneous,
                                     Real minVM,
                                     Real maxVM,
                                     sofa::helper::ReadAccessor<sofa::Data<sofa::type::vector<Real>>> vM) override;

    std::map<std::thread::id, VecDeriv> m_threadLocal_df;

};

#if  !defined(SOFA_MULTITHREADING_PARALLELTETRAHEDRONFEMFORCEFIELD_CPP)
extern template class SOFA_MULTITHREADING_PLUGIN_API ParallelTetrahedronFEMForceField<sofa::defaulttype::Vec3Types>;
#endif

} //namespace sofa::component::forcefield
