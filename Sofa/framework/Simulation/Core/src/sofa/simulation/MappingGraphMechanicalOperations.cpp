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
#include <sofa/simulation/MappingGraphMechanicalOperations.h>

namespace sofa::simulation::common
{
void MappingGraphMechanicalOperations::computeForce(const MappingGraph& mappingGraph,
                                                    core::MultiVecDerivId result,
                                                    bool clearForceBefore,
                                                    bool accumulateForcesFromMappedStates,
                                                    TaskScheduler* taskScheduler)
{
    //assumes the mapping graph is valid and properly initialized

    setF(result);
    if (clearForceBefore)
    {
        /**
         * Reset forces on all mechanical states in the mapping graph. This operation can be performed
         * in any order on all states in the graph.
         */
        mappingGraph.algorithms.traverse_([&](core::behavior::BaseMechanicalState& state)
        {
            const core::VecDerivId& stateForce = result.getId(&state);
            state.resetForce(&mparams, stateForce);
        });
    }

    /**
     * Compute f += externalForce on all mechanical states in the mapping graph. This operation can
     * be performed in any order on all states in the graph.
     */
    mappingGraph.algorithms.traverse_([&](core::behavior::BaseMechanicalState& state)
    {
        const core::VecDerivId& stateForce = result.getId(&state);
        state.accumulateForce(&mparams, stateForce);
    });

    /**
     * Compute f += f(x) on all force fields in the mapping graph. This operation can be performed
     * in any order on all states in the graph, and can be parallelized among component groups.
     */
    mappingGraph.algorithms.traverseComponentGroups_([&](core::behavior::BaseForceField& forceField)
    {
        forceField.addForce(&mparams, result);
    }, sofa::simulation::VisitorApplication::ALL_NODES, taskScheduler);

    if (accumulateForcesFromMappedStates)
    {
        /**
         * Compute f_in += J^T * f_out using mappings in the mapping graph. This operation must be
         * performed in a bottom-up order to ensure correct force accumulation.
         */
        mappingGraph.algorithms.traverseBottomUp_([&](core::BaseMapping& mapping)
        {
            mapping.applyJT(&mparams, result, result);
        });
    }
}
void MappingGraphMechanicalOperations::addMBKv(const MappingGraph& mappingGraph,
                                               core::MultiVecDerivId df, core::MatricesFactors::M m,
                                               core::MatricesFactors::B b,
                                               core::MatricesFactors::K k, bool clear,
                                               bool accumulate)
{
    const core::ConstMultiVecDerivId dx = mparams.dx();
    mparams.setDx(mparams.v());
    setDf(df);
    if (clear)
    {
        /**
         * Reset forces on all mapped mechanical states in the mapping graph. This operation can be performed
         * in any order on all mapped states in the graph.
         */
        mappingGraph.algorithms.traverse_([&](core::behavior::BaseMechanicalState& state)
        {
            const core::VecDerivId& stateForce = df.getId(&state);
            state.resetForce(&mparams, stateForce);
        }, VisitorApplication::ONLY_MAPPED_NODES);
    }
    mparams.setBFactor(b.get());
    mparams.setKFactor(k.get());
    mparams.setMFactor(m.get());
    /* useV = true */

    mappingGraph.algorithms.traverseComponentGroups_([&](core::behavior::BaseForceField& forceField)
    {
        forceField.addMBKdx(&mparams, df);
    });

    if (accumulate)
    {
        mappingGraph.algorithms.traverseBottomUp_([&](core::BaseMapping& mapping)
        {
            mapping.applyJT(&mparams, df, df);
            if( mparams.kFactor() != 0 )
            {
                mapping.applyDJT(&mparams, df, df);
            }
        });
    }

    mparams.setDx(dx);
}
}  // namespace sofa::simulation::common
