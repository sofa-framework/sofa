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

#include <sofa/component/constraint/lagrangian/solver/ConstraintForceExporter.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/AnimateEndEvent.h>

namespace sofa::component::constraint::lagrangian::solver
{

template <class DataTypes>
ConstraintForceExporter<DataTypes>::ConstraintForceExporter()
    : Inherited()
    , d_constraintForces(initData(&d_constraintForces, "constraintForces", "(output) Forces (or wrenches if rigids) computed from the mechanical state using the constraint solver."))
    , d_draw(initData(&d_draw, true, "draw", "(debug) draw forces (as a line) for each position of the mechanical object."))
    , d_drawForceScale(initData(&d_drawForceScale, 1.0, "drawForceScale", "(debug) Scale to apply on the force (draw as a line)."))
    , l_constraintSolver(initLink("constraintSolver", "Link to the constraint solver."))
{
    this->f_listening.setValue(true);

}

template <class DataTypes>
void ConstraintForceExporter<DataTypes>::init()
{
    if (!l_constraintSolver)
    {
        l_constraintSolver.set(this->getContext()->template get<ConstraintSolverImpl>());
    }

    if (!l_constraintSolver)
    {
        msg_error() << "No constraint solver found in the current context, whereas it is required. This component retrieves the matrix from a linear solver.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    // No need to check if the link to the mechanical state is correct
    // as it has been covered by canCreate()
    // Inherited::init();

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

template <class DataTypes>
void ConstraintForceExporter<DataTypes>::handleEvent(core::objectmodel::Event* event)
{
    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
    {
        return;
    }

    Inherited::handleEvent(event);

    if (simulation::AnimateEndEvent::checkEventType(event))
    {
        computeForce();
    }
}

template <class DataTypes>
void ConstraintForceExporter<DataTypes>::draw(const sofa::core::visual::VisualParams* vparams)
{
    if (!d_draw.getValue() || this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
    {
        return;
    }

    auto constraintForces = sofa::helper::getReadAccessor(d_constraintForces);
    const auto& positions = this->getMState()->read(sofa::core::ConstVecCoordId::position())->getValue();

    assert(constraintForces.size() != positions.size());

    const float lineSize = 2.0f;
    const double forceScale = d_drawForceScale.getValue();

    vparams->drawTool()->saveLastState();

    std::vector<type::Vector3> vertices;
    vertices.reserve(constraintForces.size() * 2);

    for(std::size_t i = 0 ; i < constraintForces.size() ; i++)
    {
        const auto& origin = positions[i];
        const auto& force = DataTypes::getDPos(constraintForces[i]);

        vertices.emplace_back( origin[0],
                               origin[1],
                               origin[2] );
        vertices.emplace_back( origin[0] + force[0] * forceScale,
                               origin[1] + force[1] * forceScale,
                               origin[2] + force[2] * forceScale );
    }
    vparams->drawTool()->drawLines(vertices, lineSize, sofa::type::RGBAColor::white());

    vparams->drawTool()->restoreLastState();
}

template <class DataTypes>
void ConstraintForceExporter<DataTypes>::computeForce()
{
    auto constraintForces = sofa::helper::getWriteOnlyAccessor(d_constraintForces);

    auto* constraintProblem = l_constraintSolver->getConstraintProblem();
    const auto& constraintMatrix = this->getMState()->read(core::ConstMatrixDerivId::constraintJacobian())->getValue();
    auto* lambdas = constraintProblem->getF();
    [[maybe_unused]] const int dimension = constraintProblem->getDimension();
    const auto& positions = this->getMState()->read(sofa::core::ConstVecCoordId::position())->getValue();

    assert(lambdas != nullptr);

    constraintForces.clear();
    constraintForces.resize(positions.size());

    // force are supposed to already take into account dt
    const auto dt = this->getContext()->getDt();

    for (auto rowIt = constraintMatrix.begin(); rowIt != constraintMatrix.end(); ++rowIt)
    {
        assert(rowIt.index() <= dimension);
        for (auto colIt = rowIt.begin(); colIt != rowIt.end(); ++colIt)
        {
            constraintForces[colIt.index()] += colIt.val() * lambdas[rowIt.index()] / dt;
        }
    }
}

} // namespace sofa::component::constraint::lagrangian::solver
