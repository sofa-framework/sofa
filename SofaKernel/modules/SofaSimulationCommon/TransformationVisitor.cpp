/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaSimulationCommon/TransformationVisitor.h>

namespace sofa
{

namespace simulation
{

void TransformationVisitor::processVisualModel(simulation::Node* // node
        , core::visual::VisualModel* v)
{
    v->applyScale ( scale[0], scale[1], scale[2] );
    v->applyRotation(rotation[0],rotation[1],rotation[2]);
    v->applyTranslation ( translation[0],translation[1],translation[2] );
}

void TransformationVisitor::processMechanicalState(simulation::Node* // node
        , core::behavior::BaseMechanicalState* m)
{
    m->applyScale ( scale[0], scale[1], scale[2]  );
    m->applyRotation(rotation[0],rotation[1],rotation[2]);
    m->applyTranslation ( translation[0],translation[1],translation[2] );
}

Visitor::Result TransformationVisitor::processNodeTopDown(simulation::Node* node)
{
    for_each(this, node, node->visualModel, &TransformationVisitor::processVisualModel);
    for_each(this, node, node->mechanicalState, &TransformationVisitor::processMechanicalState);

    return RESULT_CONTINUE;
}

} // namespace simulation

} // namespace sofa
