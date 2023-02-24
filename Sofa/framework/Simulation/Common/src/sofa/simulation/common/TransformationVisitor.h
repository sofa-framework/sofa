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
#include <sofa/simulation/common/config.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Quat.h>
#include <sofa/core/visual/VisualModel.h>

namespace sofa::simulation
{

namespace
{
    using sofa::type::Vec3;
}

class SOFA_SIMULATION_COMMON_API TransformationVisitor : public Visitor
{
public:
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vector3, sofa::type::Vec3);

    TransformationVisitor(const sofa::core::ExecParams* params);

    void setTranslation(SReal dx, SReal dy, SReal dz) { translation = Vec3(dx,dy,dz);}
    void setRotation(SReal rx, SReal ry, SReal rz) {    rotation=Vec3(rx,ry,rz);	}
    void setScale(SReal sx, SReal sy, SReal sz) {scale=Vec3(sx,sy,sz);}

    void processVisualModel(simulation::Node* node, core::visual::VisualModel* v);
    void processMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* m);
    Result processNodeTopDown(simulation::Node* node) override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override { return true; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "instrument"; }
    const char* getClassName() const override { return "TransformationVisitor"; }

protected:
    Vec3 translation;
    Vec3 rotation;
    Vec3 scale;
};

} // namespace sofa::simulation
