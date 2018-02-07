/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_SIMULATION_INSTRUMENTACTION_H
#define SOFA_SIMULATION_INSTRUMENTACTION_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/simulation/Visitor.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/core/visual/VisualModel.h>

namespace sofa
{

namespace simulation
{


class SOFA_SIMULATION_COMMON_API TransformationVisitor : public Visitor
{
public:
    typedef sofa::defaulttype::Vector3 Vector3;

    TransformationVisitor(const sofa::core::ExecParams* params)
        : Visitor(params)
    {
        translation = Vector3();
        rotation = Vector3();
        scale = Vector3(1.0,1.0,1.0);
    }

    void setTranslation(SReal dx, SReal dy, SReal dz) { translation = Vector3(dx,dy,dz);}
    void setRotation(SReal rx, SReal ry, SReal rz) {    rotation=Vector3(rx,ry,rz);	}
    void setScale(SReal sx, SReal sy, SReal sz) {scale=Vector3(sx,sy,sz);}


    void processVisualModel(simulation::Node* node, core::visual::VisualModel* v);
    void processMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* m);
    virtual Result processNodeTopDown(simulation::Node* node);

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const { return true; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const { return "instrument"; }
    virtual const char* getClassName() const { return "TransformationVisitor"; }

protected:
    Vector3 translation;
    Vector3 rotation;
    Vector3 scale;
};

} // namespace simulation

} // namespace sofa

#endif
