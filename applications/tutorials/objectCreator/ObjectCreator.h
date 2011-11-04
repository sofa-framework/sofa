/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef SOFA_OBJECTCREATOR_H
#define SOFA_OBJECTCREATOR_H

#include <string>
#include <sofa/simulation/common/Node.h>
#include <sofa/component/loader/MeshObjLoader.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>

namespace sofa
{

class ObjectCreator
{
public:
    static simulation::Node *CreateRootWithCollisionPipeline(const std::string &simulationType, const std::string &responseType=std::string("default"));
    static simulation::Node *CreateEulerSolverNode(simulation::Node* parent, const std::string& name, const std::string &integrationScheme=std::string("Implicit"));


    static simulation::Node *CreateObstacle(simulation::Node* parent, const std::string &filenameCollision, const std::string filenameVisual, const std::string& color,
            const Deriv3& translation=Deriv3(), const Deriv3 &rotation=Deriv3());

    //Create a collision node using Barycentric Mapping, using a 3d model specified by filename.
    //elements is a vector of type of collision models (Triangle, Line, Point, Sphere)
    //an initial transformation can be performed
    static simulation::Node *CreateCollisionNodeVec3(simulation::Node* parent, MechanicalObject3* dof, const std::string &filename, const std::vector<std::string> &elements,
            const Deriv3& translation=Deriv3(), const Deriv3 &rotation=Deriv3());
    static simulation::Node *CreateVisualNodeVec3(simulation::Node* parent, MechanicalObject3* dof,  const std::string &filename, const std::string& color,
            const Deriv3& translation=Deriv3(), const Deriv3 &rotation=Deriv3());



    //Create a collision node using Rigid Mapping, using a 3d model specified by filename.
    //elements is a vector of type of collision models (Triangle, Line, Point, Sphere)
    //an initial transformation can be performed
    static simulation::Node *CreateCollisionNodeRigid(simulation::Node* parent, MechanicalObjectRigid3* dofRigid,  const std::string &filename, const std::vector<std::string> &elements,
            const Deriv3& translation=Deriv3(), const Deriv3 &rotation=Deriv3());
    static simulation::Node *CreateVisualNodeRigid(simulation::Node* parent, MechanicalObjectRigid3* dofRigid,  const std::string &filename, const std::string& color,
            const Deriv3& translation=Deriv3(), const Deriv3 &rotation=Deriv3());

private:
    static void AddCollisionModels(simulation::Node *CollisionNode, const std::vector<std::string> &elements);
};
}

#endif
