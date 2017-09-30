/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/******************************************************************************
 * Contributors:                                                              *
 *    - damien.marchal@univ-lille1.fr                                         *
 *****************************************************************************/
#ifndef SOFA_SIMPLEAPI_H
#define SOFA_SIMPLEAPI_H

#include <SceneCreator/config.h>
#include <string>
#include <sstream>
#include <map>

#include <sofa/simulation/Node.h>
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{
namespace simpleapi
{

using sofa::core::objectmodel::BaseObject ;
using sofa::simulation::Node ;

void importPlugin(const std::string& name) ;
BaseObject::SPtr SOFA_SCENECREATOR_API createObject(Node::SPtr parent, const std::string& type, const std::map<std::string, std::string>& params={}) ;
Node::SPtr SOFA_SCENECREATOR_API createChild(Node::SPtr& node, const std::string& name, const std::map<std::string, std::string>& params={}) ;

template<class T>
std::string str(const T& t)
{
    std::stringstream s;
    s << t;
    return s.str() ;
}

namespace components{
namespace visual
{
    const std::string VisualModel {"VisualModel"} ;
    const std::string OglShader {"OglShader"} ;
}
namespace mechanical
{
    const std::string MechanicalModel {"MechanicalModel" } ;
    const std::string TetrahedronFEMForceField {"TetrahedronFEMForceField" } ;
    const std::string StiffSpringForceField {"StiffSpringForceField" } ;
}
namespace collision
{
    const std::string PointModel   {"PointModel"} ;
    const std::string LineModel    {"LineModel"} ;
    const std::string SphereModel  {"SphereModel"} ;
    const std::string OBBModel     {"OBBModel"} ;
    const std::string CapsuleModel {"CapsuleModel"} ;
}
using visual::VisualModel ;
using visual::OglShader ;
using mechanical::MechanicalModel ;
using mechanical::TetrahedronFEMForceField ;
using mechanical::StiffSpringForceField ;
using collision::PointModel ;
using collision::LineModel ;
using collision::SphereModel ;
using collision::OBBModel ;
using collision::CapsuleModel ;

namespace args
{
    namespace VisualModel
    {
        const std::string name = "name";
        const std::string filename = "filename";
    }
    namespace MechanicalModel
    {
        const std::string name = "name";
        const std::string size = "size";
    }
}

} /// components

} /// simpleapi
} /// sofa

#endif /// SOFA_SIMPLEAPI
