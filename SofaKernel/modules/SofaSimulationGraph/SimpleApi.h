/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaSimulationGraph/graph.h>
#include <string>
#include <sstream>
#include <map>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{
namespace simpleapi
{

using sofa::core::objectmodel::BaseObject ;
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;

void SOFA_SIMULATION_GRAPH_API importPlugin(const std::string& name) ;

Simulation::SPtr SOFA_SIMULATION_GRAPH_API createSimulation(const std::string& type="DAG") ;

Node::SPtr SOFA_SIMULATION_GRAPH_API createRootNode( Simulation::SPtr, const std::string& name,
    const std::map<std::string, std::string>& params = std::map<std::string, std::string>{} );

BaseObject::SPtr SOFA_SIMULATION_GRAPH_API createObject( Node::SPtr parent, const std::string& type,
    const std::map<std::string, std::string>& params = std::map<std::string, std::string>{} );

Node::SPtr SOFA_SIMULATION_GRAPH_API createChild( Node::SPtr& node, const std::string& name,
    const std::map<std::string, std::string>& params = std::map<std::string, std::string>{} );

void SOFA_SIMULATION_GRAPH_API dumpScene(Node::SPtr root) ;

template<class T>
std::string str(const T& t)
{
    std::stringstream s;
    s << t;
    return s.str() ;
}
} /// simpleapi


namespace simpleapi
{
namespace components {

namespace BaseObject
{
    static const std::string aobjectname {"BaseObject"} ;
    namespace data{
        static const std::string name {"name"} ;
    }
};

namespace MechanicalObject
{
    static const std::string objectname {"MechanicalObject"} ;
    namespace data{
        using namespace BaseObject::data ;
        static const std::string position {"position"} ;
    }
}

namespace VisualModel
{
    static const std::string objectname {"VisualModel"} ;

    namespace data {
        using namespace BaseObject::data ;
        static const std::string filename {"filename"} ;
    }
}

}

namespace meca   { using namespace simpleapi::components::MechanicalObject ; }
namespace visual { using namespace simpleapi::components::VisualModel ; }

}


} /// sofa

#endif /// SOFA_SIMPLEAPI
