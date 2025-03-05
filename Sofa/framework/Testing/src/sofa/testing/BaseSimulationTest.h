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

#include <sofa/testing/config.h>

#include <sofa/testing/BaseTest.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>

namespace sofa::testing
{
using sofa::simulation::Node ;
using sofa::simulation::Simulation ;

class SOFA_TESTING_API BaseSimulationTest : public virtual BaseTest
{
public:
    BaseSimulationTest() ;

    SOFA_ATTRIBUTE_DEPRECATED__TESTING_IMPORT_PLUGIN()
    bool importPlugin(const std::string& name) ;

    class SOFA_TESTING_API SceneInstance
    {
    public:
         SceneInstance(const std::string& rootname="root") ;
         SceneInstance(const std::string& type, const std::string& memory) ;
         ~SceneInstance() ;

        /// Create a new scene instance from the content of the filename using the factory.
        static SceneInstance LoadFromFile(const std::string& filename) ;

        Node::SPtr root ;

        void initScene() ;
        void simulate(const double timestep) ;
        void loadSceneFile(const std::string& filename);
    } ;
};

} // namespace sofa::testing
