/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_SIMULATION_NODE_FWD_H
#define SOFA_SIMULATION_NODE_FWD_H

#include <sofa/core/sptr.h>
#include <sofa/core/objectmodel/Link.h>

namespace sofa {
    namespace simulation {
        class Node ;
        using NodeSPtr = sofa::core::sptr<Node>;


    }

    namespace core {
        namespace objectmodel {
            class Base;
            class BaseData;

        template<>
        class LinkTraitsPtrCasts<sofa::simulation::Node>
        {
        public:
            static sofa::core::objectmodel::Base* getBase(sofa::simulation::Node* b) { return reinterpret_cast<sofa::core::objectmodel::Base*>(b) ; }
            static sofa::core::objectmodel::BaseData* getData(sofa::simulation::Node* /*b*/) { return NULL; }
        };

        }
    }
}





#endif // SOFA_SIMULATION_NODE_FWD
