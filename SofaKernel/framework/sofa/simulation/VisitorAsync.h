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
#ifndef Sofa_VisitorAsync_h__
#define Sofa_VisitorAsync_h__

#include <Multithreading/config.h>

#include <sofa/simulation/Visitor.h>
#include <MultiThreading/src/Tasks.h>

namespace sofa
{

    namespace simulation
    {

        /**
        * Used to execute async tasks
        */

        class SOFA_SIMULATION_CORE_API VisitorAsync : public Visitor
        {
        public:
            VisitorAsync(const sofa::core::ExecParams* params, Task::Status* status)
                : Visitor(params)
                , _status(status)
            {}

            const Task::Status* getStatus() { return _status; }

        protected:

            Task::Status* _status;

        };


    } // namespace simulation

} // namespace sofa

#endif // Sofa_VisitorAsync_h__
