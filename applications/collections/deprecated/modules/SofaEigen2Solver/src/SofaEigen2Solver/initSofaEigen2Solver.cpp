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
#include <SofaEigen2Solver/initSofaEigen2Solver.h>

#include <sofa/helper/logging/Messaging.h>

namespace sofa::component
{

void initSofaEigen2Solver()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }

    msg_warning("SofaEigen2Solver") << "SofaEigen2Solver is deprecated; Eigen classes are now located in Sofa.LinearAlgebra and SVDLinearSolver in SofaDenseSolver."
        << "You can remove SofaEigen2Solver from your scene, and if using SVDLinearSolver, please load SofaDenseSolver instead.";
}

extern "C" {
    SOFA_SOFAEIGEN2SOLVER_API void initExternalModule();
    SOFA_SOFAEIGEN2SOLVER_API const char* getModuleName();
    SOFA_SOFAEIGEN2SOLVER_API const char* getModuleVersion();
    SOFA_SOFAEIGEN2SOLVER_API const char* getModuleLicense();
    SOFA_SOFAEIGEN2SOLVER_API const char* getModuleDescription();
    SOFA_SOFAEIGEN2SOLVER_API const char* getModuleComponentList();
}

void initExternalModule()
{
    initSofaEigen2Solver();
}

const char* getModuleName()
{
    return sofa_tostring(SOFA_TARGET);
}

const char* getModuleVersion()
{
    return sofa_tostring(SOFAEIGEN2SOLVER_VERSION);
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "This plugin contains contains features about Eigen2 Solver.";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    return "";
}

} // namespace sofa::component
