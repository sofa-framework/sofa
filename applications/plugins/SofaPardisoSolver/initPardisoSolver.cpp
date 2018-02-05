/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <sofa/helper/system/config.h>
#include <SofaPardisoSolver/initPardisoSolver.h>
#include <stdio.h>

namespace sofa
{

namespace component
{


    //Here are just several convenient functions to help user to know what contains the plugin

    extern "C" {
                    SOFA_PARDISO_SOLVER_API void initExternalModule();
                    SOFA_PARDISO_SOLVER_API const char* getModuleName();
                    SOFA_PARDISO_SOLVER_API const char* getModuleVersion();
                    SOFA_PARDISO_SOLVER_API const char* getModuleLicense();
                    SOFA_PARDISO_SOLVER_API const char* getModuleDescription();
                    SOFA_PARDISO_SOLVER_API const char* getModuleComponentList();
    }

    void initExternalModule()
    {
        static bool first = true;
        if (first)
        {
            first = false;
        }
    }

    const char* getModuleName()
    {
        return "Plugin Pardiso";
    }

    const char* getModuleVersion()
    {
        return "test 1.0";
    }

    const char* getModuleLicense()
    {
        return "LGPL";
    }


    const char* getModuleDescription()
    {
        return "Solver using Pardiso library";
    }

    const char* getModuleComponentList()
    {
        return "TetrahedralCryoDiffusionForceField, MassCryo, PotentielInitialization, ExplicitBDFSolver, IsoSurface, TemperatureState, GridToGridEngine, RegularGridMass, RegularGridDiffusionForceField, RegularGridHeatGenerationForceField, FieldInitialization, NeedleConstraint, CylinderROI";
    }

    void initPardisoSolver()
    {
        static bool first = true;
        if (first)
        {
            first = false;
        }
    }

} // namespace component

} // namespace sofa

SOFA_LINK_CLASS(SparsePARDISOSolver)
