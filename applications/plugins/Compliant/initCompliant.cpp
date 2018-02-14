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
#include <Compliant/config.h>
#include "misc/CompliantSolverMerger.h"
#include "contact/CompliantContact.h"

#ifdef SOFA_HAVE_SOFAPYTHON
#include <SofaPython/PythonCommon.h>
#include <SofaPython/PythonMacros.h>
#include <SofaPython/PythonFactory.h>
extern PyMethodDef _CompliantModuleMethods[]; // functions of the _Compliant python module
#include "python/Binding_AssembledSystem.h"
#endif


namespace sofa
{


namespace component
{

//Here are just several convenient functions to help user to know what contains the plugin

extern "C" {
    SOFA_Compliant_API void initExternalModule();
    SOFA_Compliant_API const char* getModuleName();
    SOFA_Compliant_API const char* getModuleVersion();
    SOFA_Compliant_API const char* getModuleLicense();
    SOFA_Compliant_API const char* getModuleDescription();
    SOFA_Compliant_API const char* getModuleComponentList();
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        first = false;

        component::collision::CompliantSolverMerger::add();

        // previous Eigen versions have a critical bug (v.noalias()+=w does not work in some situations)
        static_assert( EIGEN_WORLD_VERSION>=3 && EIGEN_MAJOR_VERSION>=2 && EIGEN_MINOR_VERSION>=5, "" );

#ifdef SOFA_HAVE_SOFAPYTHON
        // adding _Compliant python module
        if( PythonFactory::s_sofaPythonModule ) // add the module only if the Sofa module exists (SofaPython is loaded)
        {
            simulation::PythonEnvironment::gil lock(__func__);
            static PyObject *s__CompliantPythonModule = SP_INIT_MODULE(_Compliant);

            // adding more bindings to the _Compliant module
            SP_ADD_CLASS( s__CompliantPythonModule, AssembledSystem );
        }
#endif

    }
}

const char* getModuleName()
{
    return "Compliant";
}

const char* getModuleVersion()
{
    return "0";
}

const char* getModuleLicense()
{
    return "LGPL";
}


const char* getModuleDescription()
{
    return "Simulation of deformable object using a formulation similar to the KKT system for hard constraints, regularized using a compliance matrix";
}

const char* getModuleComponentList()
{
    return ""; /// @TODO
}

}

// Ensure that our abstract factories do the registration and avoid symbol stripping on agressive
// compilers like the ones found on consoles.
SOFA_Compliant_API void initCompliant()
{
	component::collision::registerContactClasses();
}

}


//SOFA_LINK_CLASS(MyMappingPendulumInPlane)

