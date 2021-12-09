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
#include <SofaNewmat/initSofaNewmat.h>


#include <SofaNewmat/NewMatMatrix.h>
#include <SofaNewmat/NewMatVector.h>
#include <SofaBaseLinearSolver/MatrixLinearSolver.inl>

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory;

namespace sofa::component
{

namespace linearsolver
{
    // template specialization on specific matrix types
    template class SOFA_SOFANEWMAT_API MatrixLinearSolver< NewMatMatrix, NewMatVector >;
    template class SOFA_SOFANEWMAT_API MatrixLinearSolver< NewMatSymmetricMatrix, NewMatVector >;
    template class SOFA_SOFANEWMAT_API MatrixLinearSolver< NewMatBandMatrix, NewMatVector >;
    template class SOFA_SOFANEWMAT_API MatrixLinearSolver< NewMatSymmetricBandMatrix, NewMatVector >;
} // namespace linearsolver



void initSofaNewmat()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

extern "C" {
    SOFA_SOFANEWMAT_API void initExternalModule();
    SOFA_SOFANEWMAT_API const char* getModuleName();
    SOFA_SOFANEWMAT_API const char* getModuleVersion();
    SOFA_SOFANEWMAT_API const char* getModuleLicense();
    SOFA_SOFANEWMAT_API const char* getModuleDescription();
    SOFA_SOFANEWMAT_API const char* getModuleComponentList();
}

void initExternalModule()
{
    initSofaNewmat();
}

const char* getModuleName()
{
    return sofa_tostring(SOFA_TARGET);
}

const char* getModuleVersion()
{
    return sofa_tostring(SOFANEWMAT_VERSION);
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "This plugin contains contains features using Newmat.";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    static std::string classes = ObjectFactory::getInstance()->listClassesFromTarget(sofa_tostring(SOFA_TARGET));
    return classes.c_str();
}

} // namespace sofa::component
