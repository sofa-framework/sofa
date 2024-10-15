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
#include <sofa/component/linearsolver/direct/EigenSolverFactory.h>
#include <sofa/component/linearsolver/direct/init.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>

namespace sofa::component::linearsystem
{
    extern void registerMatrixLinearSystemBTDMatrix(sofa::core::ObjectFactory* factory);
    extern void registerTypedMatrixLinearSystemBTDMatrix(sofa::core::ObjectFactory* factory);
}

namespace sofa::component::linearsolver::direct
{
   
extern void registerAsyncSparseLDLSolver(sofa::core::ObjectFactory* factory);
extern void registerBTDLinearSolver(sofa::core::ObjectFactory* factory);
extern void registerCholeskySolver(sofa::core::ObjectFactory* factory);
extern void registerEigenSimplicialLDLT(sofa::core::ObjectFactory* factory);
extern void registerEigenSimplicialLLT(sofa::core::ObjectFactory* factory);
extern void registerEigenSparseLU(sofa::core::ObjectFactory* factory);
extern void registerEigenSparseQR(sofa::core::ObjectFactory* factory);
extern void registerPrecomputedLinearSolver(sofa::core::ObjectFactory* factory);
extern void registerSparseLDLSolver(sofa::core::ObjectFactory* factory);
extern void registerSVDLinearSolver(sofa::core::ObjectFactory* factory);

extern "C" {
    SOFA_EXPORT_DYNAMIC_LIBRARY void initExternalModule();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleName();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleVersion();
    SOFA_EXPORT_DYNAMIC_LIBRARY void registerObjects(sofa::core::ObjectFactory* factory);
}

void initExternalModule()
{
    init();
}

const char* getModuleName()
{
    return MODULE_NAME;
}

const char* getModuleVersion()
{
    return MODULE_VERSION;
}

template<class EigenSolverFactory, class Scalar>
void registerOrderingMethods()
{
    EigenSolverFactory::template registerSolver<Eigen::AMDOrdering<int>, Scalar >("AMD");
    EigenSolverFactory::template registerSolver<Eigen::COLAMDOrdering<int>, Scalar >("COLAMD");
    EigenSolverFactory::template registerSolver<Eigen::NaturalOrdering<int>, Scalar >("Natural");
}

template<class Scalar>
void registerOrderingMethods()
{
    registerOrderingMethods<MainSimplicialLDLTFactory, Scalar>();
    registerOrderingMethods<MainSimplicialLLTFactory, Scalar>();
    registerOrderingMethods<MainLUFactory, Scalar>();
    registerOrderingMethods<MainQRFactory, Scalar>();
}

void registerObjects(sofa::core::ObjectFactory* factory)
{
    registerAsyncSparseLDLSolver(factory);
    registerBTDLinearSolver(factory);
    registerCholeskySolver(factory);
    registerEigenSimplicialLDLT(factory);
    registerEigenSimplicialLLT(factory);
    registerEigenSparseLU(factory);
    registerEigenSparseQR(factory);
    linearsystem::registerMatrixLinearSystemBTDMatrix(factory);
    registerPrecomputedLinearSolver(factory);
    registerSparseLDLSolver(factory);
    registerSVDLinearSolver(factory);
    linearsystem::registerTypedMatrixLinearSystemBTDMatrix(factory);
}

void init()
{
    static bool first = true;
    if (first)
    {
        // make sure that this plugin is registered into the PluginManager
        sofa::helper::system::PluginManager::getInstance().registerPlugin(MODULE_NAME);

        registerOrderingMethods<float>();
        registerOrderingMethods<double>();

        first = false;
    }
}

} // namespace sofa::component::linearsolver::direct
