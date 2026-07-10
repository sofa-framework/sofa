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
#include <SofaCHOLMOD/init.h>

#include <sofa/helper/system/PluginManager.h>
#include <sofa/core/ObjectFactory.h>

#include <SofaCHOLMOD/EigenCholmodSupernodalLLT.h>
#include <SofaCHOLMOD/CholmodSolverProxy.h>
#include <Eigen/CholmodSupport>

namespace sofacholmod
{

// Definition of MainCholmodSupernodalLLTFactory::registerSolver.
// Kept here (not in the header) to avoid pulling <Eigen/CholmodSupport> everywhere.
template<typename OrderingMethodType, class ScalarType>
void MainCholmodSupernodalLLTFactory::registerSolver(const std::string& orderingMethodName)
{
    std::lock_guard lock(s_mutex);
    // CHOLMOD uses its own internal ordering; the OrderingMethodType is unused.
    // Register our own proxy (not the generic EigenSolverWrapper) so that the
    // solver can access the raw CHOLMOD factor for the optimized
    // addJMInvJtLocal() compliance-matrix computation.
    getFactory().registerProxyType<CholmodSolverProxy, ScalarType>(orderingMethodName);
}


template<class EigenSolverFactory, class Scalar>
void registerOrderingMethods()
{
    EigenSolverFactory::template registerSolver<Eigen::AMDOrdering<int>, Scalar >("AMD");
    EigenSolverFactory::template registerSolver<Eigen::COLAMDOrdering<int>, Scalar >("COLAMD");
    EigenSolverFactory::template registerSolver<Eigen::NaturalOrdering<int>, Scalar >("Natural");
}

void registerCholmodOrderingMethods()
{
    // CHOLMOD only supports double precision
    registerOrderingMethods<MainCholmodSupernodalLLTFactory, double>();
}

extern void registerEigenCholmodSupernodalLLT(sofa::core::ObjectFactory* factory);

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

void registerObjects(sofa::core::ObjectFactory* factory)
{
    registerEigenCholmodSupernodalLLT(factory);
}

void init()
{
    static bool first = true;
    if (first)
    {
        // make sure that this plugin is registered into the PluginManager
        sofa::helper::system::PluginManager::getInstance().registerPlugin(MODULE_NAME);

        registerCholmodOrderingMethods();

        first = false;
    }
}

} // namespace sofacholmod
