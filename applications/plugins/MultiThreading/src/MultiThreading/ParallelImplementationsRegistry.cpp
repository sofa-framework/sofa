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
#include <MultiThreading/ParallelImplementationsRegistry.h>
#include <sofa/helper/logging/Messaging.h>

namespace multithreading
{

sofa::type::vector<ParallelImplementationsRegistry::Implementation> ParallelImplementationsRegistry::s_implementations;

bool ParallelImplementationsRegistry::addEquivalentImplementations(
    const std::string& sequentialImplementation, const std::string& parallelImplementation)
{
    const Implementation implementation{sequentialImplementation, parallelImplementation};

    const auto it = findParallelImplementationImpl(sequentialImplementation);

    if (it == s_implementations.end())
    {
        s_implementations.push_back(implementation);
        return true;
    }

    if (parallelImplementation != it->parallel)
    {
        msg_error("ParallelImplementationsRegistry")
            << "Trying to register the sequential implementation '"
            << sequentialImplementation << "' with the parallel implementation ''"
            << parallelImplementation << "' but it has already been registered with the parallel implementation: '"
            << it->parallel << "'";
    }
    else
    {
        msg_warning("ParallelImplementationsRegistry")
            << "The sequential implementation '" << sequentialImplementation << "' has already "
            << "been registered to the parallel implementation '" << parallelImplementation << "'";
    }
    return false;
}

std::string ParallelImplementationsRegistry::findParallelImplementation(
    const std::string& sequentialImplementation)
{
    const auto it = findParallelImplementationImpl(sequentialImplementation);

    if (it != s_implementations.end())
    {
        return it->parallel;
    }
    return {};
}

const sofa::type::vector<ParallelImplementationsRegistry::Implementation>&
ParallelImplementationsRegistry::getImplementations()
{
    return s_implementations;
}

sofa::type::vector<ParallelImplementationsRegistry::Implementation>::const_iterator
ParallelImplementationsRegistry::findParallelImplementationImpl(
    const std::string& sequentialImplementation)
{
    return std::find_if(s_implementations.begin(), s_implementations.end(),
        [&sequentialImplementation](const Implementation& i)
        {
            return i.sequential == sequentialImplementation;
        });
}
}
