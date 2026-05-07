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

#include <sofa/core/ComponentLibrary.h>

namespace sofa::core
{


using Creator = sofa::core::ObjectFactory::BaseObjectCreator;

/**
 *  \brief An Generic Category of the Sofa Library
 *
 *  It contains all the components available for Sofa corresponding to a given category (force field, mass, mapping...)
 *  This Interface is used for the Modeler mainly.
 *
 */
class SOFA_CORE_API CategoryLibrary
{
public:
    typedef std::vector< ComponentLibrary* > VecComponent;
    typedef VecComponent::const_iterator VecComponentIterator;

    CategoryLibrary( const std::string &categoryName);
    virtual ~CategoryLibrary() {}

    virtual ComponentLibrary *addComponent(const std::string &componentName, ClassEntry::SPtr entry, const std::vector< std::string > &exampleFiles);
    virtual void endConstruction();

    const std::string  &getName()          const { return name;}
    const VecComponent &getComponents()    const {return components;}

    const ComponentLibrary *getComponent( const std::string &componentName) const;

    size_t getNumComponents() const {return components.size();}

    /** \brief Get the list of categories a class belongs to, based on its parent classes.
     *
     *  @param mclass the class the categorize
     *  @param outputVector the vector where to push the categories
     *
     *  The categories are: AnimationLoop, BehaviorModel,
     *  CollisionAlgorithm, CollisionAlgorithm, CollisionAlgorithm,
     *  CollisionModel, ConfigurationSetting, ConstraintSet,
     *  ConstraintSolver, ConstraintSolver, ContextObject, Controller,
     *  Engine, Exporter, ForceField, InteractionForceField, LinearSolver, LinearSystem,
     *  Loader, Mapping, Mass, MechanicalState, OdeSolver, OrderingMethod,
     *  ProjectiveConstraintSet, TopologicalMapping, Topology,
     *  TopologyObject, and VisualModel
     */
    static void getCategories(const sofa::core::objectmodel::BaseClass* mclass,
                              std::vector<std::string>& outputVector);
    static std::vector<std::string> getCategories();

protected:
    virtual ComponentLibrary *createComponent(const std::string &componentName, ClassEntry::SPtr entry, const std::vector< std::string > &exampleFiles) {return new ComponentLibrary(componentName, name, entry, exampleFiles);}

    std::string name;
    VecComponent components;
};

} // namespace sofa::core
