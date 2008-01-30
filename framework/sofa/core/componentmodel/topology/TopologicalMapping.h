/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_TOPOLOGICALMAPPING_H
#define SOFA_CORE_TOPOLOGICALMAPPING_H

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/BehaviorModel.h>

#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace topology
{

using namespace sofa::core;
using namespace sofa::defaulttype;

/**
 *  \brief This Interface is a new kind of Mapping, called TopologicalMapping, which converts an INPUT TOPOLOGY to an OUTPUT TOPOLOGY (both topologies are of type BaseTopology)
 *
 * It first initializes the mesh of the output topology from the mesh of the input topology,
 * and it creates the two Index Maps that maintain the correspondence between the indices of their common elements.
 *
 * Then, at each propagation of topological changes, it translates the topological change events that are propagated from the INPUT topology
 * into specific actions that call element adding or element removal methods on the OUTPUT topology, and it updates the Index Maps.
 *
 * So, at each time step, the geometrical and adjacency information are consistent in both topologies.
 *
 */
class TopologicalMapping : public virtual objectmodel::BaseObject
{
public:
    virtual ~TopologicalMapping() { }

    /// Accessor to the INPUT topology of the TopologicalMapping :
    virtual objectmodel::BaseObject* getFrom() = 0;

    /// Accessor to the OUTPUT topology of the TopologicalMapping :
    virtual objectmodel::BaseObject* getTo() = 0;

    /// Method called at each topological changes propagation which comes from the INPUT topology to adapt the OUTPUT topology :
    virtual void updateTopologicalMapping() = 0;

    /// Accessor to index maps :
    const std::map<unsigned int, unsigned int>& getGlob2LocMap() { return Glob2LocMap;}
    const sofa::helper::vector<unsigned int>& getLoc2GlobVec() { return Loc2GlobVec;}

protected:

    // Two index maps :

    // Array which gives for each index (local index) of an element in the OUTPUT topology
    // the corresponding index (global index) of the same element in the INPUT topology :
    sofa::helper::vector<unsigned int> Loc2GlobVec;

    // Map which gives for each index (global index) of an element in the INPUT topology
    // the corresponding index (local index) of the same element in the OUTPUT topology :
    std::map<unsigned int, unsigned int> Glob2LocMap;


};

} // namespace topology

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
