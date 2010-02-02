/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_COMPONENTMODEL_TOPOLOGY_TOPOLOGICALMAPPING_H
#define SOFA_CORE_COMPONENTMODEL_TOPOLOGY_TOPOLOGICALMAPPING_H

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace topology
{

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
    SOFA_CLASS(TopologicalMapping, objectmodel::BaseObject);

    /// Input Topology
    typedef BaseMeshTopology In;
    /// Output Topology
    typedef BaseMeshTopology Out;

    TopologicalMapping(In* from, Out* to)
        : fromModel(from), toModel(to)
    {}

    virtual ~TopologicalMapping() { }

    /// Specify the input and output topologies.
//	void setModels(In* from, Out* to)
    //{
    //	fromModel = from;
    //	toModel = to;
    //}

    /// Accessor to the INPUT topology of the TopologicalMapping :
    In* getFrom() {return fromModel;}

    /// Accessor to the OUTPUT topology of the TopologicalMapping :
    Out* getTo() {return toModel;}

    /// Method called at each topological changes propagation which comes from the INPUT topology to adapt the OUTPUT topology :
    virtual void updateTopologicalMappingTopDown() = 0;

    /// Method called at each topological changes propagation which comes from the OUTPUT topology to adapt the INPUT topology :
    virtual void updateTopologicalMappingBottomUp() {};

    /// Return true if this mapping is able to propagate topological changes from input to output model
    virtual bool propagateFromInputToOutputModel() { return true; }

    /// Return true if this mapping is able to propagate topological changes from output to input model
    virtual bool propagateFromOutputToInputModel() { return false; }

    /// return true if the output topology subdivide the input one. (the topology uses the Loc2GlobVec/Glob2LocMap/In2OutMap structs and share the same DOFs)
    virtual bool isTheOutputTopologySubdividingTheInputOne() { return true;}

    /// Accessor to index maps :
    const std::map<unsigned int, unsigned int>& getGlob2LocMap() { return Glob2LocMap;}
    //const sofa::helper::vector<unsigned int>& getLoc2GlobVec(){ return Loc2GlobVec.getValue();}

    Data <sofa::helper::vector<unsigned int> >& getLoc2GlobVec() {return Loc2GlobDataVec;}





    virtual unsigned int getGlobIndex(unsigned int ind)
    {
        if(ind< (Loc2GlobDataVec.getValue()).size())
        {
            return (Loc2GlobDataVec.getValue())[ind];
        }
        else
        {
            return 0;
        }
    }

    virtual unsigned int getFromIndex(unsigned int /*ind*/)
    {
        return 0;
    }

    /** return all the from indices in the 'In' topology corresponding to the index in the 'Out' topology.
    *   This function is used instead of  the previous one when the function isTheOutputTopologySubdividingTheInputOne() returns false.
    */
    virtual void getFromIndex( vector<unsigned int>& /*fromIndices*/, const unsigned int /*toIndex*/) const {}

    const std::map<unsigned int, sofa::helper::vector<unsigned int> >& getIn2OutMap() { return In2OutMap;}


protected:

    /// Input source BaseTopology
    In* fromModel;
    /// Output target BaseTopology
    Out* toModel;

    // Two index maps :

    // Array which gives for each index (local index) of an element in the OUTPUT topology
    // the corresponding index (global index) of the same element in the INPUT topology :
    Data <sofa::helper::vector <unsigned int> > Loc2GlobDataVec;

    // Map which gives for each index (global index) of an element in the INPUT topology
    // the corresponding index (local index) of the same element in the OUTPUT topology :
    std::map<unsigned int, unsigned int> Glob2LocMap;   //TODO put it in Data => Data allow map

    std::map<unsigned int, sofa::helper::vector<unsigned int> > In2OutMap;
};

} // namespace topology

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
