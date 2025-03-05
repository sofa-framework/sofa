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

#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa::core::topology
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
class SOFA_CORE_API TopologicalMapping : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(TopologicalMapping, objectmodel::BaseObject);

    /// Input Topology
    using In = BaseMeshTopology;
    /// Output Topology
    using Out = BaseMeshTopology;

    using ElementType = sofa::geometry::ElementType;

    using Index = sofa::Index;

protected:
    TopologicalMapping();

    ~TopologicalMapping() override { }

public:
    /// Specify the input and output models.
    virtual void setTopologies(In* from, Out* to);

    /// Set the path to the objects mapped in the scene graph
    void setPathInputObject(const std::string &o) {fromModel.setPath(o);}
    void setPathOutputObject(const std::string &o) {toModel.setPath(o);}

    /// Accessor to the INPUT topology of the TopologicalMapping :
    In* getFrom() {return fromModel.get();}

    /// Accessor to the OUTPUT topology of the TopologicalMapping :
    Out* getTo() {return toModel.get();}

    /// Method called at each topological changes propagation which comes from the INPUT topology to adapt the OUTPUT topology :
    virtual void updateTopologicalMappingTopDown() = 0;

    /// Method called at each topological changes propagation which comes from the OUTPUT topology to adapt the INPUT topology :
    virtual void updateTopologicalMappingBottomUp() {}

    /// Return true if this mapping is able to propagate topological changes from input to output model
    virtual bool propagateFromInputToOutputModel() { return true; }

    /// Return true if this mapping is able to propagate topological changes from output to input model
    virtual bool propagateFromOutputToInputModel() { return false; }

    /// return true if the output topology subdivide the input one. (the topology uses the Loc2GlobVec/Glob2LocMap/In2OutMap structs and share the same DOFs)
    virtual bool isTheOutputTopologySubdividingTheInputOne() { return true; }

    /// Accessor to index maps :
    const std::map<Index, Index>& getGlob2LocMap() { return Glob2LocMap;}

    virtual Index getGlobIndex(Index ind);

    virtual Index getFromIndex(Index ind);

    void dumpGlob2LocMap();

    void dumpLoc2GlobVec();

    /// Method to check the topology mapping maps regarding the upper topology
    virtual bool checkTopologies() {return false;}

    /** return all the from indices in the 'In' topology corresponding to the index in the 'Out' topology.
    *   This function is used instead of  the previous one when the function isTheOutputTopologySubdividingTheInputOne() returns false.
    */
    virtual void getFromIndex( type::vector<Index>& /*fromIndices*/, const Index /*toIndex*/) const {}

    const std::map<Index, sofa::type::vector<Index> >& getIn2OutMap() { return In2OutMap;}

    /// Pre-construction check method called by ObjectFactory.
    ///
    /// This implementation reads the "input" and "output" attributes and checks
    /// that the corresponding objects exist, and are not the same object.
    template<class T>
    static bool canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
    {
        In* stin = nullptr;
        Out* stout = nullptr;

        std::string inPath, outPath;

        if (arg->getAttribute("input"))
            inPath = arg->getAttribute("input");
        else
            inPath = "@../";

        context->findLinkDest(stin, inPath, nullptr);

        if (arg->getAttribute("output"))
            outPath = arg->getAttribute("output");
        else
            outPath = "@./";

        context->findLinkDest(stout, outPath, nullptr);

        if (stin == nullptr)
        {
            arg->logError("Data attribute 'input' does not point to a valid mesh topology and none can be found in the parent node context.");
            return false;
        }

        if (stout == nullptr)
        {
            arg->logError("Data attribute 'output' does not point to a valid mesh topology and none can be found in the current node context.");
            return false;
        }

        if (dynamic_cast<BaseObject*>(stin) == dynamic_cast<BaseObject*>(stout))
        {
            // we should refuse to create mappings with the same input and output model, which may happen if a State object is missing in the child node
            arg->logError("Both the input mesh and the output mesh points to the same mesh topology ('"+stin->getName()+"').");
            return false;
        }

        return BaseObject::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    ///
    /// This implementation read the object1 and object2 attributes to
    /// find the input and output topologies of this mapping.
    template<class T>
    static typename T::SPtr create (T*, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
    {
        typename T::SPtr obj = sofa::core::objectmodel::New<T>();

        if (context)
            context->addObject(obj);

        if (arg)
        {
            std::string inPath, outPath;
            if (arg->getAttribute("input"))
                inPath = arg->getAttribute("input");
            else
                inPath = "@../";

            if (arg->getAttribute("output"))
                outPath = arg->getAttribute("output");
            else
                outPath = "@./";

            obj->fromModel.setPath( inPath );
            obj->toModel.setPath( outPath );

            obj->parse(arg);
        }

        return obj;
    }

protected:
    [[nodiscard]] bool checkTopologyInputTypes();

public:
    /// Input source BaseTopology
    SingleLink<TopologicalMapping, In, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> fromModel;

    /// Output target BaseTopology
    SingleLink<TopologicalMapping, Out, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> toModel;

    // Array which gives for each index (local index) of an element in the OUTPUT topology
    // the corresponding index (global index) of the same element in the INPUT topology :
    Data <sofa::type::vector<Index> > Loc2GlobDataVec;

protected:
    // Map which gives for each index (global index) of an element in the INPUT topology
    // the corresponding index (local index) of the same element in the OUTPUT topology :
    std::map<Index, Index> Glob2LocMap;

    std::map<Index, sofa::type::vector<Index> > In2OutMap;

    ElementType m_inputType = geometry::ElementType::UNKNOWN;
    ElementType m_outputType = geometry::ElementType::UNKNOWN;
};
} // namespace sofa::core::topology

