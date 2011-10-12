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
#ifndef SOFA_CORE_TOPOLOGY_TOPOLOGICALMAPPING_H
#define SOFA_CORE_TOPOLOGY_TOPOLOGICALMAPPING_H

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/ObjectRef.h>

namespace sofa
{

namespace core
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
    SOFA_ABSTRACT_CLASS(TopologicalMapping, objectmodel::BaseObject);

    /// Name of the Input Topology
    objectmodel::DataObjectRef m_inputTopology;
    /// Name of the Output Topology
    objectmodel::DataObjectRef m_outputTopology;

    /// Input Topology
    typedef BaseMeshTopology In;
    /// Output Topology
    typedef BaseMeshTopology Out;

    TopologicalMapping(In* from, Out* to)
        : m_inputTopology(initData(&m_inputTopology, "input", "Input topology to map"))
        , m_outputTopology(initData(&m_outputTopology, "output", "Output topology to map"))
        , fromModel(from), toModel(to)

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

    /// Pre-construction check method called by ObjectFactory.
    ///
    /// This implementation read the object1 and object2 attributes and check
    /// if they are compatible with the input and output topology types of this
    /// mapping.
    template<class T>
    static bool canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
    {
        BaseMeshTopology* topoIn = NULL;
        BaseMeshTopology* topoOut = NULL;

#ifndef SOFA_DEPRECATE_OLD_API
        ////Deprecated check
        Base* bobjInput = NULL;
        Base* bobjOutput = NULL;

        //Input
        if (arg->getAttribute("object1",NULL) == NULL && arg->getAttribute("input",NULL) == NULL)
            bobjInput = arg->findObject("../..");

        if (arg->getAttribute("object1",NULL) != NULL)
            bobjInput = sofa::core::objectmodel::ObjectRef::parseFromXMLPath("object1", arg);

        if (BaseObject* bo = dynamic_cast< BaseObject* >(bobjInput))
            topoIn = dynamic_cast< BaseMeshTopology* >(bo);

        else if (core::objectmodel::BaseContext* bc = dynamic_cast< core::objectmodel::BaseContext* >(bobjInput))
            bc->get(topoIn);

        //Output
        if (arg->getAttribute("object2",NULL) == NULL && arg->getAttribute("output",NULL) == NULL)
            bobjOutput = arg->findObject("..");

        if (arg->getAttribute("object2",NULL) != NULL)
            bobjOutput = sofa::core::objectmodel::ObjectRef::parseFromXMLPath("object2", arg);

        if (BaseObject* bo = dynamic_cast< BaseObject* >(bobjOutput))
            topoOut = dynamic_cast< BaseMeshTopology* >(bo);

        else if (core::objectmodel::BaseContext* bc = dynamic_cast< core::objectmodel::BaseContext* >(bobjOutput))
            bc->get(topoOut);
        /////

        if (topoIn == NULL || topoOut == NULL)
#endif // SOFA_DEPRECATE_OLD_API
        {
            topoIn = sofa::core::objectmodel::ObjectRef::parse< BaseMeshTopology >("input", arg);
            topoOut = sofa::core::objectmodel::ObjectRef::parse< BaseMeshTopology >("output", arg);
        }

        if (topoIn == NULL)
        {
            //context->serr << "Cannot create "<<className(obj)<<" as object1 is missing or invalid." << context->sendl;
            return false;
        }

        if (topoOut == NULL)
        {
            //context->serr << "Cannot create "<<className(obj)<<" as object2 is missing or invalid." << context->sendl;
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
        typename T::SPtr obj;

        BaseMeshTopology* topoIn=NULL;
        BaseMeshTopology* topoOut=NULL;

#ifndef SOFA_DEPRECATE_OLD_API
        {
            ////Deprecated check
            std::string object1Path;
            std::string object2Path;

            Base* bobjInput = NULL;
            Base* bobjOutput = NULL;

            if( arg != NULL )
            {
                //Input
                if(arg->getAttribute("object1",NULL) == NULL && arg->getAttribute("input",NULL) == NULL)
                {
                    object1Path = "..";
                    //context->serr << "Deprecated use of implicit value for input" << context->sendl;
                    //context->serr << "Use now : input=\"@" << object1Path << "\" "<< context->sendl;
                    bobjInput = arg->findObject("../..");
                }

                if(arg->getAttribute("object1",NULL) != NULL)
                {
                    object1Path = sofa::core::objectmodel::ObjectRef::convertFromXMLPathToSofaScenePath(arg->getAttribute("object1",NULL));
                    //context->serr << "Deprecated use of attribute " << "object1" << context->sendl;
                    //context->serr << "Use now : input=\"@"
                    //              << object1Path
                    //              << "\""<< context->sendl;
                    bobjInput = sofa::core::objectmodel::ObjectRef::parseFromXMLPath("object1", arg);
                }

                if (BaseObject* bo = dynamic_cast< BaseObject* >(bobjInput))
                {
                    topoIn = dynamic_cast< BaseMeshTopology* >(bo);
                }
                else if (core::objectmodel::BaseContext* bc = dynamic_cast< core::objectmodel::BaseContext* >(bobjInput))
                {
                    bc->get(topoIn);
                }

                //Output
                if(arg->getAttribute("object2",NULL) == NULL && arg->getAttribute("output",NULL) == NULL)
                {
                    object2Path = ".";
                    //context->serr << "Deprecated use of implicit value for output" << context->sendl;
                    //context->serr << "Use now : output=\"@" << object2Path << "\" "<< context->sendl;
                    bobjOutput = arg->findObject("..");
                }

                if(arg->getAttribute("object2",NULL) != NULL)
                {
                    object2Path = sofa::core::objectmodel::ObjectRef::convertFromXMLPathToSofaScenePath(arg->getAttribute("object2",NULL));
                    //context->serr << "Deprecated use of attribute " << "object2" << context->sendl;
                    //context->serr << "Use now : output=\"@"
                    //              << object2Path
                    //              << "\""<< context->sendl;
                    bobjOutput = sofa::core::objectmodel::ObjectRef::parseFromXMLPath("object2", arg);
                }

                if (BaseObject* bo = dynamic_cast< BaseObject* >(bobjOutput))
                {
                    topoOut = dynamic_cast< BaseMeshTopology* >(bo);
                }
                else if (core::objectmodel::BaseContext* bc = dynamic_cast< core::objectmodel::BaseContext* >(bobjOutput))
                {
                    bc->get(topoOut);
                }

                /////
            }

            if(topoIn == NULL && topoOut == NULL)
#endif // SOFA_DEPRECATE_OLD_API
            {
                if(arg)
                {
                    topoIn = sofa::core::objectmodel::ObjectRef::parse< BaseMeshTopology >("input", arg);
                    topoOut = sofa::core::objectmodel::ObjectRef::parse< BaseMeshTopology >("output", arg);
                }
            }

            obj = sofa::core::objectmodel::New<T>((arg?topoIn:NULL), (arg?topoOut:NULL));

#ifndef SOFA_DEPRECATE_OLD_API
            if (!object1Path.empty())
                obj->m_inputTopology.setValue( object1Path );
            if (!object2Path.empty())
                obj->m_outputTopology.setValue( object2Path );
#endif // SOFA_DEPRECATE_OLD_API
        }

        if (context)
            context->addObject(obj);

        if (arg)
            obj->parse(arg);

        return obj;
    }

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

} // namespace core

} // namespace sofa

#endif
