#ifndef SOFA_COMPONENT_TOPOLOGY_TETRA2TRIANGLETOPOLOGICALMAPPING_H
#define SOFA_COMPONENT_TOPOLOGY_TETRA2TRIANGLETOPOLOGICALMAPPING_H

#include <sofa/core/componentmodel/topology/TopologicalMapping.h>

#include <sofa/core/componentmodel/topology/Topology.h>

#include <sofa/component/topology/TriangleSetTopology.h>
#include <sofa/component/topology/TetrahedronSetTopology.h>

#include <sofa/defaulttype/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/core/BaseMapping.h>

namespace sofa
{

namespace component
{

namespace topology
{


using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;

using namespace sofa::component::topology;
using namespace sofa::core::componentmodel::topology;

using namespace sofa::core;

/**
 * This class, called Tetra2TriangleTopologicalMapping, is a specific implementation of the interface TopologicalMapping where :
 *
 * INPUT TOPOLOGY = TetrahedronSetTopology
 * OUTPUT TOPOLOGY = TriangleSetTopology
 *
 * Tetra2TriangleTopologicalMapping class is templated by the pair (INPUT TOPOLOGY, OUTPUT TOPOLOGY)
 *
*/

template <class TIn, class TOut>
class Tetra2TriangleTopologicalMapping : public TopologicalMapping
{

public:
    /// Input Topology
    typedef TIn In;
    /// Output Topology
    typedef TOut Out;

    friend class TopologicalMapping;

protected:
    /// Input source BaseTopology
    In* fromModel;
    /// Output target BaseTopology
    Out* toModel;

    Data< std::string > object1;
    Data< std::string > object2;

public:

    /** \brief Constructor.
     *
     * @param from the topology issuing TopologyChange objects (the "source").
     * @param to   the topology for which the TopologyChange objects must be translated (the "target").
     */
    Tetra2TriangleTopologicalMapping(In* from=NULL, Out* to=NULL);

    /** \brief Destructor.
     *
     * Does nothing.
     */
    virtual ~Tetra2TriangleTopologicalMapping();

    /// Specify the input and output topologies.
    virtual void setModels(In* from, Out* to);

    /// Return the pointer to the input topology.
    In* getFromModel();
    /// Return the pointer to the output topology.
    Out* getToModel();

    /// Return the pointer to the input topology.
    objectmodel::BaseObject* getFrom();
    /// Return the pointer to the output topology.
    objectmodel::BaseObject* getTo();

    /** \brief Initializes the target BaseTopology from the source BaseTopology.
     */
    virtual void init();


    /** \brief Translates the TopologyChange objects from the source to the target.
     *
     * Translates each of the TopologyChange objects waiting in the source list so that they have a meaning and
     * reflect the effects of the first topology changes on the second topology.
     *
     */
    virtual void updateTopologicalMapping();

    /// Pre-construction check method called by ObjectFactory.
    ///
    /// This implementation read the object1 and object2 attributes and check
    /// if they are compatible with the input and output topology types of this
    /// mapping.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (arg->findObject(arg->getAttribute("object1","../..")) == NULL)
            std::cerr << "Cannot create "<<className(obj)<<" as object1 is missing.\n";
        if (arg->findObject(arg->getAttribute("object2","..")) == NULL)
            std::cerr << "Cannot create "<<className(obj)<<" as object2 is missing.\n";
        if (dynamic_cast<In*>(arg->findObject(arg->getAttribute("object1","../.."))) == NULL)
            return false;
        if (dynamic_cast<Out*>(arg->findObject(arg->getAttribute("object2",".."))) == NULL)
            return false;
        return BaseMapping::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    ///
    /// This implementation read the object1 and object2 attributes to
    /// find the input and output topologies of this mapping.
    template<class T>
    static void create(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        obj = new T(
            (arg?dynamic_cast<In*>(arg->findObject(arg->getAttribute("object1","../.."))):NULL),
            (arg?dynamic_cast<Out*>(arg->findObject(arg->getAttribute("object2",".."))):NULL));
        if (context) context->addObject(obj);
        if ((arg) && (arg->getAttribute("object1")))
        {
            obj->object1.setValue( arg->getAttribute("object1") );
            arg->removeAttribute("object1");
        }
        if ((arg) && (arg->getAttribute("object2")))
        {
            obj->object2.setValue( arg->getAttribute("object2") );
            arg->removeAttribute("object2");
        }
        if (arg) obj->parse(arg);
    }

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_TETRA2TRIANGLETOPOLOGICALMAPPING_H
