#ifndef SOFA_CORE_COMPONENTMODEL_TOPOLOGY_TOPOLOGICALMAPPING_H
#define SOFA_CORE_COMPONENTMODEL_TOPOLOGY_TOPOLOGICALMAPPING_H

#include <sofa/core/Mapping.h>
#include <sofa/core/componentmodel/topology/Topology.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace topology
{

/** A class that translates TopologyChange objects from one topology to another so that they have a meaning and
         * reflect the effects of the first topology changes on the second topology.
         */
class TopologicalMapping : public BaseMapping
{

public:

    /** \brief Constructor.
     *
     * @param from the topology issuing TopologyChange objects (the "source").
     * @param to   the topology for which the TopologyChange objects must be translated (the "target").
     */
    TopologicalMapping(BaseTopology* from, BaseTopology* to)
        : m_from(from),
          m_to(to)
    {
    }



    /** \brief Destructor.
     *
     * Does nothing.
     */
    virtual ~TopologicalMapping()
    {
    }



    /** \brief Returns source BaseTopology.
     *
     * Returns the topology issuing TopologyChange objects (the "source").
     */
    objectmodel::BaseObject* getFrom()
    {
        return m_from;
    }



    /** \brief Returns target BaseTopology.
     *
     * Returns the topology for which the TopologyChange objects must be translated (the "target").
     */
    objectmodel::BaseObject* getTo()
    {
        return m_to;
    }



    /** \brief Translates the TopologyChange objects from the source to the target.
     *
     * Translates each of the TopologyChange objects waiting in the source list so that they have a meaning and
     * reflect the effects of the first topology changes on the second topology.
     *
     * Possible translating of the TopologyChange object may be, but is not limited to :
     * - ignoring,
     * - passing it 'as is',
     * - changing indices,
     * - computation of values (averaging, interpolating, etc)
     */
    virtual void updateSpecificTopology() = 0;



    /** \brief Calls updateMapping.
     *
     * Standard implementation. Subclasses might have to override this behavior.
     */
    virtual void init()
    {
        updateMapping();
    }



    /** \brief Calls updateSpecificTopology.
     *
     */
    virtual void updateMapping()
    {
        if (m_from && m_to)
            updateSpecificTopology();
    }

protected:
    /// The topology issuing TopologyChange objects (the "source").
    BaseTopology* m_from;



    /// The topology for which the TopologyChange objects must be translated (the "target").
    BaseTopology* m_to;



    /** \brief Adds a TopologyChange object to the target list.
     *
     * Object added should be translated ones.
     *
     * @see updateSpecificTopology()
     */
    void addTopologyChangeToSpecificTopology(const TopologyChange *topologyChange)
    {
        m_to->m_changeList.push_back(topologyChange);
    }

};

} // namespace topology

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_TOPOLOGICALMAPPING_H
