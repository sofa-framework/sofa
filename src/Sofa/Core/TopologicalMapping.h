#ifndef SOFA_CORE_TOPOLOGICALMAPPING_H
#define SOFA_CORE_TOPOLOGICALMAPPING_H

#include "BasicMapping.h"
#include "BasicTopology.h"

namespace Sofa
{

namespace Core
{

/** A class that translates TopologyChange objects from one topology to another so that they have a meaning and
 * reflect the effects of the first topology changes on the second topology.
 */
class TopologicalMapping : public BasicMapping
{

public:

    /** \brief Constructor.
     *
     * @param from the topology issuing TopologyChange objects (the "source").
     * @param to   the topology for which the TopologyChange objects must be translated (the "target").
     */
    TopologicalMapping(BasicTopology* from, BasicTopology* to)
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



    /** \brief Returns source BasicTopology.
     *
     * Returns the topology issuing TopologyChange objects (the "source").
     */
    Abstract::BaseObject* getFrom()
    {
        return m_from;
    }



    /** \brief Returns target BasicTopology.
     *
     * Returns the topology for which the TopologyChange objects must be translated (the "target").
     */
    Abstract::BaseObject* getTo()
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
    BasicTopology* m_from;



    /// The topology for which the TopologyChange objects must be translated (the "target").
    BasicTopology* m_to;



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

} // namespace Core

} // namespace Sofa

#endif // SOFA_CORE_TOPOLOGICALMAPPING_H
