#ifndef CMBASETOPOLOGYENGINE_H
#define CMBASETOPOLOGYENGINE_H

#include <sofa/core/topology/CMTopologyChange.h>

namespace sofa
{

namespace core
{

namespace cm_topology
{



/** A class that will interact on a topological Data */
class TopologyEngine : public sofa::core::DataEngine
{
public:
    SOFA_ABSTRACT_CLASS(TopologyEngine, DataEngine);
    //typedef sofa::core::objectmodel::Data< sofa::helper::vector <void*> > t_topologicalData;

protected:
    TopologyEngine() {}//m_topologicalData(NULL)  {}

    virtual ~TopologyEngine()
    {
        //if (this->m_topologicalData != NULL)
        //    this->removeTopologicalData();
    }

public:

    virtual void init()
    {
        sofa::core::DataEngine::init();
        // TODO: see if necessary or not....
        // this->addInput(&m_changeList);

        // TODO: understand why this crash!!
        //this->addOutput(this->m_topologicalData);

        this->createEngineName();
    }

    virtual void handleTopologyChange() {}


public:
    // really need to be a Data??
    Data <sofa::helper::list<const TopologyChange *> >m_changeList;

    size_t getNumberOfTopologicalChanges() {return (m_changeList.getValue()).size();}

    //virtual void registerTopologicalData(t_topologicalData* topologicalData) {m_topologicalData = topologicalData;}
    /*
        virtual void removeTopologicalData()
        {
            if (this->m_topologicalData)
                delete this->m_topologicalData;
        }
    */
    //virtual const t_topologicalData* getTopologicalData() {return m_topologicalData;}

    virtual void createEngineName()
    {
        if (m_data_name.empty())
            setName( m_prefix + "no_name" );
        else
            setName( m_prefix + m_data_name );

        return;
    }

    virtual void linkToPointDataArray() {}
    virtual void linkToEdgeDataArray() {}
    virtual void linkToTriangleDataArray() {}
    virtual void linkToQuadDataArray() {}
    virtual void linkToTetrahedronDataArray() {}
    virtual void linkToHexahedronDataArray() {}

    void setNamePrefix(const std::string& s) { m_prefix = s; }

protected:
    /// Data handle by the topological engine
    //t_topologicalData* m_topologicalData;

    //TopologyHandler* m_topologyHandler;

    /// use to define engine name.
    std::string m_prefix;
    /// use to define data handled name.
    std::string m_data_name;
};

} // namespace cm_topology

} // namespace component

} // namespace sofa

#endif // CMBASETOPOLOGYENGINE_H
