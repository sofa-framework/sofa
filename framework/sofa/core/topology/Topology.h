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
#ifndef SOFA_CORE_TOPOLOGY_TOPOLOGY_H
#define SOFA_CORE_TOPOLOGY_TOPOLOGY_H

#include <stdlib.h>
#include <string>
#include <iostream>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/list.h>
#include <sofa/core/DataEngine.h>

#include <sofa/helper/vector.h>
#include <sofa/helper/fixed_array.h>
namespace sofa
{

namespace core
{

namespace topology
{

using namespace sofa::helper;

/// The enumeration used to give unique identifiers to TopologyChange objects.
enum TopologyChangeType
{
    BASE,                      ///< For TopologyChange class, should never be used.
    ENDING_EVENT,              ///< To notify the end for the current sequence of topological change events

    POINTSINDICESSWAP,         ///< For PointsIndicesSwap class.
    POINTSADDED,               ///< For PointsAdded class.
    POINTSREMOVED,             ///< For PointsRemoved class.
    POINTSMOVED,               ///< For PointsMoved class.
    POINTSRENUMBERING,         ///< For PointsRenumbering class.

    EDGESINDICESSWAP,          ///< For EdgesIndicesSwap class.
    EDGESADDED,                ///< For EdgesAdded class.
    EDGESREMOVED,              ///< For EdgesRemoved class.
    EDGESMOVED_REMOVING,       ///< For EdgesMoved class (event before changing state).
    EDGESMOVED_ADDING,         ///< For EdgesMoved class.
    EDGESRENUMBERING,          ///< For EdgesRenumbering class.

    TRIANGLESINDICESSWAP,      ///< For TrianglesIndicesSwap class.
    TRIANGLESADDED,            ///< For TrianglesAdded class.
    TRIANGLESREMOVED,          ///< For TrianglesRemoved class.
    TRIANGLESMOVED_REMOVING,   ///< For TrianglesMoved class (event before changing state).
    TRIANGLESMOVED_ADDING,     ///< For TrianglesMoved class.
    TRIANGLESRENUMBERING,      ///< For TrianglesRenumbering class.

    TETRAHEDRAINDICESSWAP,     ///< For TetrahedraIndicesSwap class.
    TETRAHEDRAADDED,           ///< For TetrahedraAdded class.
    TETRAHEDRAREMOVED,         ///< For TetrahedraRemoved class.
    TETRAHEDRAMOVED_REMOVING,  ///< For TetrahedraMoved class (event before changing state).
    TETRAHEDRAMOVED_ADDING,    ///< For TetrahedraMoved class.
    TETRAHEDRARENUMBERING,     ///< For TetrahedraRenumbering class.

    QUADSINDICESSWAP,          ///< For QuadsIndicesSwap class.
    QUADSADDED,                ///< For QuadsAdded class.
    QUADSREMOVED,              ///< For QuadsRemoved class.
    QUADSMOVED_REMOVING,       ///< For QuadsMoved class (event before changing state).
    QUADSMOVED_ADDING,         ///< For QuadsMoved class.
    QUADSRENUMBERING,          ///< For QuadsRenumbering class.

    HEXAHEDRAINDICESSWAP,      ///< For HexahedraIndicesSwap class.
    HEXAHEDRAADDED,            ///< For HexahedraAdded class.
    HEXAHEDRAREMOVED,          ///< For HexahedraRemoved class.
    HEXAHEDRAMOVED_REMOVING,   ///< For HexahedraMoved class (event before changing state).
    HEXAHEDRAMOVED_ADDING,     ///< For HexahedraMoved class.
    HEXAHEDRARENUMBERING,      ///< For HexahedraRenumbering class.

    TOPOLOGYCHANGE_LASTID      ///< user defined topology changes can start here
};


/// The enumeration used to give unique identifiers to Topological objects.
enum TopologyObjectType
{
    POINT,
    EDGE,
    TRIANGLE,
    QUAD,
    TETRAHEDRON,
    HEXAHEDRON
};



/** \brief Base class to indicate a topology change occurred.
*
* All topological changes taking place in a given BaseTopology will issue a TopologyChange in the
* BaseTopology's changeList, so that BasicTopologies mapped to it can know what happened and decide how to
* react.
* Classes inheriting from this one describe a given topolopy change (e.g. RemovedPoint, AddedEdge, etc).
* The exact type of topology change is given by member changeType.
*/
class TopologyChange
{
public:
    /** \ brief Destructor.
    *
    	* Must be virtual for TopologyChange to be a Polymorphic type.
    	*/
    virtual ~TopologyChange() {}

    /** \brief Returns the code of this TopologyChange. */
    TopologyChangeType getChangeType() const { return m_changeType;}


    /// Output empty stream
    inline friend std::ostream& operator<< ( std::ostream& os, const TopologyChange* /*t*/ )
    {
        return os;
    }

    /// Input empty stream
    inline friend std::istream& operator>> ( std::istream& in, const TopologyChange* /*t*/ )
    {
        return in;
    }

protected:
    TopologyChange( TopologyChangeType changeType = BASE )
        : m_changeType(changeType)
    {}

    TopologyChangeType m_changeType; ///< A code that tells the nature of the Topology modification event (could be an enum).
};

/** notifies the end for the current sequence of topological change events */
class EndingEvent : public core::topology::TopologyChange
{
public:
    EndingEvent()
        : core::topology::TopologyChange(core::topology::ENDING_EVENT)
    {}
};




/** A class that define topological Data general methods*/
template < class T = void* >
class BaseTopologyData : public sofa::core::objectmodel::Data <T>
{
public:
    //SOFA_CLASS(SOFA_TEMPLATE2(BaseTopologyData,T,VecT), SOFA_TEMPLATE(sofa::core::objectmodel::Data, T));

    class InitData : public sofa::core::objectmodel::BaseData::BaseInitData
    {
    public:
        InitData() : value(T()) {}
        InitData(const T& v) : value(v) {}
        InitData(const sofa::core::objectmodel::BaseData::BaseInitData& i) : sofa::core::objectmodel::BaseData::BaseInitData(i), value(T()) {}

        T value;
    };

    /** Constructor
        this constructor should be used through the initData() methods
     */
    explicit BaseTopologyData(const sofa::core::objectmodel::BaseData::BaseInitData& init)
        : Data<T>(init)
    {
    }

    /** Constructor
        this constructor should be used through the initData() methods
     */
    explicit BaseTopologyData(const InitData& init)
        : Data<T>(init)
    {
    }


    /** Constructor
    \param helpMsg help on the field
     */
    BaseTopologyData( const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false, sofa::core::objectmodel::Base* owner=NULL, const char* name="")
        : Data<T>(helpMsg, isDisplayed, isReadOnly, owner, name)
    {

    }

    /** Constructor
    \param value default value
    \param helpMsg help on the field
     */
    BaseTopologyData( const T& value, const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false, sofa::core::objectmodel::Base* owner=NULL, const char* name="")
        : Data<T>(helpMsg, isDisplayed, isReadOnly, owner, name)
    {
    }

};


class TopologyHandler
{
public:
    virtual ~TopologyHandler() {}


    /// Handle EdgeSetTopology related events, ignore others. DEPRECATED
    virtual void handleTopologyEvents( std::list< const core::topology::TopologyChange *>::const_iterator changeIt,
            std::list< const core::topology::TopologyChange *>::const_iterator &end );
};

/** A class that will interact on a topological Data */
class SOFA_CORE_API TopologyEngine : public sofa::core::DataEngine
{
public:
    SOFA_ABSTRACT_CLASS(TopologyEngine, DataEngine);
    typedef sofa::core::objectmodel::Data< sofa::helper::vector <void*> > t_topologicalData;

    TopologyEngine(): m_topologicalData(NULL)  {}

    virtual ~TopologyEngine()
    {
        if (this->m_topologicalData != NULL)
            this->removeTopologicalData();
    }

    virtual void init()
    {
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

    unsigned int getNumberOfTopologicalChanges() {return (m_changeList.getValue()).size();}

    virtual void registerTopologicalData(t_topologicalData* topologicalData) {m_topologicalData = topologicalData;}

    virtual void removeTopologicalData()
    {
        if (this->m_topologicalData)
            delete this->m_topologicalData;
    }

    virtual const t_topologicalData* getTopologicalData() {return m_topologicalData;}

    virtual void createEngineName()
    {
        if (m_data_name.empty())
            m_name = m_prefix + "no_name";
        else
            m_name = m_prefix + m_data_name;

        return;
    }

    const std::string& getName() const { return m_name; }

    void setName(const std::string& name) { m_name=name; }

    virtual void linkToPointDataArray() {}
    virtual void linkToEdgeDataArray() {}
    virtual void linkToTriangleDataArray() {}
    virtual void linkToQuadDataArray() {}
    virtual void linkToTetrahedronDataArray() {}
    virtual void linkToHexahedronDataArray() {}

protected:
    /// Data handle by the topological engine
    t_topologicalData* m_topologicalData;

    TopologyHandler* m_topologyHandler;

    /// Engine name base on Data handled: m_name = m_prefix+Data_name
    std::string m_name;
    /// use to define engine name.
    std::string m_prefix;
    /// use to define data handled name.
    std::string m_data_name;
};



class Topology : public virtual core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(Topology, core::objectmodel::BaseObject);
protected:
    Topology():BaseObject() {}
    virtual ~Topology()
    {}
public:
    // Access to embedded position information (in case the topology is a regular grid for instance)
    // This is not very clean and is quit slow but it should only be used during initialization

    virtual bool hasPos() const { return false; }
    virtual int getNbPoints() const { return 0; }
    virtual void setNbPoints(int /*n*/) {}
    virtual double getPX(int /*i*/) const { return 0.0; }
    virtual double getPY(int /*i*/) const { return 0.0; }
    virtual double getPZ(int /*i*/) const { return 0.0; }
};

} // namespace topology

} // namespace core

} // namespace sofa

#endif
