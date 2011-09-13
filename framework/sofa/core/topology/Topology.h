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
namespace sofa
{

namespace core
{

namespace topology
{

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

    EDGESADDED,                ///< For EdgesAdded class.
    EDGESREMOVED,              ///< For EdgesRemoved class.
    EDGESMOVED_REMOVING,       ///< For EdgesMoved class (event before changing state).
    EDGESMOVED_ADDING,         ///< For EdgesMoved class.
    EDGESRENUMBERING,          ///< For EdgesRenumbering class.

    TRIANGLESADDED,            ///< For TrianglesAdded class.
    TRIANGLESREMOVED,          ///< For TrianglesRemoved class.
    TRIANGLESMOVED_REMOVING,   ///< For TrianglesMoved class (event before changing state).
    TRIANGLESMOVED_ADDING,     ///< For TrianglesMoved class.
    TRIANGLESRENUMBERING,      ///< For TrianglesRenumbering class.

    TETRAHEDRAADDED,           ///< For TrianglesAdded class.
    TETRAHEDRAREMOVED,         ///< For TrianglesRemoved class.
    TETRAHEDRARENUMBERING,     ///< For TrianglesRenumbering class.

    QUADSADDED,                ///< For QuadsAdded class.
    QUADSREMOVED,              ///< For QuadsRemoved class.
    QUADSRENUMBERING,          ///< For QuadsRenumbering class.

    HEXAHEDRAADDED,            ///< For TrianglesAdded class.
    HEXAHEDRAREMOVED,          ///< For TrianglesRemoved class.
    HEXAHEDRARENUMBERING,      ///< For TrianglesRenumbering class.

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
template < class T = void*, class VecT = helper::vector<T> >
class SOFA_CORE_API TopologicalData : public sofa::core::objectmodel::Data <T>
{
public:

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
    explicit TopologicalData(const sofa::core::objectmodel::BaseData::BaseInitData& init)
        : Data<T>(init)
    {
    }

    /** Constructor
        this constructor should be used through the initData() methods
     */
    explicit TopologicalData(const InitData& init)
        : Data<T>(init)
    {
    }


    /** Constructor
    \param helpMsg help on the field
     */
    TopologicalData( const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false, sofa::core::objectmodel::Base* owner=NULL, const char* name="")
        : Data<T>(helpMsg, isDisplayed, isReadOnly, owner, name)
    {

    }

    /** Constructor
    \param value default value
    \param helpMsg help on the field
     */
    TopologicalData( const T& value, const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false, sofa::core::objectmodel::Base* owner=NULL, const char* name="")
        : Data<T>(helpMsg, isDisplayed, isReadOnly, owner, name)
    {
    }


    // Generic methods to apply changes on the Data
    //{
    /// Apply adding points elements.
    virtual void applyCreatePointFunction(const sofa::helper::vector<unsigned int>& indices) {}
    /// Apply removing points elements.
    virtual void applyDestroyPointFunction(const sofa::helper::vector<unsigned int>& indices) {}

    /// Apply adding edges elements.
    virtual void applyCreateEdgeFunction(const sofa::helper::vector<unsigned int>& indices) {}
    /// Apply removing edges elements.
    virtual void applyDestroyEdgeFunction(const sofa::helper::vector<unsigned int>& indices) {}

    /// Apply adding triangles elements.
    virtual void applyCreateTriangleFunction(const sofa::helper::vector<unsigned int>& indices) {}
    /// Apply removing triangles elements.
    virtual void applyDestroyTriangleFunction(const sofa::helper::vector<unsigned int>& indices) {}

    /// Apply adding quads elements.
    virtual void applyCreateQuadFunction(const sofa::helper::vector<unsigned int>& indices) {}
    /// Apply removing quads elements.
    virtual void applyDestroyQuadFunction(const sofa::helper::vector<unsigned int>& indices) {}

    /// Apply adding tetrahedra elements.
    virtual void applyCreateTetrahedronFunction(const sofa::helper::vector<unsigned int>& indices) {}
    /// Apply removing tetrahedra elements.
    virtual void applyDestroyTetrahedronFunction(const sofa::helper::vector<unsigned int>& indices) {}

    /// Apply adding hexahedra elements.
    virtual void applyCreateHexahedronFunction(const sofa::helper::vector<unsigned int>& indices) {}
    /// Apply removing hexahedra elements.
    virtual void applyDestroyHexahedronFunction(const sofa::helper::vector<unsigned int>& indices) {}
    //}

    // Data specific methods
    virtual void add(unsigned int nbElements, const sofa::helper::vector<T>& elements,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs) {}

    /// Add some values. Values are added at the end of the vector.
    virtual void add(unsigned int nbElements,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs) {}

    /// Remove the values corresponding to the points removed.
    virtual void remove( const sofa::helper::vector<unsigned int>& index ) {}

    /// Swaps values at indices i1 and i2.
    virtual void swap( unsigned int i1, unsigned int i2 ) {}

    /// Reorder the values.
    virtual void renumber( const sofa::helper::vector<unsigned int>& index ) {}

    /// Move a list of points
    virtual void move( const sofa::helper::vector<unsigned int>& indexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs) {}

};



/** A class that will interact on a topological Data */
class SOFA_CORE_API TopologyEngine : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(TopologyEngine, DataEngine);
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

protected:
    /// Data handle by the topological engine
    t_topologicalData* m_topologicalData;

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

    Topology():BaseObject() {}
    virtual ~Topology()
    {}

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
