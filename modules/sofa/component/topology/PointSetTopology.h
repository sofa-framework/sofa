/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGY_H

#include <sofa/core/componentmodel/topology/BaseTopology.h>
#include <sofa/component/MechanicalObject.h>

#include <sofa/component/topology/PointSetGeometryAlgorithms.h>
#include <sofa/component/topology/PointSetTopologyAlgorithms.h>
#include <sofa/component/topology/PointSetTopologyModifier.h>
#include <sofa/component/topology/PointSetTopologyContainer.h>
#include <sofa/component/topology/PointSetTopologyChange.h>

namespace sofa
{

namespace component
{

namespace topology
{
template<class DataTypes>
class PointSetTopology;

template<class DataTypes>
class PointSetTopologyAlgorithms;

template<class DataTypes>
class PointSetGeometryAlgorithms;

template< typename DataTypes >
class PointSetTopologyLoader;

template<class DataTypes>
class PointSetTopologyModifier;

class PointSetTopologyContainer;

class PointsIndicesSwap;
class PointsAdded;
class PointsRemoved;
class PointsRenumbering;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::PointID PointID;

/** Describes a topological object that only consists as a set of points :
it is a base class for all topological objects */
template<class DataTypes>
class PointSetTopology : public core::componentmodel::topology::BaseTopology
{
public:
    PointSetTopology(component::MechanicalObject<DataTypes> *obj);

    virtual ~PointSetTopology() {}

    /// Parse the given description to assign values to this object's fields and potentially other parameters
    virtual void parse(sofa::core::objectmodel::BaseObjectDescription* arg);

    virtual void init();

    /** \brief Returns the PointSetTopologyContainer object of this PointSetTopologyContainer.
    */
    PointSetTopologyContainer *getPointSetTopologyContainer() const
    {
        return static_cast<PointSetTopologyContainer *> (this->m_topologyContainer);
    }

    /** \brief Returns the PointSetTopologyModifier object of this PointSetTopology.
    */
    PointSetTopologyModifier<DataTypes> *getPointSetTopologyModifier() const
    {
        return static_cast<PointSetTopologyModifier<DataTypes> *> (this->m_topologyModifier);
    }

    /** \brief Returns the PointSetTopologyAlgorithms object of this PointSetTopology.
    */
    PointSetTopologyAlgorithms<DataTypes> *getPointSetTopologyAlgorithms() const
    {
        return static_cast<PointSetTopologyAlgorithms<DataTypes> *> (this->m_topologyAlgorithms);
    }

    /** \brief Returns the PointSetGeometryAlgorithms object of this PointSetTopology.
    */
    PointSetGeometryAlgorithms<DataTypes> *getPointSetGeometryAlgorithms() const
    {
        return static_cast<PointSetGeometryAlgorithms<DataTypes> *> (this->m_geometryAlgorithms);
    }

    /** \brief Called by a topology to warn specific topologies linked to it that TopologyChange objects happened.
    *
    * Member m_changeList should contain all TopologyChange objects corresponding to changes in this topology
    * that just happened (in the case of creation) or are about to happen (in the case of destruction) since
    * last call to propagateTopologicalChanges.
    *
    * @see BaseTopology::m_changeList
    * @sa firstChange()
    * @sa lastChange()
    */
    virtual void propagateTopologicalChanges();

    /** \brief Called by a topology to warn the Mechanical Object component that points have been added or will be removed.
    *
    * Member m_StateChangeList should contain all TopologyChange objects corresponding to vertex changes in this topology
    * that just happened (in the case of creation) or are about to happen (in the case of destruction) since
    * last call to propagateTopologicalChanges.
    *
    * @see BaseTopology::m_changeList
    * @sa firstChange()
    * @sa lastChange()
    */
    virtual void propagateStateChanges();

    /** return the latest revision number */
    virtual int getRevision() const { return revisionCounter; }

    /** \brief Returns the object where the mechanical DOFs are stored */
    component::MechanicalObject<DataTypes> *getDOF() const { return object;	}

    /** \brief Build a topology from a file : call the load member function in the modifier object
    *
    */
    virtual bool load(const char *filename);

    /** \brief Translates the DOF : call the applyTranslation member function in the modifier object
    *
    */
    virtual void applyTranslation (const double dx,const double dy,const double dz);

    /** \brief Scales the DOF : call the applyScale member function in the modifier object
    *
    */
    virtual void applyScale (const double s);

    /** \brief Return the number of DOF in the mechanicalObject this Topology deals with.
    *
    */
    virtual unsigned int getDOFNumber() const { return object->getSize(); }


    /// BaseMeshTopology API
    /// @{
    virtual void clear()                       { }
    virtual const SeqEdges& getEdges()         { static SeqEdges     empty; return empty; }
    virtual const SeqTriangles& getTriangles() { static SeqTriangles empty; return empty; }
    virtual const SeqQuads& getQuads()         { static SeqQuads     empty; return empty; }
    virtual const SeqTetras& getTetras()       { static SeqTetras    empty; return empty; }
    virtual const SeqHexas& getHexas()         { static SeqHexas     empty; return empty; }
    /// @}

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalObject.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<MechanicalObject<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return core::componentmodel::topology::BaseTopology::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    ///
    /// Get the MechanicalObject.
    template<class T>
    static void create(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        obj = new T(
            (context?dynamic_cast<MechanicalObject<DataTypes>*>(context->getMechanicalState()):NULL));
        if (context) context->addObject(obj);
        if (arg) obj->parse(arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const PointSetTopology<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

public:
    /** the object where the mechanical DOFs are stored */
    component::MechanicalObject<DataTypes> *object;					// TODO: clarify, should not this be in the container?

    // TODO: clarify, do these members have to be public?
    DataPtr< PointSetTopologyContainer > *f_m_topologyContainer;	// TODO: clarify, what is this needed for

protected:
    virtual void createComponents();

private:
    int revisionCounter;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_POINTSETTOPOLOGY_H
