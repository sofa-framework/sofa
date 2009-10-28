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
#ifndef SOFA_CORE_OBJECTMODEL_BASEOBJECT_H
#define SOFA_CORE_OBJECTMODEL_BASEOBJECT_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/core/objectmodel/Tag.h>

#include <sofa/helper/set.h>


namespace sofa
{

namespace core
{

// forward declaration of referenced classes
namespace componentmodel
{
namespace topology
{
class Topology;
//class BaseTopology;
//class BaseMeshTopology;
}
}

namespace objectmodel
{

class Event;

/**
 *  \brief Base class for simulation objects.
 *
 *  An object defines a part of the functionnality in the simulation
 *  (stores state data, specify topology, compute forces, etc).
 *  Each simulation object is related to a context, which gives access to all available external data.
 *  It is able to process events, if listening enabled (default is false).
 *
 */
class SOFA_CORE_API BaseObject : public virtual Base
{
public:
    SOFA_CLASS(BaseObject, Base);

    BaseObject();

    virtual ~BaseObject();

    /// @name Context accessors
    /// @{
    void setContext(BaseContext* n);

    const BaseContext* getContext() const;

    BaseContext* getContext();
    /// @}

    /// @name control
    ///   Basic state control
    /// @{

    /// Pre-construction check method called by ObjectFactory.
    template<class T>
    static bool canCreate(T*& /*obj*/, BaseContext* /*context*/, BaseObjectDescription* /*arg*/)
    {
        return true;
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static void create(T*& obj, BaseContext* context, BaseObjectDescription* arg)
    {
        obj = new T;
        if (context) context->addObject(obj);
        if (arg) obj->parse(arg);
    }

    /// Parse the given description to assign values to this object's fields and potentially other parameters
    virtual void parse ( BaseObjectDescription* arg );

    /// Initialization method called at graph modification, during bottom-up traversal.
    virtual void init();

    /// Initialization method called at graph modification, during top-down traversal.
    virtual void bwdInit();

    /// Update method called when variables used in precomputation are modified.
    virtual void reinit();

    /// Save the initial state for later uses in reset()
    virtual void storeResetState();

    /// Reset to initial state
    virtual void reset();

    /// Write current state to the given output stream
    virtual void writeState( std::ostream& out );

    /// Called just before deleting this object
    /// Any object in the tree bellow this object that are to be removed will be removed only after this call,
    /// so any references this object holds should still be valid.
    virtual void cleanup();

    /// @}

    /// @name events
    ///   Methods related to Event processing
    /// @{

    Data<bool> f_listening;

    /// Handle an event
    virtual void handleEvent( Event* );

    /// Handle topological Changes
    virtual void handleTopologyChange() {}

    /// Handle topological Changes from a given Topology
    virtual void handleTopologyChange(core::componentmodel::topology::Topology* t);

    /// Handle state Changes
    virtual void handleStateChange() {}

    /// Handle state Changes from a given Topology
    virtual void handleStateChange(core::componentmodel::topology::Topology* t);

    ///@}

    /**
     *  \brief Render internal data of this object, for debugging purposes.
     */
    virtual void draw() {}

    ///@}

    /// @name data access
    ///   Access to external data
    /// @{

    /// Current time
    double getTime() const;

    /// @}

    /// @name tags
    ///   Methods related to subsets belonging
    /// @{

    /// Represents the subsets the object belongs to
    Data< sofa::core::objectmodel::TagSet > f_tags;

    const sofa::core::objectmodel::TagSet& getTags() const { return f_tags.getValue(); }

    /// Return true if the object belong to the given subset
    bool hasTag( Tag t ) const;

    /// Add a subset qualification to the object
    void addTag(Tag t);
    /// Remove a subset qualification to the object
    void removeTag(Tag t);

    /// @}

    virtual bool canPrefetch() const { return false; }

    static void setPrefetching(bool b);
    static bool isPrefetching();

protected:
    BaseContext* context_;
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
