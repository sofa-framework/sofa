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

#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/DataTracker.h>
#include <sofa/core/fwd.h>

namespace sofa::core::objectmodel
{

/**
 *  \brief Base class for simulation components.
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
    SOFA_BASE_CAST_IMPLEMENTATION(BaseObject)

protected:
    BaseObject();

    ~BaseObject() override;

public:

    /// @name control
    ///   Basic control
    /// @{

    /// Pre-construction check method called by ObjectFactory.
    template<class T>
    static bool canCreate(T* /*obj*/, BaseContext* /*context*/, BaseObjectDescription* /*arg*/)
    {
        return true;
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T*, BaseContext* context, BaseObjectDescription* arg)
    {
        typename T::SPtr obj = sofa::core::objectmodel::New<T>();
        if (context) context->addObject(obj);
        if (arg) obj->parse(arg);
        return obj;
    }

    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse ( BaseObjectDescription* arg ) override;

    /// Initialization method called at graph creation and modification, during top-down traversal.
    virtual void init();

    /// Initialization method called at graph creation and modification, during bottom-up traversal.
    virtual void bwdInit();

    /// Update method called when variables used in precomputation are modified.
    virtual void reinit();

    /// Update method called when variables (used to compute other internal variables) are modified
    void updateInternal();

    /// Save the initial state for later uses in reset()
    virtual void storeResetState();

    /// Reset to initial state
    virtual void reset();

    /// Called just before deleting this object
    /// Any object in the tree bellow this object that are to be removed will be removed only after this call,
    /// so any references this object holds should still be valid.
    virtual void cleanup();

    /// @}

    /// Render internal data of this object, for debugging purposes.
    virtual void draw(const core::visual::VisualParams*)
    {
    }
    ///@}

    /// @name Context accessors
    /// @{

    const BaseContext* getContext() const;

    BaseContext* getContext();

    const BaseObject* getMaster() const;

    BaseObject* getMaster();


    typedef sofa::core::objectmodel::MultiLink<BaseObject, BaseObject, BaseLink::FLAG_DOUBLELINK|BaseLink::FLAG_STRONGLINK> LinkSlaves;
    typedef LinkSlaves::Container VecSlaves;

    const VecSlaves& getSlaves() const;

    BaseObject* getSlave(const std::string& name) const;

    virtual void addSlave(BaseObject::SPtr s);

    virtual void removeSlave(BaseObject::SPtr s);
    /// @}

    /// @name data access
    ///   Access to external data
    /// @{

    /// Current time
    SReal getTime() const;

    /// @}

    /// @name events
    ///   Methods related to Event processing
    /// @{

    Data<bool> f_listening; ///< if true, handle the events, otherwise ignore the events

    /// Handle an event
    virtual void handleEvent( Event* );

    /// Handle topological Changes
    /// @deprecated topological changes now rely on TopologyHandler
    virtual void handleTopologyChange() {}

    /// Handle topological Changes from a given Topology
    /// @deprecated topological changes now rely on TopologyHandler
    virtual void handleTopologyChange(core::topology::Topology* t);

    ///@}

    /// Bounding Box computation method.
    /// Default to empty method.
    virtual void computeBBox(const core::ExecParams* /* params */, bool /*onlyVisible*/=false) {}

    /// Sets a source Object and parses it to collect dependent Data
    void setSrc(const std::string &v, std::vector< std::string > *attributeList=nullptr);

    /// Sets a source Object and parses it to collect dependent Data
    /// Use it before scene graph insertion
    void setSrc(const std::string &v, const BaseObject *loader, std::vector< std::string > *attributeList=nullptr);

    Base* findLinkDestClass(const BaseClass* destType, const std::string& path, const BaseLink* link) override;


    /// Return the full path name of this object
    virtual std::string getPathName() const override;

    /// @name internalupdate
    ///   Methods related to tracking of data and the internal update
    /// @{
private:
    /// Tracker for all component Data linked to internal variables
    sofa::core::DataTracker m_internalDataTracker;

protected:
    /// Method called to add the Data to the DataTracker (listing the Data to track)
    void trackInternalData(const BaseData &data);
    void cleanTracker();

    /// Method called to know if a tracked Data has changed
    bool hasDataChanged(const BaseData &data);
    ///@}
    ///

    SingleLink<BaseObject, BaseContext, BaseLink::FLAG_DOUBLELINK> l_context;
    LinkSlaves l_slaves;
    SingleLink<BaseObject, BaseObject, BaseLink::FLAG_DOUBLELINK> l_master;

    /// Implementation of the internal update
    virtual void doUpdateInternal();

    /// This method insures that context is never nullptr (using BaseContext::getDefault() instead)
    /// and that all slaves of an object share its context
    void changeContextLink(BaseContext* before, BaseContext*& after);

    /// This method insures that slaves objects have master and context links set correctly
    void changeSlavesLink(BaseObject::SPtr ptr, std::size_t /*index*/, bool add);

    /// BaseNode can set the context of its own objects
    friend class BaseNode;


public:

    /// the component can insert itself directly in the right sequence in the Node
    /// so the Node does not have to test its type against all known types
    /// \returns true if the component was inserted
    virtual bool insertInNode( BaseNode* /*node*/ ) { return false; }

    /// the component can remove itself directly in the right sequence in the Node
    /// so the Node does not have to test its type against all known types
    /// \returns true if the component was removed
    virtual bool removeInNode( BaseNode* /*node*/ ) { return false; }
};

} // namespace sofa::core::objectmodel

