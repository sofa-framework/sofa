/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_OBJECTMODEL_BASECONTEXT_H
#define SOFA_CORE_OBJECTMODEL_BASECONTEXT_H

#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseLink.h>
#include <sofa/core/objectmodel/Tag.h>
#include <sofa/core/objectmodel/ClassInfo.h>
#include <sofa/core/ExecParams.h>

#ifdef SOFA_SUPPORT_MOVING_FRAMES
#include <sofa/defaulttype/SolidTypes.h>
#endif

#include <set>

namespace sofa
{

namespace simulation
{
class Visitor;
}

namespace core
{

// forward declaration of classes accessible from the context

namespace objectmodel
{
class BaseObject;
class Event;

/**
 *  \brief Base class for Context classes, storing shared variables and parameters.
 *
 *  A Context contains values or pointers to variables and parameters shared
 *  by a group of objects, typically refering to the same simulated body.
 *  Derived classes can defined simple isolated contexts or more powerful
 *  hierarchical representations (scene-graphs), in which case the context also
 *  implements the BaseNode interface.
 *
 * \author Jeremie Allard
 */
class SOFA_CORE_API BaseContext : public virtual Base
{
public:
    SOFA_CLASS(BaseContext, Base);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseContext)

    /// @name Types defined for local coordinate system handling
    /// @{

#ifdef SOFA_SUPPORT_MOVING_FRAMES
    typedef defaulttype::SolidTypes<SReal> SolidTypes;
    typedef SolidTypes::Transform Frame;
    typedef SolidTypes::SpatialVector SpatialVector;
#endif
//    typedef SolidTypes::Rot Quat;
//    typedef SolidTypes::Mat Mat33;
    /// @}

    typedef defaulttype::Vector3 Vec3;

protected:
    BaseContext();
    virtual ~BaseContext();

private:
    BaseContext(const BaseContext&);
    BaseContext& operator=(const BaseContext& );

public:
    /// Get the default Context object, that contains the default values for
    /// all parameters and can be used when no local context is defined.
    static BaseContext* getDefault();

    /// Specification of where to search for queried objects
    enum SearchDirection { SearchUp = -1, Local = 0, SearchDown = 1, SearchRoot = 2, SearchParents = 3 };

    /// @name Parameters
    /// @{

    /// The Context is active
    virtual bool isActive() const;

    /// State of the context
    virtual void setActive(bool) {}

    /// Sleeping state of the context
    virtual bool isSleeping() const;

    /// Whether the context can change its sleeping state or not
    virtual bool canChangeSleepingState() const;

    /// Simulation time
    virtual SReal getTime() const;

    /// Simulation timestep
    virtual SReal getDt() const;

    /// Animation flag
    virtual bool getAnimate() const;



#ifdef SOFA_SUPPORT_MULTIRESOLUTION
    /// Multiresolution support (UNSTABLE)
    virtual int getCurrentLevel() const;

    /// Multiresolution support (UNSTABLE)
    virtual int getCoarsestLevel() const;

    /// Multiresolution support (UNSTABLE)
    virtual int getFinestLevel() const;

    /// Multiresolution support (UNSTABLE)
    //     virtual unsigned int nbLevels() const;
#endif

    /// @}

#ifdef SOFA_SUPPORT_MOVING_FRAMES
    /// @name Local Coordinate System
    /// @{
    /// Projection from the local coordinate system to the world coordinate system.
    virtual const Frame& getPositionInWorld() const;
    /// Projection from the local coordinate system to the world coordinate system.
    virtual void setPositionInWorld(const Frame&)
    {}

    /// Spatial velocity (linear, angular) of the local frame with respect to the world
    virtual const SpatialVector& getVelocityInWorld() const;
    /// Spatial velocity (linear, angular) of the local frame with respect to the world
    virtual void setVelocityInWorld(const SpatialVector&)
    {}

    /// Linear acceleration of the origin induced by the angular velocity of the ancestors
    virtual const Vec3& getVelocityBasedLinearAccelerationInWorld() const;
    /// Linear acceleration of the origin induced by the angular velocity of the ancestors
    virtual void setVelocityBasedLinearAccelerationInWorld(const Vec3& )
    {}
    /// Gravity in local coordinates  TODO: replace with world coordinates
    virtual Vec3 getLocalGravity() const;
    ///// Gravity in local coordinates
    //virtual void setGravity( const Vec3& ) { }
    /// @}
#endif

    /// Gravity in local coordinates
    virtual const Vec3& getGravity() const;
    /// Gravity in local coordinates
    virtual void setGravity( const Vec3& )
    { }

    /// Get the root context of the graph
    virtual BaseContext* getRootContext() const;

    /// @name Containers
    /// @{

    /// Mechanical Degrees-of-Freedom
    virtual core::BaseState* getState() const;

    /// Mechanical Degrees-of-Freedom
    virtual behavior::BaseMechanicalState* getMechanicalState() const;

    /// Topology
    virtual core::topology::Topology* getTopology() const;

    /// Mesh Topology (unified interface for both static and dynamic topologies)
    virtual core::topology::BaseMeshTopology* getMeshTopology() const;

    /// Mesh Topology that is local to this context (i.e. not within parent contexts)
    virtual core::topology::BaseMeshTopology* getLocalMeshTopology() const;

    /// Mesh Topology that is relevant for this context, either local or within
    /// a parent until a mapping is reached that does not preserve topologies.
    virtual core::topology::BaseMeshTopology* getActiveMeshTopology() const;

    /// Mass
    virtual core::behavior::BaseMass* getMass() const;

    /// Global Shader
    virtual core::visual::Shader* getShader() const;

    /// Generic object access, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void* getObject(const ClassInfo& class_info, SearchDirection dir = SearchUp) const;

    /// Generic object access, given a set of required tags, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void* getObject(const ClassInfo& class_info, const TagSet& tags, SearchDirection dir = SearchUp) const;

    /// Generic object access, given a path from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void* getObject(const ClassInfo& class_info, const std::string& path) const;

    class GetObjectsCallBack
    {
    public:
        virtual ~GetObjectsCallBack() {}
        virtual void operator()(void* ptr) = 0;
    };

    /// Generic list of objects access, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void getObjects(const ClassInfo& class_info, GetObjectsCallBack& container, SearchDirection dir = SearchUp) const;

    /// Generic list of objects access, given a set of required tags, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void getObjects(const ClassInfo& class_info, GetObjectsCallBack& container, const TagSet& tags, SearchDirection dir = SearchUp) const;

    /// List all objects of this node deriving from a given class
    template<class Object, class Container>
    void getObjects(Container* list, SearchDirection dir = SearchUp)
    {
        this->get<Object, Container>(list, dir);
    }

    /// Returns a list of object of type passed as a parameter.
    template<class Container>
    Container* getObjects(Container* result, SearchDirection dir = SearchUp){
        this->get<typename std::remove_pointer<typename Container::value_type>::type, Container>(result, dir);
        return result ;
    }

    /// Returns a list of object of type passed as a parameter.
    /// eg:
    ///       sofa::helper::vector<VisualModel*> results;
    ///       context->getObjects(results) ;
    template<class Container>
    Container& getObjects(Container& result, SearchDirection dir = SearchUp){
        this->get<typename std::remove_pointer<typename Container::value_type>::type, Container>(&result, dir);
        return result ;
    }

    /// Returns a list of object of type passed as a parameter. There shoud be no
    /// Copy constructor because of Return Value Optimization.
    /// eg:
    ///    for(BaseObject* o : context->getObjects() ){ ... }
    ///    for(VisualModel* o : context->getObjects<VisualModel>() ){ ... }
    template<class Object=sofa::core::objectmodel::BaseObject>
    std::vector<Object*> getObjects(SearchDirection dir = SearchUp){
        std::vector<Object*> o;
        getObjects(o, dir) ;
        return o ;
    }


    /// Generic object access template wrapper, possibly searching up or down from the current context
    template<class T>
    T* get(SearchDirection dir = SearchUp) const
    {
        return reinterpret_cast<T*>(this->getObject(classid(T), dir));
    }


    /// Generic object access template wrapper, possibly searching up or down from the current context
    template<class T>
    void get(T*& ptr, SearchDirection dir = SearchUp) const
    {
        ptr = this->get<T>(dir);
    }

    /// Generic object access template wrapper, possibly searching up or down from the current context
    template<class T>
    void get(sptr<T>& ptr, SearchDirection dir = SearchUp) const
    {
        ptr = this->get<T>(dir);
    }

    /// Generic object access template wrapper, given a required tag, possibly searching up or down from the current context
    template<class T>
    T* get(const Tag& tag, SearchDirection dir = SearchUp) const
    {
        return reinterpret_cast<T*>(this->getObject(classid(T), TagSet(tag), dir));
    }

    /// Generic object access template wrapper, given a required tag, possibly searching up or down from the current context
    template<class T>
    void get(T*& ptr, const Tag& tag, SearchDirection dir = SearchUp) const
    {
        ptr = this->get<T>(tag, dir);
    }

    /// Generic object access template wrapper, given a required tag, possibly searching up or down from the current context
    template<class T>
    void get(sptr<T>& ptr, const Tag& tag, SearchDirection dir = SearchUp) const
    {
        ptr = this->get<T>(tag, dir);
    }

    /// Generic object access template wrapper, given a set of required tags, possibly searching up or down from the current context
    template<class T>
    T* get(const TagSet& tags, SearchDirection dir = SearchUp) const
    {
        return reinterpret_cast<T*>(this->getObject(classid(T), tags, dir));
    }

    /// Generic object access template wrapper, given a set of required tags, possibly searching up or down from the current context
    template<class T>
    void get(T*& ptr, const TagSet& tags, SearchDirection dir = SearchUp) const
    {
        ptr = this->get<T>(tags, dir);
    }

    /// Generic object access template wrapper, given a set of required tags, possibly searching up or down from the current context
    template<class T>
    void get(sptr<T>& ptr, const TagSet& tags, SearchDirection dir = SearchUp) const
    {
        ptr = this->get<T>(tags, dir);
    }

    /// Generic object access template wrapper, given a path from the current context
    template<class T>
    T* get(const std::string& path) const
    {
        return reinterpret_cast<T*>(this->getObject(classid(T), path));
    }

    /// Generic object access template wrapper, given a path from the current context
    template<class T>
    void get(T*& ptr, const std::string& path) const
    {
        ptr = this->get<T>(path);
    }

    /// Generic object access template wrapper, given a path from the current context
    template<class T>
    void get(sptr<T>& ptr, const std::string& path) const
    {
        ptr = this->get<T>(path);
    }

    template<class T, class Container>
    class GetObjectsCallBackT : public GetObjectsCallBack
    {
    public:
        Container* dest;
        GetObjectsCallBackT(Container* d) : dest(d) {}
        virtual void operator()(void* ptr)
        {
            dest->push_back(reinterpret_cast<T*>(ptr));
        }
    };

    /// Generic list of objects access template wrapper, possibly searching up or down from the current context
    template<class T, class Container>
    void get(Container* list, SearchDirection dir = SearchUp) const
    {
        GetObjectsCallBackT<T,Container> cb(list);
        this->getObjects(classid(T), cb, dir);
    }

    /// Generic list of objects access template wrapper, given a required tag, possibly searching up or down from the current context
    template<class T, class Container>
    void get(Container* list, const Tag& tag, SearchDirection dir = SearchUp) const
    {
        GetObjectsCallBackT<T,Container> cb(list);
        this->getObjects(classid(T), cb, TagSet(tag), dir);
    }

    /// Generic list of objects access template wrapper, given a set of required tags, possibly searching up or down from the current context
    template<class T, class Container>
    void get(Container* list, const TagSet& tags, SearchDirection dir = SearchUp) const
    {
        GetObjectsCallBackT<T,Container> cb(list);
        this->getObjects(classid(T), cb, tags, dir);
    }

    /// @}

    /// @name Parameters Setters
    /// @{


    /// Simulation timestep
    virtual void setDt( SReal /*dt*/ )
    { }

    /// Animation flag
    virtual void setAnimate(bool /*val*/)
    { }

    /// Sleeping state of the context
    virtual void setSleeping(bool /*val*/)
    { }

    /// Sleeping state change of the context
    virtual void setChangeSleepingState(bool /*val*/)
    { }

#ifdef SOFA_SUPPORT_MULTIRESOLUTION
    /// Multiresolution support (UNSTABLE) : Set the current level, return false if l >= coarsestLevel
    virtual bool setCurrentLevel(int )
    {
        return false;
    }

    /// Multiresolution support (UNSTABLE)
    virtual void setCoarsestLevel(int ) {}

    /// Multiresolution support (UNSTABLE)
    virtual void setFinestLevel(int ) {}
#endif

    /// @}

    /// @name Variables Setters
    /// @{

    /// Mechanical Degrees-of-Freedom
    virtual void setMechanicalState( BaseObject* )
    { }

    /// Topology
    virtual void setTopology( BaseObject* )
    { }

    /// @}


    /// Test if the given context is an ancestor of this context.
    /// An ancestor is a parent or (recursively) the parent of an ancestor.
    ///
    /// This method is an alias to BaseNode::hasAncestor, so that dynamic
    /// casts are not required to test relationships between contexts.
    virtual bool hasAncestor(const BaseContext* /*context*/) const
    {
        return false;
    }

    /// @name Adding/Removing objects. Note that these methods can fail if the context doesn't support attached objects
    /// @{

    /// Add an object, or return false if not supported
    virtual bool addObject( sptr<BaseObject> /*obj*/ )
    {
        return false;
    }

    /// Remove an object, or return false if not supported
    virtual bool removeObject( sptr<BaseObject> /*obj*/ )
    {
        return false;
    }

    /// @}

    /// @name Visitors.
    /// @{

    /// apply an action
    virtual void executeVisitor( simulation::Visitor*, bool precomputedOrder=false );

    /// Propagate an event
    virtual void propagateEvent( const core::ExecParams* params, Event* );

    /// @}


    /// @name Notifications for graph change listeners
    /// @{

    virtual void notifyAddSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave);
    virtual void notifyRemoveSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave);
    virtual void notifyMoveSlave(core::objectmodel::BaseObject* previousMaster, core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave);

    /// @}

    friend std::ostream SOFA_CORE_API & operator << (std::ostream& out, const BaseContext& c );
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif


