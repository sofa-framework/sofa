#ifndef SOFA_CORE_OBJECTMODEL_BASEOBJECT_H
#define SOFA_CORE_OBJECTMODEL_BASEOBJECT_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

class Event;

/** Base class for simulation objects.
Each simulation object is related to a context, which gives access to all available external data.
It is able to process events, if listening enabled (default is false).
*/
class BaseObject : public virtual Base
{
public:
    BaseObject();

    virtual ~BaseObject();

    /**@name context
     */
    ///@{
    void setContext(BaseContext* n);

    const BaseContext* getContext() const;

    BaseContext* getContext();
    ///@}

    /**@name control
        Basic state control
     */
    ///@{
    /// Initialization method called after each graph modification.
    virtual void init();

    /// Reset to initial state
    virtual void reset();

    /// Write current state to the given output stream
    virtual void writeState( std::ostream& out );

    ///@}

    /**@name events
    Methods related to Event processing
     */
    ///@{

    DataField<bool> f_listening;

    /// Handle an event
    virtual void handleEvent( Event* );
    ///@}

    /**@name debug
    Methods related to debugging
     */
    ///@{
    DataField<bool> f_printLog;
    ///@}

    /**@name data access
    Access to external data
     */
    ///@{
    /// Current time
    double getTime() const;
    ///@}

    /// Pre-construction check method called by ObjectFactory.
    template<class T>
    static bool canCreate(T*& /*obj*/, BaseContext* /*context*/, BaseObjectDescription* /*arg*/)
    {
        return true;
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static void create(T*& obj, BaseContext* /*context*/, BaseObjectDescription* arg)
    {
        obj = new T;
        obj->parseFields(arg->getAttributeMap());
    }

protected:
    BaseContext* context_;
    /*        bool m_isListening;
            bool m_printLog;*/
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
