#pragma once

#include "Base.h"
#include "BaseContext.h"

namespace Sofa
{

namespace Abstract
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



protected:
    BaseContext* context_;
    /*        bool m_isListening;
            bool m_printLog;*/
};

} // namespace Abstract

} // namespace Sofa



