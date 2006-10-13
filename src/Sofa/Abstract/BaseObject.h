#pragma once

#include "Base.h"
#include "BaseContext.h"
#include "FieldContainer.h"

namespace Sofa
{

namespace Abstract
{
class Event;

/** Base class for simulation objects.
Each simulation object is related to a context, which gives access to all available external data.
It is able to process events, if listening enabled (default is false).
*/
class BaseObject : public virtual Base, public virtual FieldContainer
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
    /// if true, then handle the incoming events, else ignore them
    void setListening( bool );

    /// if true, then handle the incoming events, else ignore them
    bool isListening() const;

    /// Handle an event
    virtual void handleEvent( Event* );
    ///@}

    /**@name debug
    Methods related to debugging
     */
    ///@{
    /// if true, print logs at run-time
    BaseObject* setPrintLog( bool );

    /// if true, print logs at run-time
    bool printLog() const;
    ///@}



protected:
    BaseContext* context_;
    bool m_isListening;
    bool m_printLog;
};

} // namespace Abstract

} // namespace Sofa



