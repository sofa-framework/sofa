#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/simulation/tree/xml/Element.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace core
{

namespace objectmodel
{

BaseObject::BaseObject()
    : Base()
    , f_listening(dataField( &f_listening, false, "listening", "if true, handle the events, otherwise ignore the events"))
    , f_printLog(dataField( &f_printLog, false, "printLog", "if true, print logs at run-time"))
    , context_(NULL)
/*        , m_isListening(false)
        , m_printLog(false)*/
{}

BaseObject::~BaseObject()
{}

void BaseObject::setContext(BaseContext* n)
{
    context_ = n;
}

const BaseContext* BaseObject::getContext() const
{
    //return (context_==NULL)?BaseContext::getDefault():context_;
    return context_;
}

BaseContext* BaseObject::getContext()
{
    return (context_==NULL)?BaseContext::getDefault():context_;
    //return context_;
}

/// Initialization method called after each graph modification.
void BaseObject::init()
{ }

/// Reset to initial state
void BaseObject::reset()
{ }

void BaseObject::writeState( std::ostream& )
{ }

/// Handle an event
void BaseObject::handleEvent( Event* e )
{
    using namespace simulation::tree;
    cerr<<"BaseObject "<<getName()<<" gets an event"<<endl;
    if( KeypressedEvent* ke = dynamic_cast<KeypressedEvent*>( e ) )
    {
        cerr<<"BaseObject "<<getName()<<" gets a key event: "<<ke->getKey()<<endl;
    }
}

// void BaseObject::setListening( bool b )
// {
//     m_isListening = b;
// }
//
// bool BaseObject::isListening() const
// {
//     return m_isListening;
// }
//
// BaseObject* BaseObject::setPrintLog( bool b )
// {
//     m_printLog = b;
//     return this;
// }
//
// bool BaseObject::printLog() const
// {
//     return m_printLog;
// }

double BaseObject::getTime() const
{
    return getContext()->getTime();
}


} // namespace objectmodel

} // namespace core

} // namespace sofa

