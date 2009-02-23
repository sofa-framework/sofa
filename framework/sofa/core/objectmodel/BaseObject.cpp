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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/componentmodel/topology/Topology.h>
#include <sofa/helper/TagFactory.h>
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
    , f_listening(initData( &f_listening, false, "listening", "if true, handle the events, otherwise ignore the events"))
    , f_printLog(initData( &f_printLog, false, "printLog", "if true, print logs at run-time"))
    , f_tagNames(initData( &f_tagNames, "tags", "list of the subsets the objet belongs to"))
    , context_(NULL)
/*        , m_isListening(false)
        , m_printLog(false)*/
{
    sendl.setOutputConsole(f_printLog.beginEdit()); f_printLog.endEdit();
}

BaseObject::~BaseObject()
{}

void BaseObject::parse( BaseObjectDescription* arg )
{
    std::vector< std::string > attributeList;
    arg->getAttributeList(attributeList);
    for (unsigned int i=0; i<attributeList.size(); ++i)
    {
        std::vector< BaseData* > dataModif = findGlobalField(attributeList[i]);
        for (unsigned int d=0; d<dataModif.size(); ++d)
        {
            const char* val = arg->getAttribute(attributeList[i]);
            if (val)
            {
                std::string valueString(val);

                /* test if data is a link */
                if (valueString[0] == '@')
                {
                    std::string objectName;
                    unsigned int j;
                    /* the format of the link after character '@' is objectName.dataName */
                    for(j=1; valueString[j] != '.' && valueString[j] != '\0'; ++j)
                    {
                        objectName.push_back(valueString[j]);
                    }

                    BaseObject* obj;
                    std::string dataName;

                    /* if '.' not found, try to find the data in the current object */
                    if (valueString[j] == '\0')
                    {
                        obj = this;
                        dataName = objectName;
                    }
                    else
                    {
                        obj = getContext()->get<BaseObject>(objectName);

                        if (obj == NULL)
                        {
                            serr<<"could not find object for option "<< attributeList[i] <<": " << objectName << sendl;
                            break;
                        }

                        for(unsigned int j = objectName.length()+2; valueString[j] != '\0'; ++j)
                        {
                            dataName.push_back(valueString[j]);
                        }
                    }

                    BaseData* parentData = obj->findField(dataName);

                    if (parentData == NULL)
                    {
                        serr<<"could not read value for option "<< attributeList[i] <<": " << val << sendl;
                        break;
                    }

                    /* set parent value to the child */
                    if (!dataModif[d]->setParentValue(parentData))
                    {
                        serr<<"could not copy value from parent Data "<< valueString << ". Incompatible Data types" << sendl;
                        break;
                    }
                    parentData->addOutput(dataModif[d]);
                    /* children Data can be modified changing the parent Data value */
                    dataModif[d]->setReadOnly(true);
                    break;
                }

                if( !(dataModif[d]->read( valueString ))) serr<<"could not read value for option "<< attributeList[i] <<": " << val << sendl;
            }
        }
    }
}

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

void BaseObject::init()
{ updateTagList();}

void BaseObject::bwdInit()
{ }

/// Update method called when variables used in precomputation are modified.
void BaseObject::reinit()
{
    updateTagList();
    //sout<<"WARNING: the reinit method of the object "<<this->getName()<<" does nothing."<<sendl;
}

/// Save the initial state for later uses in reset()
void BaseObject::storeResetState()
{ }

/// Reset to initial state
void BaseObject::reset()
{ }

void BaseObject::writeState( std::ostream& )
{ }

/// Called just before deleting this object
/// Any object in the tree bellow this object that are to be removed will be removed only after this call,
/// so any references this object holds should still be valid.
void BaseObject::cleanup()
{ }

/// Handle an event
void BaseObject::handleEvent( Event* /*e*/ )
{
    /*
    serr<<"BaseObject "<<getName()<<" ("<<getTypeName()<<") gets an event"<<sendl;
    if( KeypressedEvent* ke = dynamic_cast<KeypressedEvent*>( e ) )
    {
        serr<<"BaseObject "<<getName()<<" gets a key event: "<<ke->getKey()<<sendl;
    }
    */
}

/// Handle topological Changes from a given Topology
void BaseObject::handleTopologyChange(core::componentmodel::topology::Topology* t)
{
    if (t == this->getContext()->getTopology())
    {
        //	sout << getClassName() << " " << getName() << " processing topology changes from " << t->getName() << sendl;
        handleTopologyChange();
    }
}

/// Handle state Changes from a given Topology
void BaseObject::handleStateChange(core::componentmodel::topology::Topology* t)
{
    if (t == this->getContext()->getTopology())
        handleStateChange();
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


bool BaseObject::hasTag( std::string name)
{
    return (f_tagIds.find( sofa::helper::TagFactory::getID(name) ) != f_tagIds.end() );
}


bool BaseObject::hasTag( unsigned int id)
{
    return ( ( f_tagIds.find(id) ) != f_tagIds.end() );
}


void BaseObject::addTag(std::string sub)
{
    unsigned int id = sofa::helper::TagFactory::getID(sub);
    f_tagIds.insert(id);
    f_tagNames.beginEdit()->push_back(sub);
    f_tagNames.endEdit();
}

void BaseObject::removeTag(std::string sub)
{
    unsigned int id = sofa::helper::TagFactory::getID(sub);
    f_tagIds.erase(id);
    for (sofa::helper::vector<std::string>::iterator it = f_tagNames.beginEdit()->begin(); it!=f_tagNames.getValue().end(); it++)
    {
        if((*it) ==sub)
        {
            f_tagNames.beginEdit()->erase(it);
            f_tagNames.endEdit();
            return;
        }
    }
    f_tagNames.endEdit();
}

void BaseObject::updateTagList()
{
    f_tagIds.clear();
    for (sofa::helper::vector<std::string>::iterator it = f_tagNames.beginEdit()->begin(); it!=f_tagNames.getValue().end(); it++)
        f_tagIds.insert(sofa::helper::TagFactory::getID(*it));

    f_tagNames.endEdit();
}

} // namespace objectmodel

} // namespace core

} // namespace sofa

