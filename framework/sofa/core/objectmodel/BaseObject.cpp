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
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/topology/Topology.h>
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
    , l_context(initLink("context","Graph Node containing this object (or BaseContext::getDefault() if no graph is used"))
    , l_slaves(initLink("slaves","Sub-objects used internally by this object"))
    , l_master(initLink("master","NULL for regular objects, or master object for which this object is one sub-objects"))
    //, context_(NULL)
#ifdef SOFA_SMP
    ,partition_(NULL)
#endif
/*        , m_isListening(false)
, m_printLog(false)*/
{
    l_context.setValidator(&sofa::core::objectmodel::BaseObject::changeContextLink);
    l_context.set(BaseContext::getDefault());
    l_slaves.setValidator(&sofa::core::objectmodel::BaseObject::changeSlavesLink);
    f_listening.setAutoLink(false);
}

BaseObject::~BaseObject()
{
    assert(l_master.get() == NULL); // an object that is still a slave should not be able to be deleted, as at least one smart pointer points to it
    for(VecSlaves::const_iterator iSlaves = l_slaves.begin(); iSlaves != l_slaves.end(); ++iSlaves)
    {
        (*iSlaves)->l_master.reset();
    }
}

void BaseObject::parse( BaseObjectDescription* arg )
{
    std::vector< std::string > attributeList;
    arg->getAttributeList(attributeList);

    for (unsigned int i=0; i<attributeList.size(); ++i)
    {
        if (attributeList[i] == "src")
        {
            // Parse attribute 'src' for new MeshLoader architecture.
            const char* val = arg->getAttribute(attributeList[i]);
            std::string valueString(val);

            if(!val)
            {
                serr<<"ERROR: Missing argument for 'src' attribute in object: "<< this->getName() << sendl;
                break;
            }

            if (valueString[0] != '@')
            {
                serr<<"ERROR: 'src' attribute value should be a link using '@' in object "<< this->getName() << sendl;
                break;
            }

            setSrc(valueString, &attributeList);

            continue;
        }


        std::vector< BaseData* > dataModif = findGlobalField(attributeList[i]);
        for (unsigned int d=0; d<dataModif.size(); ++d)
        {
            const char* val = arg->getAttribute(attributeList[i]);
            if (val)
            {
                std::string valueString(val);

                // test if data is a link and can be linked
                if (valueString[0] == '@' && dataModif[d]->canBeLinked())
                {
                    dataModif[d]->setLinkPath(valueString);
                    std::size_t posPath = valueString.rfind('/');
                    if (posPath == std::string::npos) posPath = 0;
                    std::size_t posDot = valueString.rfind('.');
                    std::string objectName;
                    std::string dataName;

                    BaseObject* obj = NULL;

                    /* if '.' not found, try to find the data in the current object */
                    if (posPath == 0 && posDot == std::string::npos)
                    {
                        obj = this;
                        objectName = this->getName();
                        dataName = valueString;
                    }
                    else
                    {
                        if (posDot == std::string::npos) // no data specified, look for one with the same name
                        {
                            objectName = valueString;
                            dataName = attributeList[i];
                        }
                        else
                        {
                            objectName = valueString.substr(1,posDot-1);
                            dataName = valueString.substr(posDot+1);
                        }

                        if (objectName[0] == '[')
                        {
                            if (objectName[objectName.size()-1] != ']')
                            {
                                serr<<"ERROR: Missing ']' in at the end of "<< objectName << sendl;
                                continue;
                            }

                            objectName = objectName.substr(1, objectName.size()-2);

                            if (objectName.empty())
                            {
                                serr<<"ERROR: Missing object level between [] in : " << val << sendl;
                                continue;
                            }

                            int objectLevel = atoi(objectName.c_str());
                            helper::vector<BaseObject*> objects;
                            getContext()->get<BaseObject>(&objects, BaseContext::Local);

                            helper::vector<BaseObject*>::iterator it;

                            for(it = objects.begin(); it != objects.end(); ++it)
                            {
                                if ((*it) == this)
                                {
                                    it += objectLevel;
                                    if ((*it) != NULL)
                                    {
                                        obj = (*it);
                                    }
                                    break;
                                }
                            }
                        }
                        else
                        {
                            obj = getContext()->get<BaseObject>(objectName);
                        }

                        if (obj == NULL)
                        {
                            serr<<"could not find object for option "<< attributeList[i] <<": " << objectName << sendl;
                            continue;
                        }
                    }

                    BaseData* parentData = obj->findField(dataName);

                    if (parentData == NULL)
                    {
                        serr<<"could not find parent Data for option "<< attributeList[i] <<": " << val << sendl;
                        continue;
                    }

                    /* set parent value to the child */
                    if (!dataModif[d]->setParent(parentData))
                    {
                        serr<<"could not copy value from parent Data "<< valueString << ". Incompatible Data types" << sendl;
                        continue;
                    }
                    else
                        sout<<"Link from parent Data " << valueString << " (" << parentData->getValueTypeInfo()->name() << ") to Data " << attributeList[i] << "(" << dataModif[d]->getValueTypeInfo()->name() << ") OK" << sendl;
                    /* children Data cannot be modified changing the parent Data value */
                    dataModif[d]->setReadOnly(true);
                    continue;
                }

                if( !(dataModif[d]->read( valueString ))) serr<<"could not read value for option "<< attributeList[i] <<": " << val << sendl;
            }
        }
    }
}

void BaseObject::setSrc(const std::string &valueString, std::vector< std::string > *attributeList)
{
    BaseObject* loader = NULL;

    std::size_t posAt = valueString.rfind('@');
    if (posAt == std::string::npos) posAt = 0;
    std::string objectName;

    objectName = valueString.substr(posAt+1);
    loader = getContext()->get<BaseObject>(objectName);

    setSrc(valueString, loader, attributeList);
}

void BaseObject::setSrc(const std::string &valueString, const BaseObject *loader, std::vector< std::string > *attributeList)
{
    BaseObject* obj = this;

    std::multimap < std::string, BaseData*> dataLoaderMap(loader->m_aliasData);
    std::multimap < std::string, BaseData*>::iterator it_map;

    //for (unsigned int j = 0; j<loader->m_fieldVec.size(); ++j)
    //{
    //	dataLoaderMap.insert (std::pair<std::string, BaseData*> (loader->m_fieldVec[j].first, loader->m_fieldVec[j].second));
    //}

    if (attributeList != 0)
    {
        for (unsigned int j = 0; j<attributeList->size(); ++j)
        {
            it_map = dataLoaderMap.find ((*attributeList)[j]);
            if (it_map != dataLoaderMap.end())
                dataLoaderMap.erase (it_map);
        }
    }

    // -- Temporary patch, using exceptions. TODO: use a flag to set Data not to be automatically linked. --
    //{
    it_map = dataLoaderMap.find ("type");
    if (it_map != dataLoaderMap.end())
        dataLoaderMap.erase (it_map);

    it_map = dataLoaderMap.find ("filename");
    if (it_map != dataLoaderMap.end())
        dataLoaderMap.erase (it_map);
    //}


    for (it_map = dataLoaderMap.begin(); it_map != dataLoaderMap.end(); ++it_map)
    {
        BaseData* data = obj->findField( (*it_map).first );
        if (data != NULL)
        {
            if (!(*it_map).second->isAutoLink())
            {
                sout << "Disabling autolink for Data " << data->getName() << sendl;
            }
            else
            {
                //serr << "Autolinking Data " << data->getName() << sendl;
                std::string linkPath = valueString+"."+(*it_map).first;
                data->setLinkPath(linkPath);
                data->setParent( (*it_map).second);
            }
        }
    }
}

/*
void BaseObject::setContext(BaseContext* n)
{
	//context_ = n;
    l_context.set(n);
}
*/

#ifdef SOFA_SMP

void BaseObject::setPartition(Iterative::IterativePartition* p)
{
    partition_=p;
}
Iterative::IterativePartition*  BaseObject::getPartition()
{
    if(partition_)
        return partition_;
    if(getContext()&&getContext()->is_partition())
        return getContext()->getPartition();
    return 0;
}

#endif
const BaseContext* BaseObject::getContext() const
{
    //return (context_==NULL)?BaseContext::getDefault():context_;
    //return context_;
    return l_context.get();
}

BaseContext* BaseObject::getContext()
{
    //return (context_==NULL)?BaseContext::getDefault():context_;
    //return context_;
    return l_context.get();
}

const BaseObject* BaseObject::getMaster() const
{
    return l_master.get();
}

BaseObject* BaseObject::getMaster()
{
    return l_master.get();
}

const BaseObject::VecSlaves& BaseObject::getSlaves() const
{
    return l_slaves.getValue();
}

void BaseObject::addSlave(BaseObject::SPtr s)
{
    BaseObject::SPtr previous = s->getMaster();
    if (previous == this) return;
    if (previous)
        previous->l_slaves.remove(s);
    l_slaves.add(s);
    if (previous)
        this->getContext()->notifyMoveSlave(previous.get(), this, s.get());
    else
        this->getContext()->notifyAddSlave(this, s.get());
}

void BaseObject::removeSlave(BaseObject::SPtr s)
{
    if (l_slaves.remove(s))
    {
        this->getContext()->notifyRemoveSlave(this, s.get());
    }
}

/// Copy the source aspect to the destination aspect for each Data in the component.
void BaseObject::copyAspect(int destAspect, int srcAspect)
{
    Base::copyAspect(destAspect, srcAspect);
    for(VecSlaves::const_iterator iSlaves = l_slaves.begin(); iSlaves != l_slaves.end(); ++iSlaves)
    {
        (*iSlaves)->copyAspect(destAspect, srcAspect);
    }
}

/// Release memory allocated for the specified aspect.
void BaseObject::releaseAspect(int aspect)
{
    Base::releaseAspect(aspect);
    for(VecSlaves::const_iterator iSlaves = l_slaves.begin(); iSlaves != l_slaves.end(); ++iSlaves)
    {
        (*iSlaves)->releaseAspect(aspect);
    }
}

void BaseObject::init()
{
#ifdef SOFA_SMP
    if(!context_||!context_->is_partition())
        setPartition(new Iterative::IterativePartition());
#endif
}

void BaseObject::bwdInit()
{
}

/// Update method called when variables used in precomputation are modified.
void BaseObject::reinit()
{
    //sout<<"WARNING: the reinit method of the object "<<this->getName()<<" does nothing."<<sendl;
}

/// Save the initial state for later uses in reset()
void BaseObject::storeResetState()
{ }

/// Reset to initial state
void BaseObject::reset()
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
void BaseObject::handleTopologyChange(core::topology::Topology* t)
{
    if (t == this->getContext()->getTopology())
    {
        //	sout << getClassName() << " " << getName() << " processing topology changes from " << t->getName() << sendl;
        handleTopologyChange();
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

