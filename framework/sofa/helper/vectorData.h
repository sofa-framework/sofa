/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_VECTORDATA_H
#define SOFA_HELPER_VECTORDATA_H

#include <string>

#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/DataEngine.h>

#include <sofa/SofaFramework.h>

namespace sofa
{

namespace helper
{

/** A helper class which implements a vector of a variable number of data
 *
 * @todo when the component is a DataEngine, the data are automatically added as  inputs or outputs
 *
 * @author Thomas Lemaire @date 2014
 */
template<class T>
class vectorData : public vector< core::objectmodel::Data<T>* > {

public:
    typedef vector< core::objectmodel::Data<T>* > Inherit;

    vectorData(core::objectmodel::Base* component, std::string const& name, std::string const& help, const bool isInput=true, const T& defaultValue=T())
        : m_component(component)
        , m_name(name)
        , m_help(help)
        , m_isInput(isInput)
        , m_defaultValue(defaultValue)
    { }


    ~vectorData()
    {
        core::DataEngine* componentAsDataEngine = dynamic_cast<core::DataEngine*>(m_component);
        for (unsigned int i=0; i<this->size(); ++i)
        {
            if (componentAsDataEngine!=NULL)
            {
                if(m_isInput) componentAsDataEngine->delInput((*this)[i]);
                else componentAsDataEngine->delOutput((*this)[i]);
            }
            delete (*this)[i];
        }
        this->clear();
    }

    void parseSizeData(sofa::core::objectmodel::BaseObjectDescription* arg, Data<unsigned int>& size)
    {
        const char* p = arg->getAttribute(size.getName().c_str());
        if (p) {
            std::string nbStr = p;
            //            sout << "parse: setting " << size.getName() << "="<<nbStr<<sendl;
            size.read(nbStr);
            resize(size.getValue());
        }
    }


    void parseFieldsSizeData(const std::map<std::string,std::string*>& str, Data<unsigned int>& size)
    {
        std::map<std::string,std::string*>::const_iterator it = str.find(size.getName());
        if (it != str.end() && it->second)
        {
            std::string nbStr = *it->second;
            //            sout << "parseFields: setting "<< size.getName() << "=" <<nbStr<<sendl;
            size.read(nbStr);
            resize(size.getValue());
        }
    }

    void resize(const unsigned int size)
    {
        core::DataEngine* componentAsDataEngine = dynamic_cast<core::DataEngine*>(m_component);
        if (size < this->size()) {
            // some data if size is inferior than current size
            for (unsigned int i=size; i<this->size(); ++i) {
                if (componentAsDataEngine!=NULL)
                {
                    if(m_isInput) componentAsDataEngine->delInput((*this)[i]);
                    else componentAsDataEngine->delOutput((*this)[i]);
                }
                delete (*this)[i];
            }
            Inherit::resize(size);
        }
        for (unsigned int i=this->size(); i<size; ++i) {
            std::ostringstream oname, ohelp;
            oname << m_name << (i+1);
            ohelp << m_help << "(" << (i+1) << ")";
            Data< T >* d = new Data< T >(m_defaultValue, ohelp.str().c_str(), true, false);
            d->setName(oname.str());
            this->push_back(d);
            if (m_component!=NULL)
                m_component->addData(d);
            if (componentAsDataEngine!=NULL)
            {
                if(m_isInput)  componentAsDataEngine->addInput(d);
                else  componentAsDataEngine->addOutput(d);
            }

        }
    }

protected:
    core::objectmodel::Base* m_component;
    std::string m_name, m_help;
    bool m_isInput;
    T m_defaultValue;
};

} // namespace helper

} // namespace sofa

#endif // SOFA_HELPER_VECTORDATA_H
