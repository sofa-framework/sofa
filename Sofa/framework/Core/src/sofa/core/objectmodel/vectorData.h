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
#include <sofa/core/config.h>

#include <string>

#include <sofa/type/vector.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/DataEngine.h>

#include <sofa/helper/StringUtils.h>
using sofa::helper::getAStringCopy ;

namespace sofa::core::objectmodel
{

enum class DataEngineDataType {DataEngineNothing,DataEngineInput,DataEngineOutput} ;

/** A helper class which implements a vector of a variable number of Data
 *
 * When the owner component is a DataEngine, the Data can be automatically added as inputs or outputs
 *
 * @warning The first index is 1 in the Data name
 *
 * @author Thomas Lemaire @date 2014
 */
template<class T>
class vectorData : public type::vector< core::objectmodel::Data<T>* > {

public:

    typedef type::vector< core::objectmodel::Data<T>* > Inherit;
    using size_type = typename Inherit::size_type;

    /// 'dataEngineInOut' is only valid if 'component' is a DataEngine
    vectorData(core::objectmodel::Base* component, std::string const& name, std::string const& help, DataEngineDataType dataEngineDataType= DataEngineDataType::DataEngineNothing, const T& defaultValue=T())
        : m_component(component)
        , m_name(name)
        , m_help(help)
        , m_dataEngineDataType(dataEngineDataType)
        , m_defaultValue(defaultValue)
    { }

    ~vectorData()
    {
        if( m_dataEngineDataType!= DataEngineDataType::DataEngineNothing )
        {
            if( core::DataEngine* componentAsDataEngine = m_component->toDataEngine() )
            {
                for (unsigned int i=0; i<this->size(); ++i)
                {
                    if(m_dataEngineDataType== DataEngineDataType::DataEngineInput) componentAsDataEngine->delInput((*this)[i]);
                    else if(m_dataEngineDataType== DataEngineDataType::DataEngineOutput) componentAsDataEngine->delOutput((*this)[i]);
                }
            }
        }
        for (unsigned int i=0; i<this->size(); ++i)
        {
            delete (*this)[i];
        }
        this->clear();
    }

    void parseSizeData(sofa::core::objectmodel::BaseObjectDescription* arg, Data<unsigned int>& size)
    {
        const char* p = arg->getAttribute(size.getName().c_str());
        if (p) {
            const std::string nbStr = p;
            size.read(nbStr);
            resize(size.getValue());
        }
    }


    void parseFieldsSizeData(const std::map<std::string,std::string*>& str, Data<unsigned int>& size)
    {
        const std::map<std::string,std::string*>::const_iterator it = str.find(size.getName());
        if (it != str.end() && it->second)
        {
            const std::string nbStr = *it->second;
            size.read(nbStr);
            resize(size.getValue());
        }
    }

    void resize(const unsigned int count)
    {
        core::DataEngine* componentAsDataEngine = m_dataEngineDataType!= DataEngineDataType::DataEngineNothing ? m_component->toDataEngine() : nullptr;

        if (count < this->size())
        {
            // removing some data if size is inferior than current size
            for (size_type i = count; i < this->size(); ++i)
            {
                if (componentAsDataEngine!=nullptr)
                {
                    if(m_dataEngineDataType== DataEngineDataType::DataEngineInput) componentAsDataEngine->delInput((*this)[i]);
                    else if(m_dataEngineDataType== DataEngineDataType::DataEngineOutput) componentAsDataEngine->delOutput((*this)[i]);
                }
                delete (*this)[i];
            }
            if( count ) Inherit::resize(count);
            else Inherit::clear();
        }
        else
        {
            for (size_type i=this->size(); i<count; ++i)
            {
                std::ostringstream oname, ohelp;
                oname << m_name << (i+1);
                ohelp << m_help << "(" << (i+1) << ")";
                Data< T >* d = new Data< T >(m_defaultValue, getAStringCopy(ohelp.str().c_str()), true, false);
                d->setName(oname.str());
                this->push_back(d);
                if (m_component!=nullptr)
                    m_component->addData(d);
                if (componentAsDataEngine!=nullptr)
                {
                    if(m_dataEngineDataType== DataEngineDataType::DataEngineInput) componentAsDataEngine->addInput(d);
                    else if(m_dataEngineDataType== DataEngineDataType::DataEngineOutput) componentAsDataEngine->addOutput(d);
                }
            }
        }
    }

    /// merging several Data from a VectorData into a large Data (of the same type)
    static void merge(Data<T>& outputData, const vectorData<T>& vectorData)
    {
        const size_t nbInput = vectorData.size();
        size_t nbElems = 0;

        for( size_t i=0 ; i<nbInput ; ++i )
            nbElems += vectorData[i]->getValue().size();

        helper::WriteOnlyAccessor< Data<T> > out = outputData;
        out.clear();
        out.reserve(nbElems);
        for( size_t i=0 ; i<nbInput ; ++i )
        {
            helper::ReadAccessor< Data<T> > in = vectorData[i];
            for( size_t j=0, jend=in.size() ; j<jend ; ++j )
                out.push_back(in[j]);
        }
    }

protected:
    core::objectmodel::Base* m_component;
    std::string m_name, m_help;
    DataEngineDataType m_dataEngineDataType; ///< only valid if m_component is a DataEngine
    T m_defaultValue;
};

} // namespace sofa::core::objectmodel
