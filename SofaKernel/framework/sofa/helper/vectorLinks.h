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
#ifndef SOFA_HELPER_VECTORLINKS_H
#define SOFA_HELPER_VECTORLINKS_H

#include <string>

#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/core/DataEngine.h>
#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{




/** A helper class which implements a vector of a variable number of Links
 *
 * @warning The first index is 1 in the Data name
 *
 * @author Matthieu Nesme @date 2015
 */
template<class LinkType, class OwnerType>
class VectorLinks : public vector< LinkType* > {

public:

    typedef vector< LinkType* > Inherit;

    /// 'dataEngineInOut' is only valid if 'component' is a DataEngine
    VectorLinks(OwnerType* component, std::string const& name, std::string const& help)
        : m_component(component)
        , m_name(name)
        , m_help(help)
    { }

    ~VectorLinks()
    {
        for (unsigned int i=0; i<this->size(); ++i)
        {
            delete (*this)[i];
        }
        this->clear();
    }

    void parseSizeLinks(sofa::core::objectmodel::BaseObjectDescription* arg, Data<unsigned int>& size)
    {
        const char* p = arg->getAttribute(size.getName().c_str());
        if (p) {
            std::string nbStr = p;
            size.read(nbStr);
            resize(size.getValue());
        }
    }


    void parseFieldsSizeLinks(const std::map<std::string,std::string*>& str, Data<unsigned int>& size)
    {
        std::map<std::string,std::string*>::const_iterator it = str.find(size.getName());
        if (it != str.end() && it->second)
        {
            std::string nbStr = *it->second;
            size.read(nbStr);
            resize(size.getValue());
        }
    }

    void resize(const unsigned int size)
    {
        if (size < this->size()) {
            if( size ) Inherit::resize(size);
            else Inherit::clear();
        }
        else
        {
            for (unsigned int i=this->size(); i<size; ++i)
            {
                std::ostringstream oname, ohelp;
                oname << m_name << (i+1);
                ohelp << m_help << "(" << (i+1) << ")";

                LinkType* l = new LinkType();
                l->setName(oname.str());
                l->setHelp(ohelp.str().c_str());

                l->setOwner(m_component);

                this->push_back(l);
            }
        }
    }

protected:
    OwnerType* m_component;
    std::string m_name, m_help;
};

} // namespace helper

} // namespace sofa

#endif // SOFA_HELPER_VECTORDATA_H
