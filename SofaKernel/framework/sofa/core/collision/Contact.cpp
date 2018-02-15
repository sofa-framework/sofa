/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/core/collision/Contact.h>
#include <sofa/helper/Factory.inl>
#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace core
{

namespace collision
{

using namespace sofa::defaulttype;

//template class Factory<std::string, Contact, std::pair<core::CollisionModel*,core::CollisionModel*> >;

Contact::Factory* Contact::Factory::getInstance()
{
    static Factory instance;
    return &instance;
}

Contact::SPtr Contact::Create(const std::string& type, core::CollisionModel* model1, core::CollisionModel* model2, Intersection* intersectionMethod, bool verbose)
{
    std::string::size_type args = type.find('?');
    if (args == std::string::npos)
    {
        return Factory::CreateObject(type,std::make_pair(std::make_pair(model1,model2),intersectionMethod));
    }
    else
    {
        std::string otype(type, 0, args);

		if (verbose)
            msg_info("Contact") << model1->getName() << "-" << model2->getName() << " " << otype << " :";

        Contact::SPtr c = Factory::CreateObject(otype,std::make_pair(std::make_pair(model1,model2),intersectionMethod));

        if( c == NULL ) return c;

        while (args != std::string::npos)
        {
            std::string::size_type next = type.find_first_of("&?",args+1);
            std::string::size_type eq = type.find("=",args+1);
            if (eq != std::string::npos && (next == std::string::npos || eq < next))
            {
                std::string var(type, args+1, eq-args-1);
                std::string val(type, eq+1, (next == std::string::npos ? type.size() : next) - (eq+1));

				if (verbose)
                    msg_info("Contact") << " " << var << " = " << val;

                std::vector< objectmodel::BaseData* > v = c->findGlobalField( var.c_str() );
                if (v.empty() && verbose)
                    msg_error("Contact") << "parameter " << var << " not found in contact type " << otype;
                else
                    for (unsigned int i=0; i<v.size(); ++i)
                        v[i]->read(val);
            }
            args = next;
        }

        return c;
    }
}

} // namespace collision

} // namespace core

} // namespace sofa

