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
#include <sofa/core/collision/Contact.h>
#include <sofa/helper/Factory.inl>

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

Contact::SPtr Contact::Create(const std::string& type, core::CollisionModel* model1, core::CollisionModel* model2, Intersection* intersectionMethod)
{
    std::string::size_type args = type.find('?');
    if (args == std::string::npos)
    {
        return Factory::CreateObject(type,std::make_pair(std::make_pair(model1,model2),intersectionMethod));
    }
    else
    {
        std::string otype(type, 0, args);
        std::cout << otype << " :";
        Contact::SPtr c = Factory::CreateObject(otype,std::make_pair(std::make_pair(model1,model2),intersectionMethod));
        while (args != std::string::npos)
        {
            std::string::size_type next = type.find_first_of("&?",args+1);
            std::string::size_type eq = type.find("=",args+1);
            if (eq != std::string::npos && (next == std::string::npos || eq < next))
            {
                std::string var(type, args+1, eq-args-1);
                std::string val(type, eq+1, (next == std::string::npos ? type.size() : next) - (eq+1));
                std::cout << " " << var << " = " << val;
                std::vector< objectmodel::BaseData* > v = c->findGlobalField( var.c_str() );
                if (v.empty())
                    std::cerr << "ERROR: parameter " << var << " not found in contact type " << otype << std::endl;
                else
                    for (unsigned int i=0; i<v.size(); ++i)
                        v[i]->read(val);
            }
            args = next;
        }
        std::cout << std::endl;
        return c;
    }
}

} // namespace collision

} // namespace core

} // namespace sofa

