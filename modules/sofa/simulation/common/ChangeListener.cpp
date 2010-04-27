/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/simulation/common/ChangeListener.h>

namespace sofa
{

namespace simulation
{
namespace common
{

/*****************************************************************************************************************/
void ChangeListener::addObject(Node* /*parent*/, core::objectmodel::BaseObject* object)
{
    added.insert(object);
    removed.erase(object);
}


/*****************************************************************************************************************/
void ChangeListener::removeObject(Node* /*parent*/, core::objectmodel::BaseObject* object)
{
    added.erase(object);
    removed.insert(object);
}
/*****************************************************************************************************************/
bool ChangeListener::changed()
{
//       std::cerr<<"print difference"<<std::endl;
//       	std::set<core::objectmodel::BaseObject*>::iterator ibeg=added.begin();
//       	std::set<core::objectmodel::BaseObject*>::iterator iend=added.end();
// 	std::cerr<<"added"<<std::endl;
// 	while(ibeg!=iend){
// 			std::cerr<<(*ibeg)->getName()<<std::endl;
// 		ibeg++;
// 	}
//       	 ibeg=removed.begin();
//       	 iend=removed.end();
// 	std::cerr<<"removed"<<std::endl;
// 	while(ibeg!=iend){
// 			std::cerr<<*ibeg<<std::endl;
// 		ibeg++;
// 	}

    return ((!added.empty())||(!removed.empty()));

}
void ChangeListener::reset()
{
    removed.clear();
    added.clear();
}

}
} //gui
} //sofa
