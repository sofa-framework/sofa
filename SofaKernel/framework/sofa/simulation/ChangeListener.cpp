/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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

#include <sofa/simulation/ChangeListener.h>

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
