/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef INCRSAP_INL
#define INCRSAP_INL
#include <sofa/component/collision/IncrSAP.h>

namespace sofa
{
namespace component
{
namespace collision
{

inline void SAPBox::init(int boxID){
    for(int i = 0 ; i < 3 ; ++i){
        min[i]->setMin();
        max[i]->setMax();
        min[i]->setBoxID(boxID);
        max[i]->setBoxID(boxID);
    }
}


template <template<class T,class Allocator> class List,template <class T> class Allocator>
void TIncrSAP<List,Allocator>::endBroadPhase()
{
    BroadPhaseDetection::endBroadPhase();

    std::vector<CubeModel*> cube_models;
    cube_models.reserve(_new_cm.size());

    int n = 0;
    for(unsigned int i = 0 ; i < _new_cm.size() ; ++i){
        n += _new_cm[i]->getSize();
        cube_models.push_back(dynamic_cast<CubeModel*>(_new_cm[i]->getPrevious()));
    }

    _boxes.reserve(_boxes.size() + n);
    EndPoint * end_pts = new EndPoint[2*n];
    _to_del.push_back(end_pts);

    int cur_EndPtID = 0;
    int cur_boxID = _boxes.size();
    for(unsigned int i = 0 ; i < cube_models.size() ; ++i){
        CubeModel * cm = cube_models[i];
        for(int j = 0 ; j < cm->getSize() ; ++j){
            EndPoint * min = &end_pts[cur_EndPtID];
            ++cur_EndPtID;
            EndPoint * max = &end_pts[cur_EndPtID];
            ++cur_EndPtID;

            min->setBoxID(cur_boxID);
            max->setBoxID(cur_boxID);
            max->setMax();

            _end_points.push_back(min);
            _end_points.push_back(max);

            _boxes.push_back(SAPBox(Cube(cm,j),min,max));
            ++cur_boxID;
        }
    }

    _new_cm.clear();
}






}
}
}
#endif // INCRSAP_INL
