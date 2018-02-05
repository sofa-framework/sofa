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
#ifndef ENDPOINT_H
#define ENDPOINT_H
#include "config.h"

#include <iostream>
#include <stdio.h>
#include <limits>

namespace sofa
{

namespace component
{

namespace collision
{

class EndPoint{
public:
    EndPoint() : value(std::numeric_limits<double>::max()),data(0){}

    double value;

    void setMax();
    void setMin();

    //CAUTION, always first setBoxID then setMax and setMin
    void setBoxID(int ID);
    int boxID()const;
    bool max()const;
    bool min()const;
    void show()const;

    void setMinAndBoxID(int ID);
    void setMaxAndBoxID(int ID);
private:
    int data;//box ID | MinMax flag;
};

inline void EndPoint::setMinAndBoxID(int ID){
    data = 0;
    data = ID<<1;
}

inline void EndPoint::setMaxAndBoxID(int ID){
    data = 0;
    data = ID<<1;
    ++data;
}

inline void EndPoint::setBoxID(int ID){
    data = ID<<1;
}

inline int EndPoint::boxID()const{
   return data>>1;
}

inline bool EndPoint::min()const{
    return !(data&1);
}

inline bool EndPoint::max() const{
    return data&1;
}

inline void EndPoint::setMax(){
    data |= 1;
}

inline void EndPoint::setMin(){
    data >>= 1;
    data <<= 1;
}

inline void EndPoint::show()const{
    std::cout<<"\tvalue";printf("%lf\n",value);
    std::cout<<"\tdata "<<data<<std::endl;
}

class EndPointID : public EndPoint{
public:
    EndPointID() : EndPoint(), ID(-1){}

    int ID;//index of end point in an list
};


struct CompPEndPoint{
    static double tol(){return (double)(1e-15);}

    bool operator()(const EndPoint * ep1,const EndPoint * ep2)const{
        if(ep1->value != ep2->value){
            return ep1->value < ep2->value;
        }
        else if(ep1->boxID() == ep2->boxID()){
            return ep1->min() && ep2->max();
        }
        else{
            return ep1->boxID() < ep2->boxID();
        }
    }
};

}
}
}
#endif // ENDPOINT_H
