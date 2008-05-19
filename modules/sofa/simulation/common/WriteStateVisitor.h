/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
//
// C++ Interface: WriteStateVisitor
//
// Description:
//
//
// Author: Francois Faure, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_SIMULATION_WRITESTATEACTION_H
#define SOFA_SIMULATION_WRITESTATEACTION_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/simulation/common/Visitor.h>
#include <iostream>

#include <sofa/component/misc/WriteState.h>
#include <sofa/component/misc/ReadState.h>

namespace sofa
{

namespace simulation
{

class WriteStateVisitor: public Visitor
{
public:
    WriteStateVisitor( std::ostream& out );
    virtual ~WriteStateVisitor();

    virtual Result processNodeTopDown( simulation::Node*  );

protected:
    std::ostream& m_out;
};

//Create WriteState component in the graph each time needed
class WriteStateCreator: public Visitor
{
public:
    WriteStateCreator():sceneName(""), counterWriteState(0) {};
    WriteStateCreator(std::string &n, int c=0) {sceneName=n; counterWriteState=c;};
    virtual Result processNodeTopDown( simulation::Node*  );

    void setSceneName(std::string &n) { sceneName = n;}
    void setCounter(int c) {counterWriteState = c;};
protected:
    template< class DataTypes >
    void addWriteState(sofa::core::componentmodel::behavior::MechanicalState< DataTypes > *ms, simulation::Node* gnode);

    std::string sceneName;
    int counterWriteState; //avoid to have two same files if two mechanical objects has the same name
};

//Create ReadState component in the graph each time needed
class ReadStateCreator: public Visitor
{
public:
    ReadStateCreator():sceneName(""), counterReadState(0) {};
    ReadStateCreator(std::string &n, bool i=true, int c=0 ) {sceneName=n; init=i; counterReadState=c;};
    virtual Result processNodeTopDown( simulation::Node*  );

    void setSceneName(std::string &n) { sceneName = n;}
    void setCounter(int c) {counterReadState = c;};
protected:
    template< class DataTypes >
    void addReadState(sofa::core::componentmodel::behavior::MechanicalState< DataTypes > *ms, simulation::Node* gnode);
    bool init;
    std::string sceneName;
    int counterReadState; //avoid to have two same files if two mechanical objects has the same name
};


class WriteStateActivator: public Visitor
{
public:
    WriteStateActivator( bool active):state(active) {};
    virtual Result processNodeTopDown( simulation::Node*  );

    bool getState() const {return state;};
    void setState(bool active) {state=active;};
protected:
    template< class DataTypes >
    void changeStateWriter(sofa::component::misc::WriteState < DataTypes > *ws);

    bool state;
};


class ReadStateModifier: public Visitor
{
public:
    ReadStateModifier( double _time):time(_time) {};
    virtual Result processNodeTopDown( simulation::Node*  );

    double getTime() const {return time;};
    void setTime(double _time) {time=_time;};
protected:
    template< class DataTypes >
    void changeTimeReader(sofa::component::misc::ReadState < DataTypes > *rs) {rs->processReadState(time);};

    double time;
};


}
}

#endif
