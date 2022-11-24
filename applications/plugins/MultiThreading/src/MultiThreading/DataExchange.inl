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

#include <MultiThreading/DataExchange.h>

namespace sofa::core
{

template <class DataTypes>
DataExchange<DataTypes>::DataExchange( const char* from, const char* to )
    : BaseObject()
    , mSource(initData(&mSource,"from","source object to copy"))
    , mDestination(initData(&mDestination,"to","destination object to copy"))
    , mSourcePtr(nullptr)
    , mDestinationPtr(nullptr)
    , fromPath(from)
    , toPath(to)
    , mSizeInBytes(0)
{
    //f_listening.setValue(true);
}

template <class DataTypes>
DataExchange<DataTypes>::~DataExchange() = default;


template <class DataTypes>
void DataExchange<DataTypes>::copyData()
{
    mDestination.setValue( *mSource.beginEdit() );
}


template <class DataTypes>
void DataExchange<DataTypes>::init()
{
    if ( mSource.getParent() != nullptr  &&  mDestination.getParent() != nullptr)
    {
        f_listening.setValue(true);

        //DataTypes& source = *mSource.beginEdit();
        //DataTypes& destination = *mDestination.beginEdit();
        mSource.beginEdit();
        mDestination.beginEdit();

        parseField( std::string("from"), fromPath );
        parseField( std::string("to"), toPath );

        core::objectmodel::BaseData* tempParent = mSource.getParent();
        tempParent = mDestination.getParent();


        mDestination.setParent( nullptr );
        //mDestination.setReadOnly(true);
        //mDestination.addInput(tempParent);
        //mDestination.setDirtyValue();

        tempParent->setParent( &mDestination, std::string("to") );
        tempParent->setReadOnly(true);
        tempParent->setDirtyValue();

        copyData();
    }

}


template <class DataTypes>
void DataExchange<DataTypes>::handleEvent( core::objectmodel::Event* event )
{
    if ( dynamic_cast<DataExchangeEvent*>(event) != nullptr )
    {
        copyData();
    }
}

} // namespace sofa::core

