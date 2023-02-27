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

#include <MultiThreading/config.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/State.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/objectmodel/Event.h>

namespace sofa::core
{

class DataExchangeEvent : public sofa::core::objectmodel::Event
{
public:

    SOFA_EVENT_H(DataExchangeEvent)

    DataExchangeEvent( double dt )
        : sofa::core::objectmodel::Event()
          , dt(dt)
    {}

    ~DataExchangeEvent() override {}

    double getDt() const { return dt; }
    static inline const char* GetClassName() { return "DataExchangeEvent"; }
protected:
    double dt;
};


template <class DataTypes>
class DataExchange : public virtual objectmodel::BaseObject
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DataExchange, DataTypes ), objectmodel::BaseObject);

    //typedef typename DataTypes::Real        Real;
    //typedef typename DataTypes::Coord       Coord;
    //typedef typename DataTypes::VecCoord    VecCoord;

protected:

    DataExchange( const char* from, const char* to );

    ~DataExchange() override;

    void copyData();

public:

    /// Initialization method called at graph creation and modification, during top-down traversal.
    void init() override;


    void handleEvent( core::objectmodel::Event* event ) override;

    static std::string GetCustomTemplateName()
    {
        return sofa::defaulttype::DataTypeName<DataTypes>::name();
    }

    template<class T>
    static typename T::SPtr create(T*, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        std::string fromPath;
        std::string toPath;

        if(arg)
        {
            fromPath = arg->getAttribute(std::string("from"), "" );
            toPath = arg->getAttribute(std::string("to"), "" );
        }

        //context->findLinkDest(stout, fromPath, NULL);
        //context->findLinkDest(stout, toPath, NULL);

        typename T::SPtr obj = sofa::core::objectmodel::New<T>(fromPath.c_str(), toPath.c_str() );
        if (context)
        {
            context->addObject(obj);
        }
        if (arg)
        {
            obj->parse(arg);
        }
        return obj;
    }


    Data<DataTypes> mSource;      ///< source object to copy
    Data<DataTypes> mDestination; ///< destination object to copy

private:


    /// source
    //SingleLink< DataExchange<DataTypes>, Data<DataTypes>, BaseLink::FLAG_DATALINK|BaseLink::FLAG_DUPLICATE> mSourceObject;
    /// dest
    //SingleLink< DataExchange<DataTypes>, Data<DataTypes>, BaseLink::FLAG_DATALINK|BaseLink::FLAG_DUPLICATE> mDestinationObject;



    DataTypes* mSourcePtr;
    DataTypes* mDestinationPtr;
    //VecCoord mDataCopy;

    std::string fromPath;
    std::string toPath;
    //core::objectmodel::BaseObjectDescription* desc;

    std::size_t mSizeInBytes;

};




#if !defined(SOFA_MULTITHREADING_PLUGIN_DATAEXCHANGE_CPP)
extern template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::vector<sofa::type::Vec3d> >;
extern template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::vector<sofa::type::Vec2d> >;
extern template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::vector<double> >;
extern template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::Vec3d >;
extern template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< double >;

extern template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::vector<sofa::type::Vec3f> >;
extern template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::vector<sofa::type::Vec2f> >;
extern template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::vector<float> >;
extern template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::Vec3f >;
extern template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< float >;

extern template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::vector<int> >;
extern template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< sofa::type::vector<unsigned int> >;
extern template class SOFA_MULTITHREADING_PLUGIN_API DataExchange< bool >;
#endif

}

