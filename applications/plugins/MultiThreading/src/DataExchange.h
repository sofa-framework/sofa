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
#ifndef SOFA_EXCHANGE_DATA_H
#define SOFA_EXCHANGE_DATA_H


#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/State.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/objectmodel/Event.h>

#include <SofaBaseMechanics/MechanicalObject.h>


namespace sofa
{
	namespace core
	{

		class DataExchangeEvent : public sofa::core::objectmodel::Event
		{
		public:

            SOFA_EVENT_H(DataExchangeEvent)

			DataExchangeEvent( double dt )
				: sofa::core::objectmodel::Event()
				, dt(dt) 
			{}

			~DataExchangeEvent() {}

			double getDt() const { return dt; }
			virtual const char* getClassName() const { return "DataExchangeEvent"; }
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
		
			virtual ~DataExchange();
			
			void copyData();

		public:

			 /// Initialization method called at graph creation and modification, during top-down traversal.
			virtual void init();
			

			virtual void handleEvent( core::objectmodel::Event* event );

			template<class T>
			static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
			{
				std::string dataTypeName = defaulttype::DataTypeName<DataTypes>::name();
				// check for the right template
                if ( std::strcmp( dataTypeName.c_str(), arg->getAttribute("template") ) != 0 )
				{
					return false;
					// try to guess from the "from" and "to" data types
				}


				return BaseObject::canCreate(obj, context, arg);
			}



			template<class T>
			static typename T::SPtr create(T*, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
			{	
				std::string fromPath;
				std::string toPath;

				if(arg)
				{
					
					fromPath = arg->getAttribute(std::string("from"), NULL );
					toPath = arg->getAttribute(std::string("to"), NULL );

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


			Data<DataTypes> mSource;
			Data<DataTypes> mDestination;

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





		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Vec3dTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Vec2dTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Vec1dTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Vec6dTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Vec3dTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Rigid3dTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Rigid2dTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Rigid3dTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Rigid3dTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Rigid2dTypes >;

		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Vec3fTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Vec2fTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Vec1fTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Vec6fTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Vec3fTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Rigid3fTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Rigid2fTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Rigid3fTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Rigid3fTypes >;
		//extern template class SOFA_XICATHPLUGIN_API DataExchange< sofa::defaulttype::Rigid2fTypes >;


	}

}

#endif // SOFA_EXCHANGE_DATA_H
