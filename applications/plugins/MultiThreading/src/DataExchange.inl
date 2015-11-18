#ifndef _SofaDataExchange_inl__
#define _SofaDataExchange_inl__

#include "DataExchange.h"


namespace sofa
{

	namespace core
	{

		template <class DataTypes>
		DataExchange<DataTypes>::DataExchange( const char* from, const char* to )
			: BaseObject()
			, mSource(initData(&mSource,"from","source object to copy"))
			, mDestination(initData(&mDestination,"to","destination object to copy"))
			, mSourcePtr(NULL)
			, mDestinationPtr(NULL)
			, fromPath(from)
			, toPath(to)
			, mSizeInBytes(0)
		{
			//f_listening.setValue(true);
		}

		template <class DataTypes>
		DataExchange<DataTypes>::~DataExchange() 
		{
		}


		template <class DataTypes>
		void DataExchange<DataTypes>::copyData()  
		{ 
			mDestination.setValue( *mSource.beginEdit() );
		}


		template <class DataTypes>
		void DataExchange<DataTypes>::init()
		{
			
			if ( mSource.getParent() != NULL  &&  mDestination.getParent() != NULL)
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
				

				mDestination.setParent( NULL );
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
			if ( dynamic_cast<DataExchangeEvent*>(event) != NULL )
			{
				copyData();
			}
		}




	} // namespace core


}  // namespace sofa

#endif // _SofaDataExchange_inl__
