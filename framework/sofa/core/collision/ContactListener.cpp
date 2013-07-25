#include "ContactListener.h"

#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/collision/ContactManager.h>

#include <sofa/core/CollisionModel.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/CollisionBeginEvent.h>
#include <sofa/simulation/common/CollisionEndEvent.h>


namespace sofa
{
	namespace core
	{

		namespace collision
		{

			SOFA_DECL_CLASS(ContactListener);
			int ContactListenerClass = core::RegisterObject("ContactListener .. ").add< ContactListener >();



			ContactListener::ContactListener(  CollisionModel* collModel1 , CollisionModel* collModel2 )
				: 
				  //mLinkCollisionModel1( initLink("collisionModel1", "first collision model"), collModel1 )
				//, mLinkCollisionModel2( initLink("collisionModel2", "second collision model"), collModel2 )
				 mNarrowPhase(NULL)
			{
				mCollisionModel1 = collModel1;
				mCollisionModel2 = collModel2;
			}

			ContactListener::~ContactListener()
			{
			}

			void ContactListener::init(void)
			{
				helper::vector<ContactManager*> contactManagers;

				mNarrowPhase = getContext()->get<core::collision::NarrowPhaseDetection>();
				if ( mNarrowPhase != NULL )
				{
					// add to the event listening
					f_listening.setValue(true);

				}

			}

			void ContactListener::handleEvent( core::objectmodel::Event* _event )
			{
				if (dynamic_cast<simulation::CollisionBeginEvent *>(_event))
				{
					mContactsVector.clear();
				}

				else if (dynamic_cast<simulation::CollisionEndEvent *>(_event))
				{

					const NarrowPhaseDetection::DetectionOutputMap& detectionOutputsMap = mNarrowPhase->getDetectionOutputs();

					if ( detectionOutputsMap.size() == 0 )
					{
						endContact(NULL);
						return;
					}

					//core::collision::NarrowPhaseDetection::DetectionOutputMap::iterator it = detectionOutputsMap.begin();
					//const helper::vector<DetectionOutput>* detection = dynamic_cast<helper::vector<DetectionOutput>*>(it->second);
					//const TDetectionOutputVector<mCollisionModel1,mCollisionModel2>* detection = dynamic_cast<TDetectionOutputVector*>(it->second);

					if  ( mCollisionModel2 == NULL )
					{
						//// check only one collision model
						for (core::collision::NarrowPhaseDetection::DetectionOutputMap::const_iterator it = detectionOutputsMap.begin(); it!=detectionOutputsMap.end(); ++it )
						{
							const CollisionModel* collMod1 = it->first.first;
							const CollisionModel* collMod2 = it->first.second;

							if ( mCollisionModel1 == collMod1 || mCollisionModel1 == collMod2 )
							{
								if ( const helper::vector<DetectionOutput>* contacts = dynamic_cast<helper::vector<DetectionOutput>*>(it->second) )
								{
									mContactsVector.push_back( contacts );
								}
							}
						}
					}
					else
					{
						// check both collision models
						for (core::collision::NarrowPhaseDetection::DetectionOutputMap::const_iterator it = detectionOutputsMap.begin(); it!=detectionOutputsMap.end(); ++it )
						{
							const CollisionModel* collMod1 = it->first.first;
							const CollisionModel* collMod2 = it->first.second;

							if ( (mCollisionModel1==collMod1 && mCollisionModel2==collMod2) || (mCollisionModel1==collMod2 && mCollisionModel2==collMod1) )
							{
								if ( const helper::vector<DetectionOutput>* contacts = dynamic_cast<helper::vector<DetectionOutput>*>(it->second) )
								{
									mContactsVector.push_back( contacts );
								}
							}
						}

					}

					beginContact(mContactsVector);

				}

			}


		} // namespace collision

	} // namespace core

} // namespace sofa
