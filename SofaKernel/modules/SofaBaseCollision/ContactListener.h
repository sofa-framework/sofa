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
#ifndef SOFA_CONTACT_LISTENER_H
#define SOFA_CONTACT_LISTENER_H
#include "config.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/collision/Contact.h>
//#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/helper/vector.h>

//#include <sofa/core/CollisionModel.h>

namespace sofa
{
	namespace core
	{

		// forward declaration
		class CollisionModel;

		namespace collision
		{

			// forward declaration
			class NarrowPhaseDetection;
			


			class SOFA_BASE_COLLISION_API ContactListener : public virtual core::objectmodel::BaseObject
			{
			public:

				SOFA_CLASS(ContactListener, core::objectmodel::BaseObject);

			protected:

				ContactListener( CollisionModel* collModel1 = NULL, CollisionModel* collModel2 = NULL );

				virtual ~ContactListener();

				// DetectionOutput iterators
				typedef helper::vector<const helper::vector<DetectionOutput>* >::const_iterator ContactVectorsIterator;
				typedef helper::vector<DetectionOutput>::const_iterator ContactsIterator;

				virtual void beginContact(const helper::vector<const helper::vector<DetectionOutput>* >& ) {}

				virtual void endContact(void*) {}


			protected:

				const CollisionModel* mCollisionModel1;
				const CollisionModel* mCollisionModel2;

				//// are these SingleLinks necessary ? they are used only in canCreate(...) and create functions(...)
				//SingleLink<ContactListener, CollisionModel, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> mLinkCollisionModel1;

				///// Pointer to the next (finer / lower / child level) CollisionModel in the hierarchy.
				//SingleLink<ContactListener, CollisionModel, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> mLinkCollisionModel2;


			private:

				helper::vector<const helper::vector<DetectionOutput>* > mContactsVector;

				core::collision::NarrowPhaseDetection* mNarrowPhase;


			public:				

				virtual void init(void);			

				virtual void handleEvent( core::objectmodel::Event* event );



				template<class T>
				static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
				{
					core::CollisionModel* collModel1 = NULL;
					core::CollisionModel* collModel2 = NULL;

					std::string collModelPath1;
					std::string collModelPath2;

					if (arg->getAttribute("collisionModel1"))
						collModelPath1 = arg->getAttribute("collisionModel1");
					else
						collModelPath1 = "";

					context->findLinkDest(collModel1, collModelPath1, NULL);

					if (arg->getAttribute("collisionModel2"))
						collModelPath2 = arg->getAttribute("collisionModel2");
					else
						collModelPath2 = "";

					context->findLinkDest(collModel2, collModelPath2, NULL);

					if (collModel1 == NULL && collModel2 == NULL )
					{
						context->serr << "Creation of " << className(obj) << 
							" CollisonListener failed because no Collision Model links are found: \"" << collModelPath1
							<< "\" and \"" << collModelPath2 << "\" " << context->sendl;
						return false;
					}

					return BaseObject::canCreate(obj, context, arg);
				}



				template<class T>
				static typename T::SPtr create(T* , core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
				{	
					CollisionModel* collModel1 = NULL;
					CollisionModel* collModel2 = NULL;

					std::string collModelPath1;
					std::string collModelPath2;
					
					if(arg)
					{
						collModelPath1 = arg->getAttribute(std::string("collisionModel1"), NULL );
						collModelPath2 = arg->getAttribute(std::string("collisionModel2"), NULL );

						// now 3 cases
						if ( strcmp( collModelPath1.c_str(),"" ) != 0  )
						{
							context->findLinkDest(collModel1, collModelPath1, NULL);
							
							if ( strcmp( collModelPath2.c_str(),"" ) != 0 )
							{
								context->findLinkDest(collModel2, collModelPath2, NULL);
							}
						}
						else
						{
							context->findLinkDest(collModel1, collModelPath2, NULL);
						}						

						
					}

					typename T::SPtr obj = sofa::core::objectmodel::New<T>( collModel1, collModel2 );

					//if ( obj )
					//{
					//	obj->mLinkCollisionModel1.setPath( collModelPath1 );
					//	obj->mLinkCollisionModel2.setPath( collModelPath2 );
					//}

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
			

			};

		} // namespace collision

	} // namespace core

} // namespace sofa

#endif // SOFA_CONTACT_LISTENER_H
