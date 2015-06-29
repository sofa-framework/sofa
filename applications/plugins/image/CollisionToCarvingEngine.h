/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_IMAGE_COLLISIONTOCARVINGENGINE_H
#define SOFA_IMAGE_COLLISIONTOCARVINGENGINE_H

#include "initImage.h"
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>


using std::cout;
using std::endl;
namespace sofa
{

namespace component
{

namespace engine
{

using helper::vector;
using cimg_library::CImg;
using cimg_library::CImgList;

/**
 * This class computes carving in an image
 */


template <class _InImageTypes,class _OutImageTypes>
class CollisionToCarvingEngine : public core::DataEngine
{
public:
	typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE2(CollisionToCarvingEngine,_InImageTypes,_OutImageTypes),Inherited);

	typedef _InImageTypes InImageTypes;
    typedef typename InImageTypes::T Ti;
    typedef typename InImageTypes::imCoord imCoordi;
    typedef helper::ReadAccessor<Data< InImageTypes > > raImagei;

    typedef _OutImageTypes OutImageTypes;
    typedef typename OutImageTypes::T To;
    typedef typename OutImageTypes::imCoord imCoordo;
    typedef helper::WriteOnlyAccessor<Data< OutImageTypes > > waImageo;

    typedef SReal Real;
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::WriteOnlyAccessor<Data< TransformType > > waTransform;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;

    typedef vector<double> ParamTypes;
	typedef helper::ReadAccessor<Data< ParamTypes > > raParam;
	typedef sofa::defaulttype::Vec<3,SReal> Vector3;

	Data< InImageTypes > inputImage;
    Data< TransformType > inputTransform;

    Data< OutImageTypes > outputImage;
    Data< TransformType > outputTransform;

	Data< Vector3 > trackedPosition;


    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const CollisionToCarvingEngine<InImageTypes,OutImageTypes>* = NULL) { return InImageTypes::Name()+std::string(",")+OutImageTypes::Name(); }

    CollisionToCarvingEngine()    :   Inherited()
		, inputImage(initData(&inputImage,InImageTypes(),"inputImage",""))
		, inputTransform(initData(&inputTransform,TransformType(),"inputTransform",""))
		, outputImage(initData(&outputImage,OutImageTypes(),"outputImage",""))
		, outputTransform(initData(&outputTransform,TransformType(),"outputTransform",""))
		, trackedPosition(initData(&trackedPosition, Vector3(),"trackedPosition","Position de test pour la collision"))
    {
		inputImage.setReadOnly(true);
        inputTransform.setReadOnly(true);
        outputImage.setReadOnly(true);
        outputTransform.setReadOnly(true);
    }

    virtual ~CollisionToCarvingEngine()
    {
    }

    virtual void init()
    {
		update();
    }

    virtual void reinit() { update(); }

protected:
	
    virtual void update()
    {

		bool updateImage = this->inputImage.isDirty();	// change of input image -> update output image
        bool updateTransform = this->inputTransform.isDirty();	// change of input transform -> update output transform
        if(!updateImage && !updateTransform) {updateImage=true; updateTransform=true;}  // change of parameters -> update all

        raTransform inT(this->inputTransform);
        raImagei in(this->inputImage);

        cleanDirty();

        waImageo out(this->outputImage);
        waTransform outT(this->outputTransform);

        if(in->isEmpty()) return;

        const CImgList<Ti>& inimg = in->getCImgList();
        CImgList<To>& img = out->getCImgList();
        if(updateImage||true) img.assign(inimg);	// copy
        if(updateTransform||true) outT->operator=(inT);	// copy
        cimglist_for(img,l)
            cimg_forXYZ(img(l),x,y,z)
			{
				img(l)(x,y,z)=(To)0.5;
			}
		img(0)(0,0,0) = 0;
		Vector3 valueinimage = trackedPosition.getValue() - inT->getTranslation();
		if(inT->getRotation() == Vector3(0,0,0))
		{
			cout<< "L'absence de rotation n'est pas encore prise en compte" <<endl;
			//img(0)(valueinimage.x, valueinimage.y, valueinimage.z) = 1;
		}
		else{
			cout<< "La collision dans une image rotationné n'est pas encore prise en compte" <<endl;
		}
    }

    void handleEvent(sofa::core::objectmodel::Event *event)
    {
		
        if ( dynamic_cast<simulation::AnimateBeginEvent*>(event))
        { 
			cout<<"test"<<endl;
			update();
		}
		else if( dynamic_cast<sofa::core::objectmodel::MouseEvent*>(event))
		{
			cout<<"mouse"<<endl;
		}		
    }

    virtual void draw(const core::visual::VisualParams* vparams)
    {

    }
};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_DEPTHMAPTOMESHENGINE_H
