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
#ifndef SOFA_IMAGE_COLLISIONTOCARVINGENGINE_H
#define SOFA_IMAGE_COLLISIONTOCARVINGENGINE_H

#include <image/config.h>
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
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>


namespace sofa
{

namespace component
{

namespace engine
{


/**
 * This class computes carving in an image
 */


template <class _InImageTypes,class _OutImageTypes>
class CollisionToCarvingEngine : public core::DataEngine
{
public:
	// ------------ Typedefs ------------------------------------------------- 
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

    typedef helper::vector<double> ParamTypes;
	typedef helper::ReadAccessor<Data< ParamTypes > > raParam;
	typedef sofa::defaulttype::Vec<3,SReal> Vector3;
	// -------- Datas ----------------------------
	Data< InImageTypes > inputImage;
    Data< TransformType > inputTransform;

    Data< OutImageTypes > outputImage;
    Data< TransformType > outputTransform;

	Data< Vector3 > trackedPosition;

	// ------ Parameters ---------------------
	raImagei* in;
	raTransform* inT;
	waImageo* out;
	waTransform* outT;

    virtual std::string getTemplateName() const    override { return templateName(this);    }
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
		in=NULL;
		inT=NULL;
		out=NULL;
		outT=NULL;
    }

    virtual ~CollisionToCarvingEngine()
    {
		delete in;
		delete inT;
		delete out;
		delete outT;
    }

    virtual void init() override
    {
		//cout<<"init"<<endl;
		addInput(&inputImage);
        addInput(&inputTransform);
        addOutput(&outputImage);
        addOutput(&outputTransform);
		setDirtyValue();
    }

    virtual void reinit() override { update(); }

protected:
	
    virtual void update() override
    {
		
		bool updateImage = this->inputImage.isDirty();	// change of input image -> update output image
        bool updateTransform = this->inputTransform.isDirty();	// change of input transform -> update output transform
		
		if(in==NULL){in = new raImagei(this->inputImage);}
		if(inT==NULL){inT = new raTransform(this->inputTransform);}
		if(out==NULL){out = new waImageo(this->outputImage);}
		if(outT==NULL){outT = new waTransform(this->outputTransform);}

		cleanDirty();
	
		if((*in)->isEmpty()) return;

        const cimg_library::CImgList<Ti>& inimg = (*in)->getCImgList();
        cimg_library::CImgList<To>& img = (*out)->getCImgList();
        if(updateImage) img.assign(inimg);	// copy
        if(updateTransform) (*outT)->operator=(*inT);	// copy
		
		//cout << this->inputImage <<endl;
		if(updateImage || updateTransform)
		{
			cimglist_for(img,l)
				cimg_forXYZ(img(l),x,y,z)
				{
                    img(l)(x,y,z)=(To)inimg(l)(x,y,z);
				}
			img(0)(0,0,0) = 0;
		}
		Vector3 valueinimage = trackedPosition.getValue() - (*inT)->getTranslation();
		Vector3 scale = (*outT)->getScale();
		if((*outT)->getRotation() == Vector3(0,0,0))
		{
			//cout<< "L'absence de rotation n'est pas encore prise en compte" <<endl;
			
			if((*out)->isInside(valueinimage.x()/scale.x(), valueinimage.y()/scale.y(), valueinimage.z()/scale.z()))
			{
                img(0)(valueinimage.x()/scale.x(), valueinimage.y()/scale.y(), valueinimage.z()/scale.z()) = (To)1;
			}
		}
		else{
			//cout<< "La collision dans une image rotationné n'est pas encore prise en compte" <<endl;
		}
		if (updateTransform) (*outT)->update(); // update internal data
    }

    void handleEvent(sofa::core::objectmodel::Event *event) override
    {
		
        if ( simulation::AnimateBeginEvent::checkEventType(event) )
        { 
			//cout<<"test"<<endl;
			update();
		}
    }

    virtual void draw(const core::visual::VisualParams* /*vparams*/) override
    {

    }
};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_DEPTHMAPTOMESHENGINE_H
