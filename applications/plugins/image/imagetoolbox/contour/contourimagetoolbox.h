#ifndef CONTOURIMAGETOOLBOX_H
#define CONTOURIMAGETOOLBOX_H

#include "contourimagetoolboxaction.h"

#include <sofa/component/component.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
//#include <sofa/defaulttype/Vec.h>
#include "ImageTypes.h"


#include "../labelimagetoolbox.h"



#include "initImage.h"




namespace sofa
{

namespace component
{

namespace engine
{

class SOFA_IMAGE_API ContourImageToolBoxNoTemplated: public LabelImageToolBox
{
public:
    SOFA_CLASS(ContourImageToolBoxNoTemplated,LabelImageToolBox);
    
    // image data
    /*typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;*/
    
    ContourImageToolBoxNoTemplated():LabelImageToolBox()
        //, d_image(initData(&d_image,"image",""))
        , d_ip(initData(&d_ip, "imageposition",""))
        , d_p(initData(&d_p, "3Dposition",""))
        , d_axis(initData(&d_axis, (unsigned int)4,"axis",""))
        , d_value(initData(&d_value,"value",""))
        , d_vecCoord(initData(&d_vecCoord,"out",""))
    {
    
    }
    
    virtual void init()
    {
        //addInput(&d_image);
        addOutput(&d_ip);
        addOutput(&d_p);
        addOutput(&d_axis);
        addOutput(&d_value);
        
    }
    
    virtual sofa::gui::qt::LabelImageToolBoxAction* createTBAction(QObject*parent)
    {
        return new sofa::gui::qt::ContourImageToolBoxAction(this,parent);
    }
    
    
    virtual void segmentation()=0;
    
    
    
    
public:
    //Data< ImageTypes > d_image;
    Data<sofa::defaulttype::Vec3d> d_ip;
    Data<sofa::defaulttype::Vec3d> d_p;
    Data<unsigned int> d_axis;
    Data<std::string> d_value;
    Data< helper::vector<sofa::defaulttype::Vec3d> > d_vecCoord;
    
};



template<class _ImageTypes>
class SOFA_IMAGE_API ContourImageToolBox: public ContourImageToolBoxNoTemplated
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ContourImageToolBox,_ImageTypes),ContourImageToolBoxNoTemplated);
    
    typedef ContourImageToolBoxNoTemplated Inherited;
    
    // image data
    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    
    ContourImageToolBox():ContourImageToolBoxNoTemplated()
        , d_image(initData(&d_image,"image",""))
    {
    
    }
    
    virtual void init()
    {
        Inherited::init();
        addInput(&d_image);
        
    }
    
    
    virtual void segmentation()
    {
        //std::cout << "segmentation" << std::endl;
        //raImage rimage = &(this->d_image);
        
        helper::vector<sofa::defaulttype::Vec3d> v = d_vecCoord.getValue();
        
        //std::cout << v.size() << std::endl;
        
        v.push_back(d_p.getValue());
        
        //std::cout << v.size() << std::endl;
        
        d_vecCoord.setValue(v);
    }
    
public:
    Data< ImageTypes > d_image;
    
};


}}}

#endif // ContourImageToolBox_H
