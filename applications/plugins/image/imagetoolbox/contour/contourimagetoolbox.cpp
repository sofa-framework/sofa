

#include <sofa/core/ObjectFactory.h>

#include "contourimagetoolbox.h"
#include "ImageTypes.h"


namespace sofa
{

namespace component
{

namespace engine
{

SOFA_DECL_CLASS(ContourImageToolBox)

int ContourImageToolBox_Class = core::RegisterObject("ContourImageToolBox")
.add< ContourImageToolBox<sofa::defaulttype::ImageUS> >()
.addLicense("LGPL")
.addAuthor("Vincent Majorczyk");

}}}
