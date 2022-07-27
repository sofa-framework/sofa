#include "labelimagetoolbox.h"

namespace sofa
{

namespace component
{

namespace engine
{

LabelImageToolBox::LabelImageToolBox():   Inherited()
        , d_islinkedToToolBox(initData(false,"islinkedtotoolbox","true if a toobbox use this Label"))
        , d_color(initData(sofa::type::Vec4d(1, 1, 1, 1) ,"color",""))
    {
        d_islinkedToToolBox.setReadOnly(true);
    }
    
}
}
}
