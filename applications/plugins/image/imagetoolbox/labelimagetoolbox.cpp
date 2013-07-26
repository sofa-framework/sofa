#include "labelimagetoolbox.h"

namespace sofa
{

namespace component
{

namespace engine
{

LabelImageToolBox::LabelImageToolBox():   Inherited()
        , d_islinkedToToolBox(initData(&d_islinkedToToolBox,false,"islinkedtotoolbox","true if a toobbox use this Label"))
    {
        d_islinkedToToolBox.setReadOnly(true);
    }
    
}
}
}
