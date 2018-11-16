#include <sofa/gui/qt/DataWidget.h>
#include <sofa/gui/qt/DataLinkWidget.h>
#include <sofa/core/objectmodel/DataLink.h>

namespace sofa {

namespace sofaTypes {

sofa::helper::Creator<sofa::gui::qt::DataWidgetFactory, GenericDataWidget< sofa::core::objectmodel::DataFileName, QDataFilename> >	DWClass_datafilename("DataFilename",true);
sofa::helper::Creator<sofa::gui::qt::DataWidgetFactory, GenericDataWidget< BaseDataLink, QLinkWidget> >	DWClass_datalink("DataLink",true);

}

}
