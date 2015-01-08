#include "PythonMacros.h"

#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/BaseState.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/loader/BaseLoader.h>
#include <sofa/core/loader/MeshLoader.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBaseTopology/MeshTopology.h>
#include <SofaBaseTopology/GridTopology.h>
#include <SofaBaseTopology/RegularGridTopology.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaBaseTopology/PointSetTopologyModifier.h>
#include <SofaMiscMapping/SubsetMultiMapping.h>
#include <SofaBaseTopology/TriangleSetTopologyModifier.h>
#include <sofa/core/BaseMapping.h>
#include "PythonScriptController.h"


#include "Binding_PointSetTopologyModifier.h"
#include "Binding_TriangleSetTopologyModifier.h"

typedef sofa::component::container::MechanicalObject< sofa::defaulttype::Vec3Types > MechanicalObject3;
typedef sofa::component::mapping::SubsetMultiMapping< sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types > SubsetMultiMapping3_to_3;


    if( dynamic_cast<sofa::component::topology::TriangleSetTopologyModifier*>(obj) )
        return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(TriangleSetTopologyModifier));
    if( dynamic_cast<sofa::component::topology::PointSetTopologyModifier*>(obj) )
        return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(PointSetTopologyModifier));




void printPythonExceptions()
{
    PyObject *ptype, *pvalue /* error msg */, *ptraceback /*stack snapshot and many other informations (see python traceback structure)*/;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    if( pvalue ) SP_MESSAGE_EXCEPTION( PyString_AsString(pvalue) )

    // TODO improve the error message by using ptraceback
}
