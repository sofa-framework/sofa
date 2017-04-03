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

typedef sofa::component::container::MechanicalObject< sofa::defaulttype::Vec3Types > MechanicalObject3;
typedef sofa::component::mapping::SubsetMultiMapping< sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types > SubsetMultiMapping3_to_3;


void printPythonExceptions()
{
    PyObject *ptype, *pvalue /* error msg */, *ptraceback /*stack snapshot and many other informations (see python traceback structure)*/;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    if( pvalue ) SP_MESSAGE_EXCEPTION( PyString_AsString(pvalue) )

    // TODO improve the error message by using ptraceback
}
