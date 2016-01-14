#include "PythonMacros.h"

#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/Context.h>
//#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/common/Node.h>
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
#include <SofaMiscMapping/SubsetMultiMapping.h>
#include <sofa/core/BaseMapping.h>


#include "PythonScriptController.h"

#include "Binding_Base.h"
#include "Binding_BaseObject.h"
#include "Binding_BaseState.h"
#include "Binding_BaseMechanicalState.h"
#include "Binding_MechanicalObject.h"
#include "Binding_BaseContext.h"
#include "Binding_Context.h"
#include "Binding_Node.h"
#include "Binding_BaseLoader.h"
#include "Binding_MeshLoader.h"
#include "Binding_Topology.h"
#include "Binding_BaseMeshTopology.h"
#include "Binding_MeshTopology.h"
#include "Binding_GridTopology.h"
#include "Binding_RegularGridTopology.h"
#include "Binding_PythonScriptController.h"
#include "Binding_BaseMapping.h"
//#include "Binding_Mapping.h"
//#include "Binding_RigidMapping.h"
//#include "Binding_MultiMapping.h"
#include "Binding_SubsetMultiMapping.h"
#include "Binding_VisualModel.h"
#include "Binding_OBJExporter.h"
#include "Binding_DataEngine.h"

typedef sofa::component::container::MechanicalObject< sofa::defaulttype::Vec3Types > MechanicalObject3;
typedef sofa::component::mapping::SubsetMultiMapping< sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types > SubsetMultiMapping3_to_3;

using sofa::core::objectmodel::Base;

// crée un objet Python à partir d'un objet Cpp héritant de Base,
// retournant automatiquement le type Python de plus haut niveau possible
// en fonction du type de l'objet Cpp
// Ceci afin de permettre l'utilisation de fonctions des sous-classes de Base
PyObject* SP_BUILD_PYSPTR(Base* obj)
{
    if( obj->toBaseObject() )
    {
        if( obj->toBaseLoader() )
        {
            if (dynamic_cast<sofa::core::loader::MeshLoader*>(obj))
                return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(MeshLoader));
            return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(BaseLoader));
        }

        if( obj->toTopology() )
        {
            if( obj->toBaseMeshTopology() )
            {
                if (dynamic_cast<sofa::component::topology::RegularGridTopology*>(obj))
                    return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(RegularGridTopology));
                if (dynamic_cast<sofa::component::topology::GridTopology*>(obj))
                    return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(GridTopology));
                if (dynamic_cast<sofa::component::topology::MeshTopology*>(obj))
                    return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(MeshTopology));
                return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(BaseMeshTopology));
            }
            return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(Topology));
        }

        if( obj->toVisualModel())
        {
            if (dynamic_cast<sofa::component::visualmodel::VisualModelImpl*>(obj))
                return BuildPySPtr<Base>(obj, &SP_SOFAPYTYPEOBJECT(VisualModelImpl));
            return BuildPySPtr<Base>(obj, &SP_SOFAPYTYPEOBJECT(VisualModel));
        }

        if( obj->toBaseState() )
        {
            if (obj->toBaseMechanicalState())
            {
                if (dynamic_cast<sofa::component::container::MechanicalObject<sofa::defaulttype::Vec3Types>*>(obj))
                    return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(MechanicalObject));
                return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(BaseMechanicalState));
            }
            return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(BaseState));
        }


        if (obj->toBaseMapping())
        {
            if (dynamic_cast<SubsetMultiMapping3_to_3*>(obj))
                return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(SubsetMultiMapping3_to_3));
            return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(BaseMapping));
        }

        if (obj->toDataEngine())
            return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(DataEngine));

        if (dynamic_cast<sofa::component::controller::PythonScriptController*>(obj))
            return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(PythonScriptController));

        if (dynamic_cast<sofa::component::misc::OBJExporter*>(obj))
            return BuildPySPtr<Base>(obj, &SP_SOFAPYTYPEOBJECT(OBJExporter));

        return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(BaseObject)); // at least we know it is a BaseObject

    }
    else if( obj->toBaseContext() )
    {

        if (dynamic_cast<sofa::simulation::Node*>(obj))
            return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(Node));
        if (dynamic_cast<sofa::core::objectmodel::Context*>(obj))
            return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(Context));
        return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(BaseContext));

    }

    return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(Base)); // Base by default
}



void printPythonExceptions()
{
    PyObject *ptype, *pvalue /* error msg */, *ptraceback /*stack snapshot and many other informations (see python traceback structure)*/;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    if( pvalue ) SP_MESSAGE_EXCEPTION( PyString_AsString(pvalue) )

    // TODO improve the error message by using ptraceback
}
