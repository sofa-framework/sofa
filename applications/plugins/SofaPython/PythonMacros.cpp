#include "PythonMacros.h"

#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/core/BaseState.h>
using namespace sofa::simulation;
using namespace sofa::simulation::tree;
using namespace sofa::core::objectmodel;
using namespace sofa::core;

#include "Binding_Base.h"
#include "Binding_BaseObject.h"
#include "Binding_BaseState.h"
#include "Binding_BaseContext.h"
#include "Binding_Context.h"
#include "Binding_Node.h"
#include "Binding_GNode.h"

using namespace sofa::core;

// crée un objet Python à partir d'un objet Cpp héritant de Base,
// retournant automatiquement le type Python de plus haut niveau possible
// en fonction du type de l'objet Cpp
// Ceci afin de permettre l'utilisation de fonctions des sous-classes de Base
PyObject* SP_BUILD_PYSPTR(Base* obj)
{
    if (dynamic_cast<GNode*>(obj))
        return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(GNode));
    if (dynamic_cast<Node*>(obj))
        return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(Node));
    if (dynamic_cast<Context*>(obj))
        return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(Context));
    if (dynamic_cast<BaseContext*>(obj))
        return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(BaseContext));

    if (dynamic_cast<BaseState*>(obj))
        return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(BaseState));
    if (dynamic_cast<BaseObject*>(obj))
        return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(BaseObject));

    // par défaut...
    return BuildPySPtr<Base>(obj,&SP_SOFAPYTYPEOBJECT(Base));
}
