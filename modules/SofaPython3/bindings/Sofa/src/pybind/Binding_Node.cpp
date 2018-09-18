
/// Neede to have automatic conversion from pybind types to stl container.
#include <pybind11/stl.h>

#include "Binding_Node.h"

#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/simulation/Simulation.h>
using sofa::core::objectmodel::BaseData;

#include <SofaSimulationGraph/SimpleApi.h>
namespace simpleapi = sofa::simpleapi;

#include "Binding_PythonController.h"

std::string toSofaParsableString(const py::handle& p)
{
    if(py::isinstance<py::list>(p) || py::isinstance<py::tuple>(p))
    {
        std::stringstream tmp;
        for(auto pa : p){
            tmp << toSofaParsableString(pa) << " ";
        }
        return tmp.str();
    }
    return py::repr(p);
}

/// RVO optimized function. Don't care about copy on the return code.
std::map<std::string, std::string> toStringMap(const py::dict& dict)
{
    std::map<std::string, std::string> tmp;
    for(auto kv : dict)
    {
        tmp[py::str(kv.first)] = toSofaParsableString(kv.second);
    }
    return tmp;
}

void moduleAddNode(py::module &m) {
    py::class_<sofa::core::objectmodel::BaseNode, Base,
            sofa::core::objectmodel::BaseNode::SPtr>(m, "BaseNode");

    py::class_<sofa::core::objectmodel::BaseContext,
            sofa::core::objectmodel::Base,
            sofa::core::objectmodel::BaseContext::SPtr>(m, "BaseContext");

    py::class_<sofa::core::objectmodel::Context,
            sofa::core::objectmodel::BaseContext,
            sofa::core::objectmodel::Context::SPtr>(m, "Context");

    py::class_<Node, sofa::core::objectmodel::BaseNode,
            sofa::core::objectmodel::Context, Node::SPtr>
            p(m, "Node");

    p.def("createObject", [](Node& self, const std::string& type, const
          py::kwargs& kwargs) -> py::object
    {
        return py::cast( simpleapi::createObject(&self, type,
                                                 toStringMap(kwargs)) );
    });

    p.def("addPythonObject", [](Node& self, py::handle handle) {
        // convert PyObject* (the handle) -  to BaseObject*
        py::detail::type_caster<BaseObject> obj_caster;
        if (!obj_caster.load(handle, true))
          throw py::value_error();
        BaseObject* o = obj_caster;

        Py_INCREF(handle.ptr());
        // create a sptr from the python pointer, that will properly remove the
        // decref when deleted
        std::shared_ptr<PyObject> pyptr(handle.ptr(), [](PyObject *ob) {
          Py_DECREF(ob);
        }); // custom desctructor

        // Where the magic happen: creates a aliasing shared_ptr, sharing ownership
        // of the PyObj and holding the BaseObject pointer
        std::shared_ptr<BaseObject> baseObj(pyptr, o);

        PythonObjectWrapper::SPtr f(new PythonObjectWrapper(baseObj));

        // finally, add the wrapped object to the Node!
        self.addObject(f);

        // maybe I can return handle instead...? didn't check that yet
        return f;
      });

    p.def("createChild", &Node::createChild);
    p.def("getRoot", &Node::getRoot);

    // TODO in Sofa.Runtime ?
    p.def("simulationStep", [](Node &n, double dt) {
        sofa::simulation::getSimulation()->animate(&n, dt);
    }, "dt"_a);

    p.def("reset", [](Node &n) { sofa::simulation::getSimulation()->reset(&n);
                               }); p.def("init", [](Node &n) {
        sofa::simulation::getSimulation()->init(&n); });

    p.def("__getattr__", [](Node& self, const std::string& name) -> py::object
    {
        /// I'm not sure implicit behavior is nice but we could do:
        ///    - The attribute is a data,
        ///         returns it if it is a container
        ///         returns the value/specific binding otherwise
        ///    - The attribute is a link, return it.
        ///    - The attribute is an object or a child return it.
        ///    - The attribute is not existing:
        ///                raise an exception or search using difflib for
        /// close match.
        BaseData* d = self.findData(name); if(d!=nullptr)
            return py::cast(d);

        //TODO missing link search.

        std::cout << "SEARCHING FOR: " << name << std::endl;
        BaseObject *object = self.getObject(name);
        if (object)
            return py::cast(object);

        Node *child = self.getChild(name);
        if (child)
            return py::cast(child);

        return py::none();
    });

    p.def("getChild", [](Node &n, const std::string &name) -> py::object {
        sofa::simulation::Node *child = n.getChild(name);
        if (child)
            return py::cast(child);
        else
            return py::none();
    });
}
