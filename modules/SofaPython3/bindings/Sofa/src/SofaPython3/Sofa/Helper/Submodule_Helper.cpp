
#include <sofa/helper/logging/Messaging.h>
#include <SofaPython3/PythonEnvironment.h>
#include "Submodule_Helper.h"

namespace sofapython3
{
using sofa::core::objectmodel::Base;
using sofa::helper::logging::ComponentInfo;
using sofa::helper::logging::SofaComponentInfo;

static const std::string s_emitter = "PythonScript";

template<class Action>
static void parse_emitter_message_then(py::args args, const Action& action) {
    py::object py_emitter {py::none()};

    const size_t argSize = args.size();
    if( argSize == 1 )
    {
        /// no emitter
        std::string message = py::cast<std::string>(args[0]);
        action(ComponentInfo::SPtr(new ComponentInfo(s_emitter)), message.c_str(),
                                   PythonEnvironment::getPythonCallingPointAsFileInfo());
    } else if( argSize == 2 ) {
        /// SOURCE, "Message"
        py_emitter = args[0];
        std::string message = py::cast<std::string>(args[1]);

        if( py::isinstance<py::str>(py_emitter) )
        {
            action(ComponentInfo::SPtr(new ComponentInfo(py::cast<std::string>(py_emitter))),
                   message.c_str(), PythonEnvironment::getPythonCallingPointAsFileInfo());
        }else if (py::isinstance<Base>(py_emitter))
        {
            action(ComponentInfo::SPtr(new SofaComponentInfo(py::cast<Base*>(py_emitter))),
                   message.c_str(), PythonEnvironment::getPythonCallingPointAsFileInfo());
        }else
        {
            throw py::type_error("The first parameter must be a string or a Sofa.Core.Base object");
        }
    } else if( argSize == 3 ){
        /// "Message", "FILENAME", LINENO
        std::string message = py::cast<std::string>(args[0]);
        std::string filename = py::cast<std::string>(args[1]);
        int lineno = py::cast<int>(args[2]);

        action(ComponentInfo::SPtr(new ComponentInfo(s_emitter)),
               message.c_str(), SOFA_FILE_INFO_COPIED_FROM(filename, lineno));
    } else if (argSize == 4 ){
        /// SOURCE, "Message", "FILENAME", LINENO
        py_emitter = args[0];
        std::string message = py::cast<std::string>(args[1]);
        std::string filename = py::cast<std::string>(args[2]);
        int lineno = py::cast<int>(args[3]);

        if( py::isinstance<std::string>(py_emitter) )
        {
            action(ComponentInfo::SPtr(new ComponentInfo(py::cast<std::string>(py_emitter))),
                   message.c_str(),  SOFA_FILE_INFO_COPIED_FROM(filename, lineno));
        }else if (py::isinstance<Base>(py_emitter))
        {
            action(ComponentInfo::SPtr(new SofaComponentInfo(py::cast<Base*>(py_emitter))),
                   message.c_str(),  SOFA_FILE_INFO_COPIED_FROM(filename, lineno));
        }else
        {
            throw py::type_error("The first parameter must be a string or a Sofa.Core.Base object");
        }
    } else {
        throw py::type_error("Invalid arguments type to function.");
    }
}

#define MESSAGE_DISPATCH(MessageType) \
    parse_emitter_message_then(args, [](const ComponentInfo::SPtr& emitter, \
                                  const char* message, \
                                  const sofa::helper::logging::FileInfo::SPtr& fileinfo) \
    { MessageType(emitter) << message << fileinfo; });

/// The first parameter must be named the same as the module file to load.
pybind11::module addSubmoduleHelper(py::module& p)
{   
    py::module helper = p.def_submodule("Helper");

    helper.def("msg_info", [](py::args args) { MESSAGE_DISPATCH(msg_info); });
    helper.def("msg_warning", [](py::args args) { MESSAGE_DISPATCH(msg_warning); });
    helper.def("msg_error", [](py::args args) { MESSAGE_DISPATCH(msg_error); });
    helper.def("msg_deprecated", [](py::args args) { MESSAGE_DISPATCH(msg_deprecated); });
    helper.def("msg_fatal", [](py::args args) { MESSAGE_DISPATCH(msg_fatal); });

    return helper;
}

} ///namespace sofapython3
