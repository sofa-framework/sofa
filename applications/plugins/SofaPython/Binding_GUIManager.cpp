#include "Binding_GUIManager.h"
#include "PythonMacros.h"
#include "PythonToSofa.inl"
#include <sofa/gui/GUIManager.h>
#include <sofa/gui/BaseGUI.h>
#include <sofa/simulation/Node.h>

static PyObject * GUIManager_listSupportedGUI(PyObject * /*self*/, PyObject * /*args*/) {
    const std::vector<std::string> guiList = sofa::gui::GUIManager::ListSupportedGUI();
    PyObject * list = PyList_New(guiList.size());
    for (unsigned int i = 0; i < guiList.size(); ++i) {
        PyObject *gui = PyString_FromString(guiList[i].c_str());
        PyList_SetItem(list, i, gui);
    }
    return list;
}

static PyObject * GUIManager_setConfigDirectoryPath(PyObject * /*self*/, PyObject * args) {
    const char * path_ptr;
    PyObject * createIfNecessaryObj = nullptr;

    if (!PyArg_ParseTuple(args, "s|O", &path_ptr, &createIfNecessaryObj)) {
        SP_MESSAGE_ERROR("GUIManager.setConfigDirectoryPath requires a path as first argument.");
        return nullptr;
    }

    bool createIfNecessary = (createIfNecessaryObj == nullptr || PyObject_IsTrue(createIfNecessaryObj));
    sofa::gui::BaseGUI::setConfigDirectoryPath(path_ptr, createIfNecessary);

    Py_RETURN_NONE;
}

static PyObject * GUIManager_getConfigDirectoryPath(PyObject * /*self*/, PyObject * /*args*/) {
    const std::string & path = sofa::gui::BaseGUI::getConfigDirectoryPath();
    return PyString_FromString(path.c_str());
}

static PyObject * GUIManager_setScreenshotDirectoryPath(PyObject * /*self*/, PyObject * args) {
    const char * path_ptr;
    PyObject * createIfNecessaryObj = nullptr;

    if (!PyArg_ParseTuple(args, "s|O", &path_ptr, &createIfNecessaryObj)) {
        SP_MESSAGE_ERROR("GUIManager.setScreenshotDirectoryPath requires a path as first argument.");
        return nullptr;
    }

    bool createIfNecessary = (createIfNecessaryObj == nullptr || PyObject_IsTrue(createIfNecessaryObj));
    sofa::gui::BaseGUI::setScreenshotDirectoryPath(path_ptr, createIfNecessary);

    Py_RETURN_NONE;
}

static PyObject * GUIManager_getScreenshotDirectoryPath(PyObject * /*self*/, PyObject * /*args*/) {
    const std::string & path = sofa::gui::BaseGUI::getScreenshotDirectoryPath();
    return PyString_FromString(path.c_str());
}

static PyObject * GUIManager_setSofaPrefix(PyObject * /*self*/, PyObject * args) {
    const char * path_ptr;

    if (!PyArg_ParseTuple(args, "s", &path_ptr)) {
        SP_MESSAGE_ERROR("GUIManager.setSofaPrefix requires a path as first argument.");
        return nullptr;
    }

    sofa::gui::GUIManager::SetSofaPrefix(path_ptr);
    Py_RETURN_NONE;
}

static PyObject * GUIManager_getSofaPrefix(PyObject * /*self*/, PyObject * /*args*/) {
    const std::string & path = sofa::gui::GUIManager::GetSofaPrefix();
    return PyString_FromString(path.c_str());
}

static PyObject * GUIManager_init(PyObject * /*self*/, PyObject * args) {
    const char * name_ptr;
    if (!PyArg_ParseTuple(args, "s", &name_ptr)) {
        std::string msg = "GUIManager.init requires a valid gui name. Registered GUIs are '" + sofa::gui::GUIManager::ListSupportedGUI(',') + "'";
        SP_MESSAGE_ERROR(msg.c_str());
        return nullptr;
    }
    sofa::gui::GUIManager::Init("sofa", name_ptr);
    Py_RETURN_NONE;
}

static PyObject * GUIManager_createGUI(PyObject * /*self*/, PyObject * /*args*/) {
    int err = sofa::gui::GUIManager::createGUI();
    return PyInt_FromLong(err);
}

static PyObject * GUIManager_setDimension(PyObject * /*self*/, PyObject * args) {
    unsigned int width;
    unsigned int height;

    if (!PyArg_ParseTuple(args, "II", &width, &height)) {
        std::string msg = "GUIManager.setDimension requires two arguments (width and height)";
        SP_MESSAGE_ERROR(msg.c_str());
        return nullptr;
    }

    sofa::gui::GUIManager::SetDimension(width, height);

    Py_RETURN_NONE;
}

static PyObject * GUIManager_setScene(PyObject * /*self*/, PyObject * args) {
    PyObject * node_ptr  = nullptr;
    const char *filename = nullptr;

    if (!PyArg_ParseTuple(args, "O|s", &node_ptr, &filename)) {
        std::string msg = "GUIManager.setScene requires a root node";
        SP_MESSAGE_ERROR(msg.c_str());
        return nullptr;
    }

    auto * root = sofa::py::unwrap<sofa::simulation::Node>(node_ptr);
    sofa::gui::GUIManager::SetScene(root, filename);

    Py_RETURN_NONE;
}

static PyObject * GUIManager_MainLoop(PyObject * /*self*/, PyObject * args) {
    PyObject * node_ptr  = nullptr;
    const char *filename = nullptr;

    if (!PyArg_ParseTuple(args, "O|s", &node_ptr, &filename)) {
        std::string msg = "GUIManager.MainLoop requires a root node";
        SP_MESSAGE_ERROR(msg.c_str());
        return nullptr;
    }

    auto * root = sofa::py::unwrap<sofa::simulation::Node>(node_ptr);
    int err = sofa::gui::GUIManager::MainLoop(root, filename);

    return PyInt_FromLong(err);
}

static PyObject * GUIManager_closeGUI(PyObject * /*self*/, PyObject * /*args*/) {
    sofa::gui::GUIManager::closeGUI();
    Py_RETURN_NONE;
}


SP_MODULE_METHODS_BEGIN(GUIManager)
SP_MODULE_METHOD(GUIManager, listSupportedGUI)
SP_MODULE_METHOD(GUIManager, setConfigDirectoryPath)
SP_MODULE_METHOD(GUIManager, getConfigDirectoryPath)
SP_MODULE_METHOD(GUIManager, setScreenshotDirectoryPath)
SP_MODULE_METHOD(GUIManager, getScreenshotDirectoryPath)
SP_MODULE_METHOD(GUIManager, setSofaPrefix)
SP_MODULE_METHOD(GUIManager, getSofaPrefix)
SP_MODULE_METHOD(GUIManager, init)
SP_MODULE_METHOD(GUIManager, createGUI)
SP_MODULE_METHOD(GUIManager, setDimension)
SP_MODULE_METHOD(GUIManager, setScene)
SP_MODULE_METHOD(GUIManager, MainLoop)
SP_MODULE_METHOD(GUIManager, closeGUI)
SP_MODULE_METHODS_END