#include "Binding_GUIManager.h"
#include "PythonMacros.h"
#include "PythonToSofa.inl"
#include <sofa/gui/GUIManager.h>
#include <sofa/gui/BaseGUI.h>
#include <sofa/simulation/Node.h>

static constexpr const char* listSupportedGUI_DOC =
R"DOC(
Lists the supported GUI types.

:return: A list of strings populated with the supported GUI types
:rtype: list
)DOC";
static PyObject * GUIManager_listSupportedGUI(PyObject * /*self*/, PyObject * /*args*/) {
    const std::vector<std::string> guiList = sofa::gui::GUIManager::ListSupportedGUI();
    PyObject * list = PyList_New(guiList.size());
    for (unsigned int i = 0; i < guiList.size(); ++i) {
        PyObject *gui = PyString_FromString(guiList[i].c_str());
        PyList_SetItem(list, i, gui);
    }
    return list;
}


static constexpr const char* setConfigDirectoryPath_DOC =
R"DOC(
Set the Sofa's configuration directory path.

:param path: The path to the Sofa's configuration directory
:type path: str
)DOC";
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

static constexpr const char* getConfigDirectoryPath_DOC =
R"DOC(
Get the current Sofa's configuration directory path.

:return: The current Sofa's configuration directory path.
:rtype: str
)DOC";
static PyObject * GUIManager_getConfigDirectoryPath(PyObject * /*self*/, PyObject * /*args*/) {
    const std::string & path = sofa::gui::BaseGUI::getConfigDirectoryPath();
    return PyString_FromString(path.c_str());
}

static constexpr const char* setScreenshotDirectoryPath_DOC =
R"DOC(
Set the Sofa's screenshot directory path.

:param path: The path to the Sofa's screenshot directory
:type path: str
)DOC";
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

static constexpr const char* getScreenshotDirectoryPath_DOC =
R"DOC(
Get the current Sofa's screenshot directory path.

:return: The current Sofa's screenshot directory path.
:rtype: str
)DOC";
static PyObject * GUIManager_getScreenshotDirectoryPath(PyObject * /*self*/, PyObject * /*args*/) {
    const std::string & path = sofa::gui::BaseGUI::getScreenshotDirectoryPath();
    return PyString_FromString(path.c_str());
}

static constexpr const char* setSofaPrefix_DOC =
R"DOC(
Set the Sofa's prefix directory path.

:param path: The path to the Sofa's prefix directory
:type path: str
)DOC";
static PyObject * GUIManager_setSofaPrefix(PyObject * /*self*/, PyObject * args) {
    const char * path_ptr;

    if (!PyArg_ParseTuple(args, "s", &path_ptr)) {
        SP_MESSAGE_ERROR("GUIManager.setSofaPrefix requires a path as first argument.");
        return nullptr;
    }

    sofa::gui::GUIManager::SetSofaPrefix(path_ptr);
    Py_RETURN_NONE;
}

static constexpr const char* getSofaPrefix_DOC =
R"DOC(
Get the current Sofa's prefix directory path.

:return: The current Sofa's prefix directory path.
:rtype: str
)DOC";
static PyObject * GUIManager_getSofaPrefix(PyObject * /*self*/, PyObject * /*args*/) {
    const std::string & path = sofa::gui::GUIManager::GetSofaPrefix();
    return PyString_FromString(path.c_str());
}

static constexpr const char* init_DOC =
R"DOC(
Initialize the GUI manager.

:param type: The GUI's type.
:type path: str
)DOC";
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

static constexpr const char* createGUI_DOC =
R"DOC(
Create a new GUI.

:return: 1 if the GUI creation failed, 0 otherwise
:rtype: int
)DOC";
static PyObject * GUIManager_createGUI(PyObject * /*self*/, PyObject * /*args*/) {
    int err = sofa::gui::GUIManager::createGUI();
    return PyInt_FromLong(err);
}

static constexpr const char* setDimension_DOC =
R"DOC(
Set the GUI's dimension (width and height)

:param width: The GUI's width.
:param height: The GUI's height.
:type width: int
:type height: int
)DOC";
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

static constexpr const char* setScene_DOC =
R"DOC(
Set the GUI's main scene

:param root: The scene's root node
:param name: (optional) The scene's name
:type root: sofa::simulation::Node
:type name: str
)DOC";
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

static constexpr const char* MainLoop_DOC =
R"DOC(
Start the GUI's main loop

:param root: The scene's root node
:param filename: (optional) The scene's filename
:type root: sofa::simulation::Node
:type filename: str
)DOC";
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
SP_MODULE_METHOD_DOC(GUIManager, listSupportedGUI, listSupportedGUI_DOC)
SP_MODULE_METHOD_DOC(GUIManager, setConfigDirectoryPath, setConfigDirectoryPath_DOC)
SP_MODULE_METHOD_DOC(GUIManager, getConfigDirectoryPath, getConfigDirectoryPath_DOC)
SP_MODULE_METHOD_DOC(GUIManager, setScreenshotDirectoryPath, setScreenshotDirectoryPath_DOC)
SP_MODULE_METHOD_DOC(GUIManager, getScreenshotDirectoryPath, getScreenshotDirectoryPath_DOC)
SP_MODULE_METHOD_DOC(GUIManager, setSofaPrefix, setSofaPrefix_DOC)
SP_MODULE_METHOD_DOC(GUIManager, getSofaPrefix, getSofaPrefix_DOC)
SP_MODULE_METHOD_DOC(GUIManager, init, init_DOC)
SP_MODULE_METHOD_DOC(GUIManager, createGUI, createGUI_DOC)
SP_MODULE_METHOD_DOC(GUIManager, setDimension, setDimension_DOC)
SP_MODULE_METHOD_DOC(GUIManager, setScene, setScene_DOC)
SP_MODULE_METHOD_DOC(GUIManager, MainLoop, MainLoop_DOC)
SP_MODULE_METHOD(GUIManager, closeGUI)
SP_MODULE_METHODS_END