#include "PythonInteractor.h"
#include "Scene.h"

#include <SofaPython/PythonCommon.h>
#include <SofaPython/PythonEnvironment.h>
#include <SofaPython/ScriptEnvironment.h>
#include <SofaPython/PythonMacros.h>
#include <SofaPython/PythonScriptController.h>
#include <SofaPython/PythonScriptFunction.h>

#include <qqml.h>
#include <QDebug>
#include <QSequentialIterable>
#include <vector>

PythonInteractor::PythonInteractor(QObject *parent) : QObject(parent), QQmlParserStatus(),
	myScene(0),
	myPythonScriptControllers()
{
	connect(this, &PythonInteractor::sceneChanged, this, &PythonInteractor::handleSceneChanged);
}

PythonInteractor::~PythonInteractor()
{
	
}

void PythonInteractor::classBegin()
{

}

void PythonInteractor::componentComplete()
{
	if(!myScene)
		setScene(qobject_cast<Scene*>(parent()));
}

void PythonInteractor::setScene(Scene* newScene)
{
	if(newScene == myScene)
		return;

	myScene = newScene;

	sceneChanged(newScene);
}

void PythonInteractor::handleSceneChanged(Scene* scene)
{
	if(scene)
	{
		if(scene->isReady())
			retrievePythonScriptControllers();

		connect(scene, &Scene::loaded, this, &PythonInteractor::retrievePythonScriptControllers);
	}
}

void PythonInteractor::retrievePythonScriptControllers()
{
	myPythonScriptControllers.clear();

	if(!myScene || !myScene->isReady())
		return;

	std::vector<PythonScriptController*> pythonScriptControllers;
	myScene->sofaSimulation()->GetRoot()->get<PythonScriptController>(&pythonScriptControllers, sofa::core::objectmodel::BaseContext::SearchDown);

	for(size_t i = 0; i < pythonScriptControllers.size(); ++i)
		myPythonScriptControllers.insert(pythonScriptControllers[i]->m_classname.getValue().c_str(), pythonScriptControllers[i]);
}

static PyObject* PythonBuildValueHelper(const QVariant& parameter)
{
	PyObject* value = 0;
	if(!parameter.isNull())
	{
		switch(parameter.type())
		{
		case QVariant::Bool:
			value = Py_BuildValue("b", parameter.toBool());
			break;
		case QVariant::Int:
			value = Py_BuildValue("i", parameter.toInt());
			break;
		case QVariant::UInt:
			value = Py_BuildValue("I", parameter.toUInt());
			break;
		case QVariant::Double:
			value = Py_BuildValue("d", parameter.toDouble());
			break;
		case QVariant::String:
			value = Py_BuildValue("s", parameter.toString().toLatin1().constData());
			break;
		default:
			value = Py_BuildValue("");
			qDebug() << "ERROR: buildPythonParameterHelper, type not supported:" << parameter.typeName();
			break;
		}
	}

	return value;
}

static PyObject* PythonBuildTupleHelper(const QVariant& parameter, bool mustBeTuple)
{
	PyObject* tuple = 0;

	if(!parameter.isNull())
	{
		if(QVariant::List == parameter.type())
		{
			QSequentialIterable parameterIterable = parameter.value<QSequentialIterable>();
			tuple = PyTuple_New(parameterIterable.size());

			int count = 0;
			for(const QVariant& i : parameterIterable)
				PyTuple_SetItem(tuple, count++, PythonBuildTupleHelper(i, false));
		}
		else if(QVariant::Map == parameter.type())
		{
			PyObject* dict = PyDict_New();

			QVariantMap map = parameter.value<QVariantMap>();

			for(QVariantMap::const_iterator i = map.begin(); i != map.end(); ++i)
				PyDict_SetItemString(dict, i.key().toLatin1().constData(), PythonBuildTupleHelper(i.value(), false));

			if(mustBeTuple)
			{
				tuple = PyTuple_New(1);
				PyTuple_SetItem(tuple, 0, dict);
			}
			else
			{
				tuple = dict;
			}
		}
		else
		{
			if(mustBeTuple)
			{
				tuple = PyTuple_New(1);
				PyTuple_SetItem(tuple, 0, PythonBuildValueHelper(parameter));
			}
			else
			{
				tuple = PythonBuildValueHelper(parameter);
			}
		}
	}

	return tuple;
}

static QVariant ExtractPythonValueHelper(PyObject* parameter)
{
	QVariant value;

	if(parameter)
	{
		if(PyBool_Check(parameter))
			value = (Py_False != parameter);
		else if(PyInt_Check(parameter))
            value = (int)PyInt_AsLong(parameter);
		else if(PyFloat_Check(parameter))
			value = PyFloat_AsDouble(parameter);
		else if(PyString_Check(parameter))
			value = PyString_AsString(parameter);
	}

	return value;
}

static QVariant ExtractPythonTupleHelper(PyObject* parameter)
{
	QVariant value;

	if(!parameter)
		return value;
	
	if(PyTuple_Check(parameter) || PyList_Check(parameter))
	{
		QVariantList tuple;

		PyObject *iterator = PyObject_GetIter(parameter);
		PyObject *item;

		if(!iterator)
			return value;

		while(item = PyIter_Next(iterator))
		{
			tuple.append(ExtractPythonTupleHelper(item));

			Py_DECREF(item);
		}
		Py_DECREF(iterator);

		if(PyErr_Occurred())
			qDebug() << "ERROR: during python tuple/list iteration";

		return tuple;
	}
	else if(PyDict_Check(parameter))
	{
		QVariantMap map;

		PyObject* key;
		PyObject* item;
		Py_ssize_t pos = 0;

		while(PyDict_Next(parameter, &pos, &key, &item))
			map.insert(PyString_AsString(key), ExtractPythonTupleHelper(item));

		if(PyErr_Occurred())
			qDebug() << "ERROR: during python dictionary iteration";

		return map;
	}
	else
	{
		value = ExtractPythonValueHelper(parameter);
	}	

	return value;
}

QVariant PythonInteractor::onCall(const QString& pythonClassName, const QString& funcName, const QVariant& parameter)
{
	QVariant result;

	if(!myScene)
	{
		qDebug() << "ERROR: cannot call Python function on a null scene";
		return result;
	}

	if(!myScene->isReady())
	{
		qDebug() << "ERROR: cannot call Python function on a scene that is still loading";
		return result;
	}

	if(pythonClassName.isEmpty())
	{
		qDebug() << "ERROR: cannot call Python function without a valid python class name";
		return result;
	}

	if(funcName.isEmpty())
	{
		qDebug() << "ERROR: cannot call Python function without a valid python function name";
		return result;
	}

	auto pythonScriptControllerIterator = myPythonScriptControllers.find(pythonClassName);
	if(myPythonScriptControllers.end() == pythonScriptControllerIterator)
	{
		qDebug() << "ERROR: cannot send Python event on an unknown script controller:" << pythonClassName;
		if(myPythonScriptControllers.isEmpty())
		{
			qDebug() << "There is no PythonScriptController";
		}
		else
		{
			qDebug() << "Known PythonScriptController(s):";
			for(const QString& pythonScriptControllerName : myPythonScriptControllers.keys())
				qDebug() << "-" << pythonScriptControllerName;
		}

		return result;
	}

	PythonScriptController* pythonScriptController = pythonScriptControllerIterator.value();
	if(pythonScriptController)
	{
		PyObject* pyCallableObject = PyObject_GetAttrString(pythonScriptController->scriptControllerInstance(), funcName.toLatin1().constData());
		if(!pyCallableObject)
		{
			qDebug() << "ERROR: cannot call Python function without a valid python class and function name";
		}
		else
		{
			PythonScriptFunction pythonScriptFunction(pyCallableObject, true);
			PythonScriptFunctionParameter pythonScriptParameter(PythonBuildTupleHelper(parameter, true), true);
			PythonScriptFunctionResult pythonScriptResult;

			pythonScriptFunction(&pythonScriptParameter, &pythonScriptResult);

			result = ExtractPythonTupleHelper(pythonScriptResult.data());
		}
	}

	return result;
}

void PythonInteractor::sendEvent(const QString& pythonClassName, const QString& eventName, const QVariant& parameter)
{
	if(!myScene)
	{
		qDebug() << "ERROR: cannot send Python event on a null scene";
		return;
	}

	if(!myScene->isReady())
	{
		qDebug() << "ERROR: cannot send Python event on a scene that is still loading";
		return;
	}

	auto pythonScriptControllerIterator = myPythonScriptControllers.find(pythonClassName);
	if(myPythonScriptControllers.end() == pythonScriptControllerIterator)
	{
		qDebug() << "ERROR: cannot send Python event on an unknown script controller:" << pythonClassName;
		if(myPythonScriptControllers.isEmpty())
		{
			qDebug() << "There is no PythonScriptController";
		}
		else
		{
			qDebug() << "Known PythonScriptController(s):";
			for(const QString& pythonScriptControllerName : myPythonScriptControllers.keys())
				qDebug() << "-" << pythonScriptControllerName;
		}

		return;
	}

	PythonScriptController* pythonScriptController = pythonScriptControllerIterator.value();

	PyObject* pyParameter = PythonBuildValueHelper(parameter);
	if(!pyParameter)
		pyParameter = Py_BuildValue("");

	sofa::core::objectmodel::PythonScriptEvent pythonScriptEvent(myScene->sofaSimulation()->GetRoot(), eventName.toLatin1().constData(), pyParameter);
	pythonScriptController->handleEvent(&pythonScriptEvent);
}

void PythonInteractor::sendEventToAll(const QString& eventName, const QVariant& parameter)
{
	for(const QString& pythonClassName : myPythonScriptControllers.keys())
		sendEvent(pythonClassName, eventName, parameter);
}
