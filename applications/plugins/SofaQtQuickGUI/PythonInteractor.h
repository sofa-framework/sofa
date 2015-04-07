#ifndef PYTHONINTERACTOR_H
#define PYTHONINTERACTOR_H

#include "SofaQtQuickGUI.h"
#include <QObject>
#include <QQmlParserStatus>
#include <QMap>
#include <QString>
#include <QVariant>

#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

namespace sofa
{

namespace component
{

namespace controller
{
	class PythonScriptController;
}

}

namespace qtquick
{

class Scene;

class SOFA_SOFAQTQUICKGUI_API PythonInteractor : public QObject, public QQmlParserStatus
{
	Q_OBJECT
	Q_INTERFACES(QQmlParserStatus)

public:
	PythonInteractor(QObject *parent = 0);
	~PythonInteractor();

	void classBegin();
	void componentComplete();
	
public:
    Q_PROPERTY(sofa::qtquick::Scene* scene READ scene WRITE setScene NOTIFY sceneChanged)

public:
	Scene* scene() const	{return myScene;}
	void setScene(Scene* newScene);
	
signals:
    void sceneChanged(sofa::qtquick::Scene* newScene);
	
public:
    QVariant call(const QString& pythonClassName, const QString& funcName, const QVariant& parameter = QVariant());

protected:
    Q_INVOKABLE QVariant onCall(const QString& pythonClassName, const QString& funcName, const QVariant& parameter = QVariant());

public slots:
	void sendEvent(const QString& pythonClassName, const QString& eventName, const QVariant& parameter = QVariant());
	void sendEventToAll(const QString& eventName, const QVariant& parameter = QVariant());

private slots:
	void handleSceneChanged(Scene* scene);
	void retrievePythonScriptControllers();

private:
	typedef sofa::component::controller::PythonScriptController PythonScriptController;

	Scene*									myScene;
	QMap<QString, PythonScriptController*>	myPythonScriptControllers;
	
};

}

}

#endif // PYTHONINTERACTOR_H
