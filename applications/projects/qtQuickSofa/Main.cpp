#include "Viewer.h"
#include "Scene.h"

#include <QtWidgets/QApplication>
#include <QQmlApplicationEngine>
#include <QQuickWindow>
#include <QOpenGLContext>
#include <sofa/helper/system/FileRepository.h>

int main(int argc, char **argv)
{
    QApplication app(argc, argv);

	qmlRegisterType<Scene>("Scene", 1, 0, "Scene");
    qmlRegisterType<Viewer>("Viewer", 1, 0, "Viewer");

    QUrl mainScriptUrl = QUrl("qrc:///data/qml/Main.qml");

	/*QSurfaceFormat format;
	format.setMajorVersion(3);
	format.setMajorVersion(2);
	format.setProfile(QSurfaceFormat::OpenGLContextProfile::CompatibilityProfile);*/

    QQmlApplicationEngine engine(mainScriptUrl);
    QObject* topLevel = engine.rootObjects().value(0);
    QQuickWindow* window = qobject_cast<QQuickWindow *>(topLevel);
    if(0 == window)
    {
        qDebug() << "Your QML root object should be a window";
        return 1;
    }

	//window->setFormat(format);
    window->show();

    return app.exec();
}
