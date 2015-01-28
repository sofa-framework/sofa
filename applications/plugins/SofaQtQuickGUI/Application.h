#ifndef APPLICATION_H
#define APPLICATION_H

#include <QQmlApplicationEngine>
#include <QApplication>

class Application : public QQmlApplicationEngine
{
    Q_OBJECT

public:
	Application(QObject* parent = 0);
	Application(const QUrl& url, QObject* parent = 0);
	Application(const QString& filePath, QObject* parent = 0);
    ~Application();

private:
	void init();

public:
	Q_INVOKABLE void trimCache(QObject* object = 0);

	Q_INVOKABLE void clearSettingGroup(const QString& group);

};

#endif // APPLICATION_H
