#include <sofa/gui/qt/VersionChecker.h>

namespace sofa::gui::qt
{
VersionChecker::VersionChecker(QObject* parent): QObject(parent)
{}

void VersionChecker::checkLatestVersion(const QString& owner,
    const QString& repo)
{
    QThread* thread = new QThread;
    connect(thread, &QThread::started, this,
        [this, owner, repo]() {
            QNetworkAccessManager* manager = new QNetworkAccessManager;
            connect(manager, &QNetworkAccessManager::finished, this, &VersionChecker::onReplyFinished);

            const QUrl url("https://api.github.com/repos/" + owner + "/" + repo + "/releases/latest");
            QNetworkRequest request(url);
            request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

            manager->get(request);
    });
    connect(this, &VersionChecker::versionCheckFinished, this, &VersionChecker::onVersionCheckFinished);
    connect(this, &VersionChecker::versionCheckFinished, thread, &QThread::quit);
    connect(thread, &QThread::finished, thread, &QThread::deleteLater);
    connect(thread, &QThread::finished, this, &VersionChecker::deleteLater);
    moveToThread(thread);
    thread->start();
}

void VersionChecker::onReplyFinished(QNetworkReply* reply)
{
    if (reply->error() == QNetworkReply::NoError)
    {
        QByteArray responseData = reply->readAll();
        QJsonDocument jsonDocument = QJsonDocument::fromJson(responseData);

        QString latestVersion;
        if (jsonDocument.isObject())
        {
            QJsonObject jsonObject = jsonDocument.object();
            QJsonValue versionValue = jsonObject.value("tag_name");
            if (versionValue.isString())
            {
                latestVersion = versionValue.toString();
            }
        }

        emit versionCheckFinished(latestVersion);
    }
    else
    {
        std::cout << "Error " << reply->error() << std::endl;
        emit versionCheckFinished("");
    }

    reply->deleteLater();
}

void VersionChecker::onVersionCheckFinished(const QString& latestVersion)
{
    std::string currentVersion = "1.0"; // Your software's current version

    if (latestVersion.isEmpty())
    {
        std::cout << "Unable to retrieve latest version from GitHub." << std::endl;
    }
    else
    {
        if (latestVersion > currentVersion.c_str())
        {
            std::cout << "A newer version (" << latestVersion.toStdString() << ") is available!" << std::endl;
        }
        else
        {
            std::cout << "You are using the latest version (" << currentVersion << ")." << std::endl;
        }
    }
}

}
