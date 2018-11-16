#pragma once

#include <sofa/gui/qt/DataWidget.h>
#include <sofa/core/objectmodel/DataLink.h>
#include <QFileIconProvider>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/helper/system/FileSystem.h>
#include <QFileDialog>
#include <QPushButton>

namespace sofa {

namespace sofaTypes {


class QDataFilename : public QWidget {
    Q_OBJECT
public:
    QDataFilename(sofa::gui::qt::DataWidget* parent, const sofa::core::objectmodel::DataFileName & /*data*/) : QWidget(parent) {
        QHBoxLayout * layout = new QHBoxLayout();

        m_edit = new QLineEdit();
        m_edit->setPalette(Qt::white);

        m_button = new QPushButton();
        QFileIconProvider icon;
        m_button->setIcon(icon.icon(QFileIconProvider::Folder));

        layout->addWidget(m_edit);
        layout->addWidget(m_button);

        this->setLayout(layout);

        QObject::connect(m_edit, &QLineEdit::editingFinished, parent, &sofa::gui::qt::DataWidget::updateDataValue);
        QObject::connect(m_button, &QPushButton::clicked, this, &QDataFilename::open);

        setFixedHeight(30);
    }

    void readFromData(const sofa::core::objectmodel::DataFileName& data) {
        m_edit->setText(QString(data.getFullPath().c_str()));
    }

    void writeToData(sofa::core::objectmodel::DataFileName& data) {
        data.setValue(m_edit->text().toStdString());
    }

public slots:

    void open() {
        std::string startdir = helper::system::FileSystem::getParentDirectory(m_edit->text().toStdString());
//        if (startdir.empty()) startdir = helper::system::DataRepository.getScenePath();

        QString fileName =
          QFileDialog::getOpenFileName(nullptr,
                                       tr("Open Lib Scene"),
                                       QString(startdir.c_str()),
                                       tr("Files (*)"));

        if (!QFileInfo::exists(fileName))
            return;

       m_edit->setText(fileName);
    }

private:
    QLineEdit * m_edit;
    QPushButton * m_button;
};


class QLinkWidget : public QWidget {
    Q_OBJECT
public:

    QLinkWidget(sofa::gui::qt::DataWidget* parent, const BaseDataLink & /*data*/) : QWidget(parent) {
        setLayout(new QHBoxLayout());

        m_label = new QLabel(this);
        m_label->setEnabled(false);
        m_label->setPalette(Qt::white);

        m_edit = new QLineEdit(this);
//        m_edit->setPalette(Qt::white);

        layout()->addWidget(m_edit);
        layout()->addWidget(m_label);

        connect(m_edit, &QLineEdit::textChanged, this, &QLinkWidget::textChanged);
        connect(this, &QLinkWidget::setWidgetDirty, parent, &sofa::gui::qt::DataWidget::setWidgetDirty);
        connect(m_edit, &QLineEdit::editingFinished, parent, &sofa::gui::qt::DataWidget::updateDataValue);

        setFixedHeight(30);
    }

    void readFromData(const BaseDataLink & data) {
        if (data.getBaseLink() == NULL) m_label->setText("Not found");
        else m_label->setText("@"+QString(data.getBaseLink()->getName().c_str()));
    }

    void writeToData(BaseDataLink & data) {
        data.setValue(m_edit->text().toStdString());
        data.computeLink();
    }

public slots:
    void textChanged(const QString & /*str*/) {
        emit setWidgetDirty(true);
    }

signals:
    void setWidgetDirty(bool);

private:
    QLineEdit * m_edit;
    QLabel * m_label;
};

}

}
