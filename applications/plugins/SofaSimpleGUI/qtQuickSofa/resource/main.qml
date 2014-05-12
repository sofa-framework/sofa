import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import QSofaViewer 1.0

ApplicationWindow {

    title: "QtQuick SofaViewer"
    width: 320
    height: 480

    ColumnLayout {

        id: layout
        anchors.fill: parent
        spacing: 6

        Rectangle {
            color: Qt.rgba(1, 1, 1, 0.7)
            radius: 10
            border.width: 1
            border.color: "white"
            anchors.fill: label
            anchors.margins: -10
        }

        QSofaViewer {
            SequentialAnimation on t {
                NumberAnimation { to: 1; duration: 2500; easing.type: Easing.InQuad }
                NumberAnimation { to: 0; duration: 2500; easing.type: Easing.OutQuad }
                loops: Animation.Infinite
                running: true
            }
        }

        Rectangle {
            color: Qt.rgba(1, 1, 1, 0.7)
            radius: 10
            border.width: 1
            border.color: "white"
            anchors.fill: label
            anchors.margins: -10
        }

        Text {
            id: label
            color: "black"
            wrapMode: Text.WordWrap
            text: "Console"
            anchors.right: parent.right
            anchors.left: parent.left
            anchors.bottom: parent.bottom
            anchors.margins: 20
        }

        Button {
            text: "Push me !"
            width: 320
            height: 40
        }
    }

}
