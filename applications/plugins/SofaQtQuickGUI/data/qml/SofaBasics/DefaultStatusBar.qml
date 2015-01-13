import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0

ToolBar {
    id: root
    height: 20
    anchors.margins: 0
    anchors.leftMargin: 0
    property string statusMessage: ""

    RowLayout {
        anchors.fill: parent

        Rectangle {
            Layout.fillHeight: true
            Layout.preferredWidth: fpsDisplay.implicitWidth + 10
            color: "transparent"
            border.width: 1
            border.color: "grey"
            radius: 2

            FPSDisplay {
                id: fpsDisplay
                anchors.fill: parent
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumWidth: 256
            color: "transparent"
            border.width: 1
            border.color: "grey"
            radius: 2

            Label {
                id: statusLabel
                anchors.fill: parent
                anchors.margins: 5
                verticalAlignment: Text.AlignVCenter
                text: root.statusMessage
            }
        }
    }
}
