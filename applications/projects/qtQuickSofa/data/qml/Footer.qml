import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import SofaBasics 1.0

ToolBar {
    height: 20
    anchors.margins: 0
    anchors.leftMargin: 0

    property alias  statusMessage:  statusLabel.text
    property int    statusDuration: 5000

    RowLayout {
        anchors.fill: parent

        Rectangle {
            Layout.fillHeight: true
            Layout.preferredWidth: 60
            color: "transparent"
            border.width: 1
            border.color: "grey"
            radius: 2

            FPSDisplay {
                anchors.fill: parent
                anchors.margins: 5
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

                onTextChanged: clearStatusTimer.restart()

                Timer {
                    id: clearStatusTimer
                    running: false
                    repeat: false
                    interval: statusDuration
                    onTriggered: statusLabel.text = ""
                }
            }
        }
    }
}
