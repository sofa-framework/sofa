import QtQuick 2.0
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.2
import SofaBasics 1.0

CollapsibleGroupBox {
    id: root

    title: "Scene Parameters"
    enabled: scene.ready

    ColumnLayout {
        anchors.fill: parent

        GroupBox {
            Layout.fillWidth: true

            title: "Parameters"

            GridLayout {
                anchors.fill: parent
                columns: 3

                Label {
                    Layout.preferredWidth: implicitWidth
                    text: ""
                }
                RowLayout {
                    Layout.columnSpan: 2
                    Layout.fillWidth: true

                    Label {
                        Layout.fillWidth: true
                        Layout.preferredWidth: 20
                        horizontalAlignment: Text.AlignHCenter
                        text: "X    "
                        color: enabled ? "red" : "darkgrey"
                        font.bold: true
                    }
                    Label {
                        Layout.fillWidth: true
                        Layout.preferredWidth: 20
                        horizontalAlignment: Text.AlignHCenter
                        text: "Y    "
                        color: enabled ? "green" : "darkgrey"
                        font.bold: true
                    }
                    Label {
                        Layout.fillWidth: true
                        Layout.preferredWidth: 20
                        horizontalAlignment: Text.AlignHCenter
                        text: "Z    "
                        color: enabled ? "blue" : "darkgrey"
                        font.bold: true
                    }
                }

                // gravity
                Label {
                    Layout.preferredWidth: implicitWidth
                    text: "Gravity"
                }
                Vector3DSpinBox {
                    id: gravity
                    Layout.columnSpan: 2
                    Layout.fillWidth: true
                    decimals: 4
                    stepSize:0.01

                    function update() {
                        scene.pythonInteractor.call("PythonScript", "setGravity", vx, vy, vz);
                    }

                    Component.onCompleted: {
                        setValueFromArray(scene.pythonInteractor.call("PythonScript", "getGravity"));

                        onVxChanged.connect(update);
                        onVyChanged.connect(update);
                        onVzChanged.connect(update);
                    }
                }

                // point location
                Label {
                    Layout.preferredWidth: implicitWidth
                    text: "Position"
                }
                Vector3DSpinBox {
                    id: pointLocation
                    Layout.columnSpan: 2
                    Layout.fillWidth: true
                    decimals: 4
                    stepSize:0.01
                    enabled: false

                    Timer {
                        id: timer
                        running: false
                        repeat: true
                        interval: 20

                        onTriggered: if(scene.ready) pointLocation.update()
                    }

                    function update() {
                        setValueFromArray(scene.pythonInteractor.call("PythonScript", "getPointLocation"));
                    }

                    Component.onCompleted: timer.start()
                }
            }
        }
    }
}
