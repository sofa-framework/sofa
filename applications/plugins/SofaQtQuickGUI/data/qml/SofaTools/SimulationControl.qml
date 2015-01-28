import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.2
import SofaBasics 1.0

CollapsibleGroupBox {
    id: root
    implicitWidth: 0

    title: "Simulation Control"
    property int priority: 100

    enabled: scene.ready

    GridLayout {
        anchors.fill: parent
        columns: 3

        Button {
            id: animateButton
            Layout.columnSpan: 3
            Layout.fillWidth: true
            text: "Animate"
            checkable: true
            onCheckedChanged: scene.play = animateButton.checked
            tooltip: ""

            Connections {
                target: scene
                ignoreUnknownSignals: true
                onPlayChanged: {
                    animateButton.checked = scene.play;
                }
            }
        }

        Button {
            id: stepButton
            Layout.columnSpan: 3
            Layout.fillWidth: true
            text: "Step"
            tooltip: ""

            onClicked: {
                if(scene) {
                    scene.step();
                }
            }
        }
        Button {
            id: resetButton
            Layout.columnSpan: 3
            Layout.fillWidth: true
            text: "Reset"
            tooltip: "Reset the scene"

            onClicked: {
                if(scene) {
                    scene.reset();
                }
            }
        }

        Label {
            Layout.preferredWidth: implicitWidth
            text: "DT"
        }
        SpinBox {
            id: dtSpinBox
            Layout.columnSpan: 2
            Layout.fillWidth: true
            decimals: 3
            prefix: value <= 0 ? "Real-time " : ""
            suffix: " seconds"
            stepSize: 0.001
            value: scene.dt
            onValueChanged: scene.dt = value

            Component.onCompleted: {
                valueChanged();
            }
        }

        Label {
            Layout.preferredWidth: implicitWidth
            text: "Interaction stiffness"
        }
        Slider {
            id: interactionStiffnessSlider
            Layout.fillWidth: true
            maximumValue: 1000
            value: scene.pickingInteractor.stiffness
            onValueChanged: scene.pickingInteractor.stiffness = value
            stepSize: 1

            Component.onCompleted: {
                minimumValue = 1;
            }
        }
        TextField {
            Layout.preferredWidth: 32
            enabled: false
            text: interactionStiffnessSlider.value
        }

        Button {
            id: displayGraphButton
            Layout.columnSpan: 3
            Layout.fillWidth: true
            text: "Display scene graph"
            tooltip: ""
            onClicked: {
                displayGraphText.text = scene.dumpGraph();
                displayGraphDialog.open();
            }

            Dialog {
                id: displayGraphDialog
                title: "Simulation Scene Graph"
                width: window.width * 2.0 / 3.0
                height: window.height * 2.0 / 3.0

                contentItem: Item {
                    GridLayout {
                        anchors.fill: parent
                        anchors.margins: 5
                        TextArea {
                            id: displayGraphText
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            readOnly: true

                            Component.onCompleted: {
                                wrapMode = TextEdit.NoWrap;
                            }
                        }
                    }
                }
            }
        }
    }
}
