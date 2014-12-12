import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.1

ApplicationWindow {
    id: window

    title: "QtQuick SofaViewer"
    width: 1280
    height: 720
    property string filePath: ""

    menuBar: Header {

    }

    // sofa scene
    Scene {
        id: scene
    }

    // dialog
    FileDialog {
        id: openDialog
        nameFilters: ["Scene files (*.xml *.scn *.pscn *.py *.simu *)"]
        onAccepted: {
            scene.source = fileUrl;
        }
    }

    FileDialog {
        id: saveDialog
        selectExisting: false
        nameFilters: ["Scene files (*.scn)"]
        onAccepted: {
            scene.save(fileUrl);
        }
    }

    // action
    Action {
        id: openAction
        text: "&Open..."
        shortcut: "Ctrl+O"
        onTriggered: openDialog.open();
        tooltip: "Open a Sofa Scene"
    }

    Action {
        id: reloadAction
        text: "&Reload"
        shortcut: "Ctrl+R"
        onTriggered: scene.reload();
        tooltip: "Reload the Sofa Scene"
    }

    Action {
        id: saveAction
        text: "&Save"
        shortcut: "Ctrl+S"
        onTriggered: if(0 == filePath.length) saveDialog.open(); else scene.save(filePath);
        tooltip: "Save the Sofa Scene"
    }

    Action {
        id: saveAsAction
        text: "&Save As..."
        onTriggered: saveDialog.open();
        tooltip: "Save the Sofa Scene at a specific location"
    }

    Action
    {
        id: exitAction
        text: "&Exit"
        shortcut: "Ctrl+Q"
        onTriggered: close()
    }

    MessageDialog {
        id: aboutDialog
        title: "About"
        text: "Welcome in the qtQuickSofa Application"
        onAccepted: visible = false
    }

    Action
    {
        id: aboutAction
        text: "&About"
        onTriggered: aboutDialog.visible = true;
        tooltip: "What is this application ?"
    }

    // content

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            Viewer {
                id: viewer

                Layout.fillWidth: true
                Layout.fillHeight: true
                width: 75

                scene: scene

                focus: true

                Keys.onPressed: {
                    if(event.isAutoRepeat) {
                        event.accepted = true;
                        return;
                    }

                    if(event.key === Qt.Key_F4) {
                        if(0 !== window.visibility) { // not hidden
                            if(2 & window.visibility) // windowed
                                window.visibility = "FullScreen";
                            else // fullscreen
                                window.visibility = "Windowed";
                        }
                        event.accepted = true;
                    }
                }
            }

            Rectangle {
                id: toolPanel

                Layout.fillWidth: true
                Layout.fillHeight: true
                width: 25

                color: "lightgrey"

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 5

                    SimulationControl {
                        id: simulationControl
                        Layout.fillWidth: true

                        Connections {
                            target: viewer
                            onSceneChanged: {
                                simulationControl.animateButton.checked = viewer.scene.play;
                            }
                        }

                        Connections {
                            target: viewer.scene
                            onPlayChanged: {
                                simulationControl.animateButton.checked = viewer.scene.play;
                            }
                        }

                        animateButton.onCheckedChanged: viewer.scene.play = animateButton.checked
                    }

					Item {
						Layout.fillWidth: true
						Layout.fillHeight: true
					}
                }
            }
        }

        Footer {
            id: footer

            Layout.fillWidth: true
            height: 20
        }
    }
}
