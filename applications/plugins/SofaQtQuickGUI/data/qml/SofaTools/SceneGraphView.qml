import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.2
import SofaBasics 1.0

CollapsibleGroupBox {
    id: root
    implicitWidth: 0

    title: "Scene Graph"
    property int priority: 90

    property Scene scene

    enabled: scene ? scene.ready : false

    /*ListView {
        anchors.fill: parent

        model: ListModel {
            ListElement {
                name: "root"
            }
        }

        delegate: Text {
            text: "name is " + name
        }
    }*/
/*
    GridLayout {
        id: layout
        anchors.fill: parent
        columns: 1

        Component {
            id: delegate
            Item {
                height: 20
                Column {
                    Text {
                        text: display
                    }
                }
            }
        }

        ListView {
            Layout.fillWidth: true
            Layout.preferredHeight: 300
            model: scene
            delegate: delegate
            focus: true
        }
    }
*/
}
