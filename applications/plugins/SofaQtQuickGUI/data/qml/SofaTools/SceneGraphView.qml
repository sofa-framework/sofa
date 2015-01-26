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
/*
    GridLayout {
        id: layout
        anchors.fill: parent
        columns: 1

        Component {
            id: delegate
            Item {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.leftMargin: (depth) * 16
                height: visible ? 16 : 0
                //visible: listView.elementCollasped(parentIndex)
                Row {

                    Button {
                        visible: isNode
                        checkable: true
                        checked: true
                        height: 16
                        width: height
                        //onClicked: listView.setElementCollapsed(index, checked)
                    }

                    Text {
                        text: type + " - " + name
                        color: Qt.darker(Qt.rgba((depth * 6) % 9 / 8.0, depth % 9 / 8.0, (depth * 3) % 9 / 8.0, 1.0), 1.5)
                        font.bold: isNode
                    }
                }
            }
        }

        ScrollView {
            Layout.fillWidth: true
            Layout.preferredHeight:400

            ListView {
                id: listView
                anchors.fill: parent
                clip: true
                model: scene
                delegate: delegate
                focus: true

                property var elements

                Component.onCompleted: modelChanged()
                onModelChanged: {
                    elements = Array(model.count).join[{"collapsed": false}];
                }

                function elementCollapsed(index) {
                    return elements[index].collapsed;
                }

                function setElementCollapsed(index, state) {
                    if(state === elements[index].collapsed)
                        return;

                    elements[index].collapsed = state;

                    refresh();
                }

                function refresh() {

                }
            }
        }
    }
*/
}
