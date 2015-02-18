import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.2
import SofaBasics 1.0
import SceneListModel 1.0

CollapsibleGroupBox {
    id: root
    implicitWidth: 0

    title: "Scene Graph"
    property int priority: 90

    property Scene scene
    property real rowHeight: 16

    enabled: scene ? scene.ready : false

    GridLayout {
        id: layout
        anchors.fill: parent
        columns: 1

        Component {
            id: delegate
            Item {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.leftMargin: depth * 16
                height: visible ? rowHeight : 0
                visible: !(SceneListModel.Hidden & visibility)

                MouseArea {
                    anchors.fill: parent
                    enabled: isNode
                    onClicked: listView.model.setCollapsed(index, !(SceneListModel.Collapsed & visibility))
                }

                Row {
                    Image {
                        visible: isNode
                        source: !(SceneListModel.Collapsed & visibility) ? "qrc:/icon/downArrow.png" : "qrc:/icon/rightArrow.png"
                        height: rowHeight
                        width: height
                    }

                    Text {
                        text: 0 !== type.length || 0 !== name.length ? type + " - " + name : ""
                        color: Qt.darker(Qt.rgba((depth * 6) % 9 / 8.0, depth % 9 / 8.0, (depth * 3) % 9 / 8.0, 1.0), 1.5)
                        font.bold: isNode
                    }
                }
            }
        }

        ScrollView {
            id: scrollView
            Layout.fillWidth: true
            Layout.preferredHeight: 400 //Math.min(scene.listModel.count * rowHeight, 400)
            clip: true

            ListView {
                id: listView
                width: parent.width
                model: scene.listModel
                delegate: delegate
                focus: true
            }
        }
    }
}
