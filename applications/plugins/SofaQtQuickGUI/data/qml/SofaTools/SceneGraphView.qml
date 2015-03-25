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

        ListView {
            id: listView
            Layout.fillWidth: true
            Layout.preferredHeight: 400
            clip: true

            model: scene.listModel
            focus: true
            Component.onCompleted: scene.listModel.selectedId = currentIndex
            onCurrentIndexChanged: scene.listModel.selectedId = currentIndex
            onCountChanged: {
                if(currentIndex >= count)
                    currentIndex = -1;
            }

            highlight: Rectangle {
                color: "lightsteelblue";
                radius: 5
            }

            delegate: Item {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.leftMargin: depth * rowHeight
                height: visible ? rowHeight : 0
                visible: !(SceneListModel.Hidden & visibility)

                RowLayout {
                    anchors.fill: parent
                    spacing: 0

                    Image {
                        visible: isNode
                        source: !(SceneListModel.Collapsed & visibility) ? "qrc:/icon/downArrow.png" : "qrc:/icon/rightArrow.png"
                        Layout.preferredHeight: rowHeight
                        Layout.preferredWidth: Layout.preferredHeight

                        MouseArea {
                            anchors.fill: parent
                            enabled: isNode
                            onClicked: listView.model.setCollapsed(index, !(SceneListModel.Collapsed & visibility))
                        }
                    }

                    Text {
                        text: 0 !== type.length || 0 !== name.length ? type + " - " + name : ""
                        color: Qt.darker(Qt.rgba((depth * 6) % 9 / 8.0, depth % 9 / 8.0, (depth * 3) % 9 / 8.0, 1.0), 1.5)
                        font.bold: isNode

                        Layout.fillWidth: true
                        Layout.preferredHeight: rowHeight

                        MouseArea {
                            anchors.fill: parent
                            onClicked: listView.currentIndex = index
                        }
                    }
                }
            }
        }
    }
}
