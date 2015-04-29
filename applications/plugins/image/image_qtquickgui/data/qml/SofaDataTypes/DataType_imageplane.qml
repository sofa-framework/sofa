import QtQuick 2.0
import QtQuick.Controls 1.3
import QtQuick.Layouts 1.1
import SofaBasics 1.0
import SceneData 1.0
import ImagePlaneModel 1.0
import ImagePlaneView 1.0

GridLayout {
    id: root
    columns: 2

    property var dataObject

    ImagePlaneModel {
        id: imagePlane

        sceneData: root.dataObject.data
    }

    Component {
        id: sliceComponent

        ColumnLayout {
            readonly property int sliceIndex: slice.index

            Rectangle {
                id: rectangle
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.preferredWidth: slice.implicitWidth
                Layout.preferredHeight: slice.implicitHeight
                color: "black"

                border.color: "darkgrey"
                border.width: 1

                ImagePlaneView {
                    id: slice
                    anchors.fill: parent
                    anchors.margins: rectangle.border.width

                    imagePlaneModel: imagePlane
                    index: slider.value
                    axis: sliceAxis

                    Component.onCompleted: update();

                    Connections {
                        target: scene
                        onStepEnd: slice.update()
                    }
                }
            }

            Slider {
                id: slider
                Layout.fillWidth: true

                minimumValue: 0
                maximumValue: slice.length > 0 ? slice.length - 1 : 0
                value: slice.length / 2
                stepSize: 1
                tickmarksEnabled: true
            }
        }
    }

    Loader {
        id: planeX
        Layout.fillWidth: true
        Layout.fillHeight: true

        sourceComponent: sliceComponent
        property int sliceAxis: 0
        readonly property int sliceIndex: item ? item.sliceIndex : 0
    }

    Loader {
        id: planeY
        Layout.fillWidth: true
        Layout.fillHeight: true

        sourceComponent: sliceComponent
        property int sliceAxis: 1
        readonly property int sliceIndex: item ? item.sliceIndex : 0
    }

    Item {
        id: info
        Layout.fillWidth: true
        Layout.fillHeight: true

        TextArea {
            anchors.fill: parent
            readOnly: true

            text: "Info:\n\n" +
                  "x: " + planeX.sliceIndex + "\n" +
                  "y: " + planeY.sliceIndex + "\n" +
                  "z: " + planeZ.sliceIndex + "\n"
        }
    }

    Loader {
        id: planeZ
        Layout.fillWidth: true
        Layout.fillHeight: true

        sourceComponent: sliceComponent
        property int sliceAxis: 2
        readonly property int sliceIndex: item ? item.sliceIndex : 0
    }
}

