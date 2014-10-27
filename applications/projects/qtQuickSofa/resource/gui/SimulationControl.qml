import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0

GroupBox {
    id: root
    title: "Simulation Control"

    property alias animateButton: animateButton

    RowLayout {
		anchors.fill: parent
        Button {
            id: animateButton
            Layout.fillWidth: true
            text: "Animate"
            checkable: true

            /*onClicked: {
                root.animateClicked(checked);
            }*/
        }
    }
}
