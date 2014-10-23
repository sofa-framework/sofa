import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0

GroupBox {
    id: root
    title: "Simulation Control"

    signal animateClicked(var checked)

    Column {
        Button {
            width: 150
            text: "Animate"
            checkable: true

            onClicked: {
                root.animateClicked(checked);
            }
        }
    }
}
