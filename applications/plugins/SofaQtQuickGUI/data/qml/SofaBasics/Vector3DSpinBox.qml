import QtQuick 2.0
import QtQuick.Controls 1.3
import QtQuick.Layouts 1.0

GridLayout {
    id: root

    property alias  vx:                 vxSpinBox.value
    property alias  vy:                 vySpinBox.value
    property alias  vz:                 vzSpinBox.value

    function setValueFromArray(array) {
        var values = [Number(array[0]), Number(array[1]), Number(array[2])];
        if(values[0] !== values[0] || values[1] !== values[1] || values[2] !== values[2]) {
            console.error("Value is Nan");
            return;
        }

        vx = values[0];
        vy = values[1];
        vz = values[2];
    }

    property real   minimumValue:      -9999999.0
    property real   maximumValue:       9999999.0
    property int    decimals:           2
    property real   stepSize:           1

    SpinBox {
        id: vxSpinBox
        Layout.fillWidth:               true
        Layout.preferredWidth:          20
        minimumValue:                   root.minimumValue
        maximumValue:                   root.maximumValue
        decimals:                       root.decimals
        value:                          0.0
        stepSize:                       root.stepSize
    }
    SpinBox {
        id: vySpinBox
        Layout.fillWidth:               true
        Layout.preferredWidth:          20
        minimumValue:                   root.minimumValue
        maximumValue:                   root.maximumValue
        decimals:                       root.decimals
        value:                          0.0
        stepSize:                       root.stepSize
    }
    SpinBox {
        id: vzSpinBox
        Layout.fillWidth:               true
        Layout.preferredWidth:          20
        minimumValue:                   root.minimumValue
        maximumValue:                   root.maximumValue
        decimals:                       root.decimals
        value:                          0.0
        stepSize:                       root.stepSize
    }
}
