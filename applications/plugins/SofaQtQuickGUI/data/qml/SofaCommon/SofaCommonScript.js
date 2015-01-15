/*
function generateUiId(uiIds) {
    var uiId = 1;
    var uiIdList = uiIds.split(";");

    var previousId = 0;
    for(var i = 0; i < uiIdList.length; ++i) {
        if(0 === uiIdList[i].length)
            continue;

        uiId = Number(uiIdList[i]);

        if(previousId + 1 !== uiId) {
            uiId = previousId + 1;
            break;
        }

        previousId = uiId;
        ++uiId;
    }

    return [uiIds, uiId];
}

function addUiId(uiIds, uiId) {
    if(0 === uiId)
        return;

    if(-1 === uiIds.search(";" + uiId.toString() + ";")) {
        uiIds += uiId.toString() + ";";
        update();
    }

    return uiIds;
}

function removeUiId(uiIds, uiId) {
    if(0 === uiId)
        return;

    uiIds = uiIds.replace(";" + uiId.toString() + ";", ";");

    application.clearSettingGroup("ui_" + uiId.toString());

    return uiIds;
}

function replaceUiId(uiIds, previousUiId, uiId) {
    if(0 === uiId)
        return;

    uiIds = uiIds.replace(";" + uiId.toString() + ";", ";");

    if(-1 === uiIds.search(";" + uiId.toString() + ";")) {
        uiIds += uiId.toString() + ";";
        update();
    }

    return uiIds;
}

function update(uiIds) {
    var uiIdList = uiIds.split(";");
    uiIdList.sort(function(a, b) {return Number(a) - Number(b);});

    uiIds = ";";
    for(var i = 0; i < uiIdList.length; ++i)
        if(0 !== uiIdList[i].length)
            uiIds += uiIdList[i] + ";";

    return uiIds;
}
*/
