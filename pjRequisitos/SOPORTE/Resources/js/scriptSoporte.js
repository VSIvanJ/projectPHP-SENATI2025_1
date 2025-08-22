function ConsultarSoporte() {
    idPS=$('#txtCIS').val();
    idAS=$('#txtCAS').val();
    if(idPS==='' ||idAS===''){
        $('#selectSoporte').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procConsultaSoporte.php',
        type:'POST',
        data:{txtCIS:idPS,txtCAS:idAS},
        success: function (respuesta) {
            $('#selectSoporte').html(respuesta);
        },
        error: function () {
            $('#selectSoporte').text('ERROR AL INTENTAR CONSULTAR');
        }
    });
}

/*function ActualizarSoporte() {
    IDP=$('#txtActIdPer').val();
    DNIP=$('#txtActDNIPer').val();
    CORP=$('#txtActCor').val();
    if(IDP==='' || DNIP==='' || CORP===''){
        $('#updateSoporte').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procActualizaSoporte.php',
        type:'POST',
        data:{txtActIdPer:IDP,txtActDNIPer:DNIP,txtActCor:CORP},
        success: function (respuesta2) {
            $('#updateSoporte').html(respuesta2);
        },
        error: function () {
            $('#updateSoporte').text('ERROR AL INTENTAR ACTUALIZAR');
        }
    });
}*/

function EliminarSoporte() {
    EIds=$('#txtEIS').val();
    EIdps=$('#txtEIPS').val();
    if(EIds==='' || EIdps===''){
        $('#deleteSoporte').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procEliminaSoporte.php',
        type:'POST',
        data:{txtEIS:EIds,txtEIPS:EIdps},
        success: function (respuesta3) {
            $('#deleteSoporte').html(respuesta3);
        },
        error: function () {
            $('#deleteSoporte').text('ERROR AL INTENTAR ELIMINAR');
        }
    });
}

function LimpiarConsultaS() {
    $('#txtCIS').val('');
    $('#txtCAS').val('');
    $('#selectSoporte').text('');
}

/*function LimpiarActualizarS() {
    $('#txtActIdPer').val('');
    $('#txtActDNIPer').val('');
    $('#txtActCor').val('');
    $('#updateSoporte').text('');
}*/

function LimpiarEliminarS() {
    $('#txtEIS').val('');
    $('#txtEIPS').val('');
    $('#deleteSoporte').text('');
}