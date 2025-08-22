function ConsultarPersonal() {
    dniP=$('#txtConDniPer').val();
    if(dniP===''){
        $('#selectPersonal').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procConsultaPersonal.php',
        type:'POST',
        data:{txtConDniPer:dniP},
        success: function (respuesta) {
            $('#selectPersonal').html(respuesta);
        },
        error: function () {
            $('#selectPersonal').text('ERROR AL INTENTAR CONSULTAR');
        }
    });
}

function ActualizarPersonal() {
    IDP=$('#txtActIdPer').val();
    DNIP=$('#txtActDNIPer').val();
    CORP=$('#txtActCor').val();
    if(IDP==='' || DNIP==='' || CORP===''){
        $('#updatePersonal').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procActualizaPersonal.php',
        type:'POST',
        data:{txtActIdPer:IDP,txtActDNIPer:DNIP,txtActCor:CORP},
        success: function (respuesta2) {
            $('#updatePersonal').html(respuesta2);
        },
        error: function () {
            $('#updatePersonal').text('ERROR AL INTENTAR ACTUALIZAR');
        }
    });
}

function EliminarPersonal() {
    IDEPer=$('#txtEIDPer').val();
    DNIEPer=$('#txtEDniPer').val();
    if(IDEPer==='' || DNIEPer===''){
        $('#deletePersonal').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procEliminaPersonal.php',
        type:'POST',
        data:{txtEIDPer:IDEPer,txtEDniPer:DNIEPer},
        success: function (respuesta3) {
            $('#deletePersonal').html(respuesta3);
        },
        error: function () {
            $('#deletePersonal').text('ERROR AL INTENTAR ELIMINAR');
        }
    });
}

function LimpiarConsultaP() {
    $('#txtConDniPer').val('');
    $('#selectPersonal').text('');
}

function LimpiarActualizarP() {
    $('#txtActIdPer').val('');
    $('#txtActDNIPer').val('');
    $('#txtActCor').val('');
    $('#updatePersonal').text('');
}

function LimpiarEliminarP() {
    $('#txtEIDPer').val('');
    $('#txtEDniPer').val('');
    $('#deletePersonal').text('');
}