function ConsultarUsuario() {
    emailU=$('#txtConEmail').val();
    if(emailU===''){
        $('#selectUsuario').text('Ingrese el email del usuario');
        return;
    };
    
    $.ajax({
        url:'procConsultaUsuario.php',
        type:'POST',
        data:{txtConEmail:emailU},
        success: function (respuesta) {
            $('#selectUsuario').html(respuesta);
        },
        error: function () {
            $('#selectUsuario').text('ERROR AL INTENTAR CONSULTAR');
        }
    });
}

function LimpiarConsultaU() {
    $('#txtConEmail').val('');
    $('#selectUsuario').text('');
}