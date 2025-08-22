<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Acciones Personal</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js" crossorigin="anonymous"></script>
        <script src="Resources/js/scriptPersonal.js" type="text/javascript"></script>
        <link href="../Resources/css/baseCRUD.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>ACCIONES DE PERSONAL</h1>
        <div class="crudTabla">
            <h2>Consultar personal:</h2>
            <label for="dni">Nº DNI:</label>
            <input type="text" id="txtConDniPer" name="txtConDniPer" required minlength="8" maxlength="8">
            <br><br>
            <button onclick="ConsultarPersonal()">CONSULTAR</button>
            <button onclick="LimpiarConsultaP()">LIMPIAR</button>
            <br><br>
            <div id="selectPersonal"></div>
        </div>
        <br><br>
        <div class="crudTabla">
            <h2>Actualizar correo:</h2>
            <label for="id_personal">ID Personal:</label>
            <input type="text" id="txtActIdPer" name="txtActIdPer" required>
            <br><br>
            <label for="dni">Nº DNI:</label>
            <input type="text" id="txtActDNIPer" name="txtActDNIPer" required minlength="8" maxlength="8">
            <br><br>
            <label for="correo">Nuevo Correo:</label><br>
            <input type="email" id="txtActCor" name="txtActCor" required>
            <br><br>
            <button onclick="ActualizarPersonal()">ACTUALIZAR</button>
            <button onclick="LimpiarActualizarP()">LIMPIAR</button>
            <br><br>
            <div id="updatePersonal"></div>
        </div>
        <br><br>
        <div class="crudTabla">
            <h2>Eliminar personal:</h2>
            <label for="id_personal">ID Personal:</label>
            <input type="text" id="txtEIDPer" name="txtEIDPer" required>
            <br><br>
            <label for="dni">Nº DNI:</label>
            <input type="text" id="txtEDniPer" name="txtEDniPer" required minlength="8" maxlength="8">
            <br><br>
            <button onclick="EliminarPersonal()">ELIMINAR</button>
            <button onclick="LimpiarEliminarP()">LIMPIAR</button>
            <br><br>
            <div id="deletePersonal"></div>
        </div>
    </body>
</html>