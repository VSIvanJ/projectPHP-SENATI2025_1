<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Actualizar Datos</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js" crossorigin="anonymous"></script>
        <script src="Resources/js/scriptAdmin.js" type="text/javascript"></script>
        <link href="Resources/css/style-actualizar.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>ACCIONES DE ADMINISTRADOR:</h1>
        <div class="crudTabla">
            <h2>Actualizar Email:</h2>
            <label for="email">Nuevo email:</label>
            <input type="email" id="txtEmailAdmin" name="txtEmailAdmin" required>
            <br><br>
            <label for="clave">Ingrese contraseña:</label>
            <input type="password" id="txtClaveAdmin" name="txtClaveAdmin" required>
            <br><br>
            <label for="dni">Ingrese NºDNI:</label>
            <input type="text" id="txtDNIAdmin" name="txtDNIAdmin" minlength="8" maxlength="8" required>
            <br><br>
            <button onclick="ActualizarEmail()">ACTUALIZAR</button>
            <button onclick="LimpiarCampos()">LIMPIAR</button>
            <br><br>
            <div id="updateAdmin"></div>
        </div>
    </body>
</html>