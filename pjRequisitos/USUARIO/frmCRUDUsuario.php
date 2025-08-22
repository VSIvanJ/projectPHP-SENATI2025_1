<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Acciones Usuario</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js" crossorigin="anonymous"></script>
        <script src="Resources/js/scriptUsuario.js" type="text/javascript"></script>
        <link href="../Resources/css/baseCRUD.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>ACCIONES PARA USUARIO:</h1>
        <div class="crudTabla">
            <h2>Consultar Usuario:</h2>
            <label for="nombre">Ingrese email:</label>
            <input type="email" id="txtConEmail" name="txtConEmail" required>
            <br><br>
            <button onclick="ConsultarUsuario()">CONSULTAR</button>
            <button onclick="LimpiarConsultaU()">LIMPIAR</button>
            <br><br>
            <div id="selectUsuario"></div>
        </div>
    </body>
</html>