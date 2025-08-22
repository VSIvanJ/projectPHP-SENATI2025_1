<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Crear cuenta Soporte</title>
        <link href="Resources/css/style-regEmp.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>Crear cuenta a Personal de Soporte:</h1>
        <div class="registroTabla">
            <form action="procRegEmpleado.php" method="post">
                <label for="id_soporte">ID Personal de Soporte:</label>
                <input type="text" id="txtIDSop" name="txtIDSop">
                <br><br>
                <label for="usuario">Nombre de Usuario:</label>
                <input type="text" id="txtUserEmp" name="txtUserEmp">
                <br><br>
                <label for="clave">Contraseña:</label>
                <input type="password" id="txtContraEmp" name="txtContraEmp">
                <br><br>
                <label for="clave">Confirmar Contraseña:</label>
                <input type="password" id="txtConfirCE" name="txtConfirCE">
                <br><br>
                <input type="submit">
            </form>
        </div>
    </body>
</html>
