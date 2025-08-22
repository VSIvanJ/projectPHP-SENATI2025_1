<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Registro de Usuario</title>
        <link href="Resources/css/style-regUsuario.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>Formulario de Registro:</h1>
        <div class="registroTabla">
            <form action="procRegUsuario.php" method="post">
                <label for="usuario">Nombre de Usuario:</label>
                <input type="text" id="txtUser" name="txtUser">
                <br><br>
                <label for="email">Correo electrónico:</label>
                <input type="email" id="txtEmail" name="txtEmail">
                <br><br>
                <label for="clave">Contraseña:</label>
                <input type="password" id="txtContra" name="txtContra">
                <br><br>
                <label for="clave">Confirmar Contraseña:</label>
                <input type="password" id="txtConfirContra" name="txtConfirContra">
                <br><br>
                <input type="submit">
            </form>
        </div>
    </body>
</html>