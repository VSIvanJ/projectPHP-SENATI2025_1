<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Inicio de Sesion</title>
        <link href="Resources/css/style-loginUsuario.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>Bienvenido a SAC</h1>
        <div class="login-container">
            <h2>Iniciar Sesión:</h2>
            <form action="procLoginUsuario.php" method="POST">
                <label for="usuario">Usuario:</label>
                <input type="text" id="txtUUser" name="txtUUser" required>
                <br><br>
                <label for="clave">Contraseña:</label>
                <input type="password" id="txtCUser" name="txtCUser" required>
                <br><br>
                <button type="submit">Ingresar</button>
            </form>
        </div>
    </body>
</html>