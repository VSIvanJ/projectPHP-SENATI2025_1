<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Login Administrador</title>
        <link href="Resources/css/style-login.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>Bienvenido Administrador</h1>
        <div class="login-container">
            <h2>Iniciar Sesión:</h2>
            <form action="procLoginAdministrador.php" method="POST">
                <label for="nombre">Usuario:</label>
                <input type="text" id="txtNAdmin" name="txtNAdmin" required>
                <br><br>
                <label for="clave">Contraseña:</label>
                <input type="password" id="txtCAdmin" name="txtCAdmin" required>
                <br><br>
                <button type="submit">Ingresar</button>
            </form>
        </div>
    </body>
</html>