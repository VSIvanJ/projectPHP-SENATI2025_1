<?php

include '../PDO/Conexion.php';
include '../PDO/UsuarioDAO.php';

$email=$_POST['txtConEmail'];

$cUser = new UsuarioDAO();

$consulUsuario=$cUser->consultarUsuario($email);

if($consulUsuario){
    echo "RESULTADO:<br>";
    echo "================== <br>";
    echo "ID: ".$consulUsuario['id_usuario']."<br>";
    echo "Usuario: ".$consulUsuario['usuario']."<br>";
    echo "Email: ".$consulUsuario['email']."<br>";
    echo "Contraseña: ".$consulUsuario['clave']."<br>";
}