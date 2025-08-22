<?php

include '../PDO/Conexion.php';
include '../PDO/UsuarioDAO.php';

$user=$_REQUEST['txtUser'];
$email=$_REQUEST['txtEmail'];
$clave=$_REQUEST['txtContra'];
$validar=$_REQUEST['txtConfirContra'];

if($clave==$validar){
    //Cifrado de contraseña
    $hash= password_hash($clave, PASSWORD_DEFAULT);
    
    sleep(2);
    $u = new UsuarioDAO();
    $rUsuario = $u->registrarUsuario($user, $email, $hash);
}else{
    echo "Las contraseñas ingresadas no coinciden...Por favor, vuelva a intentarlo";
}