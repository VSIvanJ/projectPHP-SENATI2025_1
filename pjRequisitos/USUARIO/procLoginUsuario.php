<?php

//MANTENER SESION
session_start();

include '../PDO/Conexion.php';
include '../PDO/UsuarioDAO.php';

$user=$_POST['txtUUser'];
$clave=$_POST['txtCUser'];

$lUser = new UsuarioDAO();

$logUser = $lUser->loginUsuario($user, $clave);

if($logUser){
    // Guarda el ID del usuario
    $_SESSION['id_usuario'] = $logUser['id_usuario'];
    sleep(3);
    header("Location: ../Intranet.php");
    exit();
}