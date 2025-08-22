<?php

//MANTENER SESION
session_start();

include '../PDO/Conexion.php';
include '../PDO/AdministradorDAO.php';

$user=$_POST['txtNAdmin'];
$clave=$_POST['txtCAdmin'];

$logAdmin = new AdministradorDAO();

$sesionAdmin = $logAdmin->loginAdministrador($user, $clave);

if($sesionAdmin){
    //Guardar ID de Administrador
    $_SESSION['id_admin'] = $sesionAdmin['id_admin'];
    sleep(3);
    header("Location: ../accionesAdmin.php");
    exit();
}