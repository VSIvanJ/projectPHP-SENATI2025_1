<?php

include '../PDO/Conexion.php';
include '../PDO/AdministradorDAO.php';

$email=$_POST['txtEmailAdmin'];
$hash=$_POST['txtClaveAdmin'];
$dni=$_POST['txtDNIAdmin'];

$adm = new AdministradorDAO();

$Actadmin = $adm->actualizarAdministrador($email, $clave, $dni);