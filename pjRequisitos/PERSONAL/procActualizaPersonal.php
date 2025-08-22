<?php

include '../PDO/Conexion.php';
include '../PDO/PersonalDAO.php';

$id=$_POST['txtActIdPer'];  
$dni=$_POST['txtActDNIPer'];
$correo=$_POST['txtActCor'];

$aPer = new PersonalDAO();

$actPer = $aPer->actualizarPersonal($id, $dni, $correo);