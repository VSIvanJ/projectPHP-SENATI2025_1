<?php

include '../PDO/Conexion.php';
include '../PDO/PersonalDAO.php';

$id=$_POST['txtEIDPer'];
$dni=$_POST['txtEDniPer'];

$ePer = new PersonalDAO();

$elimPer = $ePer->eliminarPersonal($id, $dni);