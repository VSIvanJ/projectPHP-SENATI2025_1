<?php

include '../PDO/Conexion.php';
include '../PDO/PeticionDAO.php';

$cliente=$_REQUEST['txtCliP'];
$contrato=$_REQUEST['txtConP'];
$detalle=$_REQUEST['txtDetP'];

//Obtener la fecha actual
$fechaPeticion=date('Y-m-d');

$rPet = new PeticionDAO();

$regPeticion = $rPet->registrarPeticion($cliente, $contrato, $detalle, $fechaPeticion);