<?php

include '../PDO/Conexion.php';
include '../PDO/PeticionDAO.php';

$peticion=$_POST['txtAIPet'];
$contrato=$_POST['txtACPet'];

$aPet = new PeticionDAO();

$actPet = $aPet->actualizarPeticion($peticion, $contrato);