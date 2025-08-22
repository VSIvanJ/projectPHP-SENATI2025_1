<?php

include '../PDO/Conexion.php';
include '../PDO/PeticionDAO.php';

$peticion=$_POST['txtBIPet'];

$ePet = new PeticionDAO();

$elimPet = $ePet->eliminarPeticion($peticion);