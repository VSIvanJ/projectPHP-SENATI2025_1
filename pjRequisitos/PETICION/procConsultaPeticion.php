<?php

include '../PDO/Conexion.php';
include '../PDO/PeticionDAO.php';

$cliente=$_POST['txtCClP'];
$contrato=$_POST['txtCCP'];

$cPet = new PeticionDAO();

$consulPet = $cPet->consultarPeticion($cliente, $contrato);

if($consulPet){
    echo "RESULTADO:<br>";
    echo "================== <br>";
    echo "ID Petición: ".$consulPet['id_peticion']."<br>";
    echo "ID Cliente: ".$consulPet['id_cliente']."<br>";
    echo "ID Contrato: ".$consulPet['id_contrato']."<br>";
    echo "Detalle: ".$consulPet['detalle']."<br>";
    echo "Fecha de petición: ".$consulPet['fecha_peticion']."<br>";
    echo "Estado: ".$consulPet['estado']."<br>";
}
