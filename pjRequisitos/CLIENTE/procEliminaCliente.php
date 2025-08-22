<?php

include '../PDO/Conexion.php';
include '../PDO/ClienteDAO.php';

$ruc=$_POST['txtEIC'];
$idcli=$_POST['txtERC'];

$eCli = new ClienteDAO();

$elimCli = $eCli->eliminarCliente($ruc, $idcli);