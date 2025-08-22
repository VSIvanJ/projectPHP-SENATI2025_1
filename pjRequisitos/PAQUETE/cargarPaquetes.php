<?php

require_once __DIR__.'/../PDO/Conexion.php';
require_once __DIR__.'/../PDO/PaqueteDAO.php';

$oPaq = new PaqueteDAO();

$paquetes= $oPaq->obtenerPaquetes();

foreach ($paquetes as $paquete) {
    echo '<div class="paquete">';
    echo '<h2>' . $paquete['nombre'] . '</h2>';
    echo '<p>' . $paquete['descripcion'] . '</p>';
    echo '<p><strong>Precio: S/' . number_format($paquete['precio'],2) . '</strong></p>';
    echo '<a href="CLIENTE/frmRegCliente.php?paquete_id=' . $paquete['id_paquete'] . '" class="boton-comprar">COMPRAR</a>';
    echo '<br><br>';
    echo '</div>';
}