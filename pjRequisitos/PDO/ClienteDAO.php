<?php

class ClienteDAO extends Conexion {
    public function registrarCliente($iduser,$idpaquete,$rsocial,$ruc,$depart,$direc) {
        try {
            $sql="INSERT INTO CLIENTE (id_usuario,id_paquete,razon_social,ruc,departamento,direccion) "
            . "VALUES (:id_usuario,:id_paquete,:razon_social,:ruc,:departamento,:direccion)";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':id_usuario',$iduser);
            $unir->bindParam(':id_paquete',$idpaquete);
            $unir->bindParam(':razon_social',$rsocial);
            $unir->bindParam(':ruc',$ruc);
            $unir->bindParam(':departamento',$depart);
            $unir->bindParam(':direccion',$direc);
            $unir->execute();
            
            echo "Se registró la compra";
            
        } catch (Exception $ex) {
            echo "Error al intentar registrar: ".$ex;
        }
    }
    
    public function consultarCliente($nruc) {
        try {
            $sql="SELECT c.*,p.nombre FROM CLIENTE c "
                    . "INNER JOIN PAQUETE p ON c.id_paquete = p.id_paquete "
                    . "WHERE ruc=:ruc";
            $unir=$this->conn->prepare($sql);
            $unir->bindParam(':ruc',$nruc);
            $unir->execute();
            $rpta=$unir->fetch(PDO::FETCH_ASSOC);
            
            if($rpta){
                return $rpta;
            }else{
                echo "No existe";
            }
            
        } catch (Exception $ex) {
            echo "Error al consultar: ".$ex;
        }
    }
    
    public function actualizarCliente($ruc,$direccion) {
        try {
            $sql="UPDATE CLIENTE SET direccion=:direccion WHERE ruc=:ruc";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':ruc',$ruc);
            $unir->bindParam(':direccion',$direccion);
            $unir->execute();
            
            echo "Se actualizaron los datos del cliente";
            
        } catch (Exception $ex) {
            echo "Error al intentar actualizar: ".$ex;
        }
    }
    
    public function eliminarCliente($ruc,$idcli) {
        try {
            $sql="DELETE FROM CLIENTE WHERE ruc=:ruc AND id_cliente=:id_cliente";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':ruc',$ruc);
            $unir->bindParam(':id_cliente',$idcli);
            $unir->execute();
            
            echo "Se eliminó al cliente";
            
        } catch (Exception $ex) {
            echo "Error al intentar eliminar: ".$ex;
        }
    }
}
