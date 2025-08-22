<?php

class PeticionDAO extends Conexion{
    public function registrarPeticion($cliente,$contrato,$detalle,$fechaPeticion) {
        try {
            $estado=1;
                       
            // Verificar que el cliente y contrato existen
            $stmt = $this->conn->prepare("SELECT COUNT(*) FROM CONTRATO WHERE id_cliente = :id_cliente "
                    . "AND id_contrato = :id_contrato");
            $stmt->bindParam(':id_cliente', $cliente);
            $stmt->bindParam(':id_contrato', $contrato);
            $stmt->execute();
            
            if ($stmt->fetchColumn() == 0) {
                throw new Exception("El ID de cliente y/o ID de contrato no existe.");
            }
                        
            $sql="INSERT INTO PETICION (id_cliente, id_contrato, detalle, fecha_peticion, estado) VALUES "
                    . "(:id_cliente, :id_contrato, :detalle, :fecha_peticion, :estado)";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':id_cliente',$cliente);
            $unir->bindParam(':id_contrato',$contrato);
            $unir->bindParam(':detalle',$detalle);
            $unir->bindParam(':fecha_peticion',$fechaPeticion);
            $unir->bindParam(':estado',$estado);
            $unir->execute();
            
            echo "Se envió la petición correctamente. Se le enviará un email indicando la fecha de atención";
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
    
    public function consultarPeticion($cliente,$contrato) {
        try {
            $sql="SELECT * FROM PETICION WHERE id_cliente = :id_cliente AND id_contrato=:id_contrato";
            $unir=$this->conn->prepare($sql);
            $unir->bindParam(':id_cliente',$cliente);
            $unir->bindParam(':id_contrato',$contrato);
            $unir->execute();
            $rpta=$unir->fetch(PDO::FETCH_ASSOC);
            
            if($rpta){
                return $rpta;
            }else{
                return "No se encontró ninguna petición";
            }
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
    
    public function actualizarPeticion($peticion,$contrato) {
        try {
            $estado=0;
            
            $sql="UPDATE PETICION SET estado = :estado WHERE id_peticion = :id_peticion AND "
                    . "id_contrato = :id_contrato";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':id_peticion',$peticion);
            $unir->bindParam(':id_contrato',$contrato);
            $unir->bindParam(':estado',$estado);
            $unir->execute();
            
            echo "Se actualizó el estado de la petición";
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
    
    public function eliminarPeticion($peticion) {
        try {
            
            $sql="DELETE FROM PETICION WHERE id_peticion = :id_peticion AND estado = 0";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':id_peticion',$peticion);
            $unir->execute();
            
            if ($unir->rowCount() > 0) {
                echo "Se eliminó la petición.";
            } else {
                echo "No se eliminó la petición: El estado de petición sigue activo";
            }
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
}
