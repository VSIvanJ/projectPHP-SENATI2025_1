<?php

class ContratoDAO extends Conexion{
    public function registrarContrato($cliente,$soporte,$contenido,$inicio,$fin,$monto) {
        try {
            
            $estado=1;
            // Verificar que el cliente existe
            $stmt = $this->conn->prepare("SELECT COUNT(*) FROM CLIENTE WHERE id_cliente = :id_cliente");
            $stmt->bindParam(':id_cliente', $cliente);
            $stmt->execute();
            
            if ($stmt->fetchColumn() == 0) {
                throw new Exception("El ID de cliente no existe.");
            }
            
            // Verificar que el personal de soporte existe
            $stmt = $this->conn->prepare("SELECT COUNT(*) FROM SOPORTE WHERE id_soporte = :id_soporte");
            $stmt->bindParam(':id_soporte', $soporte);
            $stmt->execute();
            
            if ($stmt->fetchColumn() == 0) {
                throw new Exception("El ID del personal de soporte no existe.");
            }
            
            $sql="INSERT INTO CONTRATO (id_cliente,id_soporte,contenido,fecha_inicio,fecha_fin,monto,estado) "
                    . "VALUES (:id_cliente,:id_soporte,:contenido,:fecha_inicio,:fecha_fin,:monto,:estado)";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':id_cliente',$cliente);
            $unir->bindParam(':id_soporte',$soporte);
            $unir->bindParam(':contenido',$contenido);
            $unir->bindParam(':fecha_inicio',$inicio);
            $unir->bindParam(':fecha_fin',$fin);
            $unir->bindParam(':monto',$monto);
            $unir->bindParam(':estado',$estado);
            $unir->execute();
            
            echo "Se registró el contrato";
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
    
    public function consultarContrato($cliente,$soporte) {
        try {
            $sql="SELECT c.*,cl.razon_social,p.nombre,p.apellido "
                    . "FROM CONTRATO c "
                    . "INNER JOIN CLIENTE cl ON c.id_cliente = cl.id_cliente "
                    . "INNER JOIN SOPORTE s ON c.id_soporte = s.id_soporte "
                    . "INNER JOIN PERSONAL p ON s.id_personal = p.id_personal "
                    . "WHERE c.id_cliente = :id_cliente AND c.id_soporte = :id_soporte";
            $unir=$this->conn->prepare($sql);
            $unir->bindParam(':id_cliente',$cliente);
            $unir->bindParam(':id_soporte',$soporte);
            $unir->execute();
            $rpta=$unir->fetch(PDO::FETCH_ASSOC);
            
            if($rpta){
                return $rpta;
            }else{
                return "El contrato no existe";
            }
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
    
    public function actualizarContrato($contrato,$cliente) {
        try {
            $estado=0;
            
            $sql="UPDATE CONTRATO SET estado = :estado WHERE id_contrato = :id_contrato AND "
                    . "id_cliente = :id_cliente";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':estado',$estado);
            $unir->bindParam(':id_contrato',$contrato);
            $unir->bindParam(':id_cliente',$cliente);
            $unir->execute();
            
            echo "Se actualizó el estado del contrato";
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
    
    public function eliminarContrato($contrato,$soporte) {
        try {
            
            $sql="DELETE FROM CONTRATO WHERE id_contrato = :id_contrato AND id_soporte = :id_soporte AND estado = 0";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':id_contrato',$contrato);
            $unir->bindParam(':id_soporte',$soporte);
            $unir->execute();
            
            if ($unir->rowCount() > 0) {
                echo "Se eliminó el contrato.";
            } else {
                echo "No se eliminó el contrato: El estado del contrato está activo";
            }
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
}