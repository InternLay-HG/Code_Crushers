package com.QueLet.QueLet.Repository;

import com.QueLet.QueLet.Model.Appointment;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.relational.core.sql.In;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface AppointmentRepository extends JpaRepository<Appointment,Long> {
    @Query("SELECT a FROM Appointment a WHERE a.business.id = :businessId")
    List<Appointment> findByBusinessId(Long businessId);
    Appointment findByid(Long id);

}
