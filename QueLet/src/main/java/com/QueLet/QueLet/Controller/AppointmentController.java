package com.QueLet.QueLet.Controller;

import com.QueLet.QueLet.Model.Appointment;
import com.QueLet.QueLet.Service.AppointmentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

@RestController
@RequestMapping("/api/appointments")
public class AppointmentController {

    @Autowired
    private AppointmentService appointmentService;

    @PostMapping("/add")
    public Appointment addAppointment(
            @RequestParam Long customerId,
            @RequestParam Long businessId,
            @RequestParam String appointmentTime) {

        LocalDateTime time = LocalDateTime.parse(appointmentTime);
        return appointmentService.addAppointment(customerId, businessId, time);
    }
    @GetMapping("/business/{businessId}")
    public List<Appointment> getAppointmentsByBusinessId(@PathVariable Long businessId) {
        return appointmentService.findAppointmentsByBusinessId(businessId);
    }
    @PostMapping("/complete/{appointmentId}")
    public Appointment completeAppointment(@PathVariable Long appointmentId){
        return appointmentService.completeAppointment(appointmentId);
    }

}